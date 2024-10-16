import math
from typing import Optional, NamedTuple, Protocol, overload

from torch import BoolTensor, FloatTensor, LongTensor, Tensor, inference_mode, nn
from torch.nn import Linear

from .t5_common import T5Config
from .t5_decoder import T5DecoderStack
from .t5_encoder import T5EncoderStack

####
#### T5 label preparation
####


def labels_to_decoder_input_ids(
    labels: LongTensor,
    pad_token_id: int,
    decoder_start_token_id: int,
    label_ignore_index: int,
) -> LongTensor:
    # splice a decoder_start into the start of the sequence
    # note: we can lose final token of max-length prompts here (unless we planned ahead)
    shifted_labels = labels.detach().roll(1)
    shifted_labels[..., 0] = decoder_start_token_id
    # the token -100 (which loss ignores) can't be embedded (it's negative), so we replace it with PAD
    shifted_labels.masked_fill_(shifted_labels == label_ignore_index, pad_token_id)
    return shifted_labels


def label_mask_to_decoder_mask(label_mask: Tensor) -> LongTensor:
    shifted_label_mask = label_mask.detach().roll(1)
    # reveal decoder_start_token_id
    shifted_label_mask[..., 0] = 1
    return shifted_label_mask


####
#### T5 decoding protocols
####


class LogitsAndKV(NamedTuple):
    logits: FloatTensor
    # permit model to refrain from returning kv, if it retains reference via closure rather than recurrence
    kv: Optional[FloatTensor]


class Decode(Protocol):
    @overload
    def __call__(
        x: LongTensor,
        **opt_fwd_args,
    ) -> FloatTensor: ...

    @overload
    def __call__(
        x: LongTensor,
        kv: Optional[FloatTensor],
        curr_ctx_len: int,
        generate_prompt_logits: bool,
        cache: bool,
        use_prealloc_kv: bool,
        **opt_fwd_args,
    ) -> LogitsAndKV: ...

    @staticmethod
    def __call__(
        x: LongTensor,
        kv: Optional[FloatTensor] = None,
        curr_ctx_len=0,
        generate_prompt_logits=False,  # if False, returns only the last token's logits. otherwise returns all tokens' logits
        cache=True,
        use_prealloc_kv=True,
        **opt_fwd_args,
    ) -> FloatTensor | LogitsAndKV: ...


class DecoderAndPrompt(NamedTuple):
    decode: Decode
    decoder_start: LongTensor


####
#### T5 model
####


class T5(nn.Module):
    config: T5Config
    lm_head: Linear
    encoder: T5EncoderStack
    decoder: T5DecoderStack
    param_count: int
    lm_head_param_count: int

    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.encoder = T5EncoderStack(config)
        self.decoder = T5DecoderStack(config)
        self.lm_head = Linear(
            in_features=config.hidden_dim,
            out_features=config.vocab_size,
            bias=False,
            dtype=config.linear_weight_dtype,
        )
        self.lm_head_param_count = self.config.hidden_dim * self.config.vocab_size
        self.param_count = self.lm_head_param_count + self.encoder.param_count + self.decoder.param_count

    @classmethod
    def to_pydantic_config(cls, user_config: dict) -> T5Config:
        return T5Config(**user_config)

    def forward(
        self,
        encoder_input_ids: LongTensor,
        decoder_input_ids: LongTensor,
        encoder_input_mask: Optional[BoolTensor] = None,
        decoder_input_mask: Optional[BoolTensor] = None,
        decoder_cross_mask: Optional[BoolTensor] = None,
    ) -> FloatTensor:
        encoding: FloatTensor = self.encoder(encoder_input_ids, input_mask=encoder_input_mask)
        decoder_input_embeds = self.encoder.vocab_embed(decoder_input_ids.flatten(end_dim=-2))
        decoder_out: FloatTensor = self.decoder(
            decoder_input_embeds,
            encoding,
            input_mask=decoder_input_mask,
            cross_mask=decoder_cross_mask,
        )
        logits: FloatTensor = self.lm_head(decoder_out)
        return logits

    @inference_mode()
    def make_bound_forward(
        self,
        encoder_input_ids: LongTensor,
        encoder_input_mask: Optional[BoolTensor] = None,
    ) -> DecoderAndPrompt:
        encoding: FloatTensor = self.encoder(encoder_input_ids, input_mask=encoder_input_mask)
        batch_size: int = encoder_input_ids.size(0)
        decoder_start: LongTensor = encoder_input_ids.new_full(
            (batch_size, 1), fill_value=self.config.decoder_start_token_id
        )
        self_past_kv: Optional[FloatTensor] = None
        cross_kv: Optional[FloatTensor] = None

        def ensure_cache_inited() -> None:
            nonlocal cross_kv, self_past_kv
            assert (self_past_kv is None) == (cross_kv is None)
            if self_past_kv is None:
                self_past_kv = self.decoder.get_kv_cache(
                    batch_size=batch_size,
                    length=None,
                    dtype=encoding.dtype,
                    device=encoding.device,
                )
                cross_kv = self.decoder.get_cross_kv(encoding)

        def decode(
            x: LongTensor,
            kv: Optional[FloatTensor] = None,
            curr_ctx_len=0,
            generate_prompt_logits=False,
            cache=True,
            use_prealloc_kv=True,
            **opt_fwd_args,
        ) -> FloatTensor | LogitsAndKV:
            nonlocal cross_kv, self_past_kv
            assert kv is None, "received non-None kv, despite managing access to kv via closure rather than recurrence"
            kwargs = {}
            if cache:
                assert use_prealloc_kv, "haven't implemented support for late-alloc kv"
                ensure_cache_inited()
                cache_kwargs = {
                    "self_past_kv": self_past_kv[:, :, :, :, : curr_ctx_len + 1, :],
                    "cross_kv": cross_kv,
                }
                kwargs = {**kwargs, **cache_kwargs}
                x = x[:, -1:]
            x_emb: FloatTensor = self.encoder.vocab_embed(x.flatten(end_dim=-2))
            decoder_out: FloatTensor = self.decoder.forward(
                input_embeds=x_emb, encoding=encoding, input_mask=None, encoding_mask=encoder_input_mask, **kwargs
            )
            logits: FloatTensor = self.lm_head(decoder_out)
            if not generate_prompt_logits:
                logits = logits[..., -1, :].unsqueeze(1)
            if cache:
                return LogitsAndKV(logits=logits, kv=None)
            return logits

        return DecoderAndPrompt(decode, decoder_start)

    def flop_count_per_sequence(self, input_ids_len: int, labels_len: int) -> int:
        encoder_flos = self.encoder.flop_count_per_sequence(input_ids_len, labels_len)
        # deliberately counting the encoder.vocab_embed(labels) as 0 FLOs
        decoder_flos = self.decoder.flop_count_per_sequence(input_ids_len, labels_len)

        # 2x due to mul+add, 3x due to fwd+bwd
        lm_head_flos = 6 * labels_len * self.lm_head_param_count

        return encoder_flos + decoder_flos + lm_head_flos

    def init_weights(self) -> None:
        nn.init.normal_(self.lm_head.weight, std=1 / math.sqrt(self.config.hidden_dim))
        self.encoder.init_weights()
        self.decoder.init_weights()
