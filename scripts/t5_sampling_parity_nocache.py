from dataclasses import dataclass, field
from einops import rearrange
from functools import partial
from typing import Dict, Generator, List, OrderedDict, Set

import torch
from torch import BoolTensor, FloatTensor, LongTensor, Tensor, inference_mode
from torch.amp import autocast
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList, TensorType
from transformers.generation.streamers import BaseStreamer
from transformers.models.t5 import T5ForConditionalGeneration, T5TokenizerFast
from transformers.models.t5.configuration_t5 import T5Config as T5ConfigHF
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils.generic import TensorType

from nai_t5 import T5, T5Config, hf_to_based_t5_state, to_based_config
from nai_t5.sampling import BoundDecode, MakeLogitGenerator, generate_greedy, generate_until


@dataclass
class StopOnToken(StoppingCriteria):
    stop_tokens: Set[int]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids.squeeze()[-1].item() in self.stop_tokens


@dataclass
class TokenStreamer(BaseStreamer):
    tokenizer: T5TokenizerFast
    print_tokens: bool = False
    ids: List[int] = field(default_factory=list, init=False)
    tokens: List[str] = field(default_factory=list, init=False)

    def put(self, value: LongTensor) -> None:
        token_id: int = value.squeeze().item()
        token_str: int = self.tokenizer.convert_ids_to_tokens(token_id)
        if self.print_tokens:
            print(token_str, flush=True, end="")
        self.ids.append(token_id)
        self.tokens.append(token_str)

    def end(self):
        """Function that is called by `.generate()` to signal the end of generation"""
        if self.print_tokens:
            print("", flush=True)


def main():
    device = torch.device("cuda")
    hf_model_name = "google/t5-v1_1-small"
    hf_config: T5ConfigHF = T5ConfigHF.from_pretrained(hf_model_name)
    hf_tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained(hf_model_name, legacy=False)
    hf_t5 = T5ForConditionalGeneration.from_pretrained(hf_model_name).eval()

    my_config: T5Config = to_based_config(hf_config, n_tokens=hf_tokenizer.model_max_length)
    my_t5 = T5(my_config).eval()

    input_encoding: BatchEncoding = hf_tokenizer(
        "Today is a fine <extra_id_0> on which to walk my <extra_id_1> in the park.",
        return_tensors=TensorType.PYTORCH,
        # for some reason, ending the input with </s> results in a saner answer on t5 small.
        # t5 large is fine without </s>; we should generally prefer False, but it's nice to debug on small
        add_special_tokens=True,
    ).to(device)
    label_encoding: BatchEncoding = hf_tokenizer(
        "<pad><extra_id_0> day<extra_id_1> dog<extra_id_2>",
        return_tensors=TensorType.PYTORCH,
        add_special_tokens=False,
    ).to(device)
    input_ids = input_encoding["input_ids"]
    input_ids_mask = input_encoding["attention_mask"]
    expected_output = label_encoding["input_ids"]
    mask2: int = hf_tokenizer.convert_tokens_to_ids("<extra_id_2>")

    hf_state: OrderedDict[str, Tensor] = hf_t5.state_dict()
    converted_enc_state: Dict[str, Tensor] = hf_to_based_t5_state(hf_state, my_config)
    my_t5.load_state_dict(converted_enc_state)
    my_t5.to(device)

    max_new_tokens = 16
    seed = 42

    run_hf_baseline = False
    if run_hf_baseline:
        hf_t5.to(device)

        streamer_factory = lambda: TokenStreamer(tokenizer=hf_tokenizer, print_tokens=True)
        streamer: TokenStreamer = streamer_factory()
        stopping_criteria = StoppingCriteriaList([StopOnToken(set((mask2, hf_t5.config.eos_token_id)))])

        with inference_mode(), autocast(device_type=device.type, dtype=torch.float16):
            # seed the random, so that we can parity-test things like dropout (if enabled)
            torch.manual_seed(seed)
            generate_out: LongTensor = hf_t5.generate(
                generation_config=GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    decoder_start_token_id=hf_t5.config.decoder_start_token_id,
                    eos_token_id=hf_t5.config.eos_token_id,
                    pad_token_id=hf_t5.config.pad_token_id,
                ),
                input_ids=input_ids,
                attention_mask=input_ids_mask,
                streamer=streamer,
                stopping_criteria=stopping_criteria,
            )
        assert generate_out.squeeze().equal(
            expected_output.squeeze()
        ), "baseline HF generation no longer outputs the fixture we originally documented"
        # gen_decoded_s: str = hf_tokenizer.decode(generate_out.squeeze())
        gen_decoded: List[str] = hf_tokenizer.convert_ids_to_tokens(generate_out.squeeze())
    else:
        generate_out: LongTensor = expected_output

    ####
    #### and now we try to do the same greedy search as generate, manually:
    ####
    torch.manual_seed(seed)

    batch_size = 1
    use_based = True
    with inference_mode(), autocast(device_type=device.type, dtype=torch.float16):
        if use_based:
            encoding: FloatTensor = my_t5.encoder(input_ids, input_mask=input_ids_mask.bool())

            cross_mask = rearrange(input_ids_mask.bool(), "b k -> b 1 k")
            cross_mask = cross_mask.expand(-1, max_new_tokens, -1)

            decoder_input_mask = encoding.new_ones((batch_size, max_new_tokens), dtype=torch.bool)
        else:
            hf_t5.to(device)
            encoding: FloatTensor = hf_t5.encoder.forward(
                input_ids=input_ids, attention_mask=input_ids_mask
            ).last_hidden_state

    def decode(in_tokens: LongTensor, encoding: FloatTensor, input_ids_mask: BoolTensor) -> FloatTensor:
        with inference_mode(), autocast(device_type=device.type, dtype=torch.float16):
            if use_based:
                decoder_input_embeds: FloatTensor = my_t5.encoder.vocab_embed(in_tokens)
                input_mask: BoolTensor = decoder_input_mask[:, : in_tokens.size(-1)]
                cross_mask_: BoolTensor = cross_mask[:, : in_tokens.size(-1), :]
                decoder_out: FloatTensor = my_t5.decoder(
                    decoder_input_embeds,
                    encoding,
                    input_mask=input_mask,
                    cross_mask=cross_mask_,
                )
                logits: FloatTensor = my_t5.lm_head(decoder_out)
            else:
                hidden_states: FloatTensor = hf_t5.decoder.forward(
                    input_ids=in_tokens,
                    attention_mask=None,
                    encoder_hidden_states=encoding,
                    encoder_attention_mask=input_ids_mask,
                ).last_hidden_state
                logits: FloatTensor = hf_t5.lm_head(hidden_states)
            return logits

    bound_decode: BoundDecode = partial(decode, encoding=encoding, input_ids_mask=input_ids_mask)
    make_gen: MakeLogitGenerator = partial(generate_greedy, decode=bound_decode)

    gen: Generator[LongTensor, None, LongTensor] = generate_until(
        make_gen=make_gen,
        device=device,
        batch_size=batch_size,
        max_tokens=max_new_tokens,
        stop_tokens={mask2, hf_t5.config.eos_token_id},
        decoder_start_token_id=hf_t5.config.decoder_start_token_id,
        pad_start_token_id=hf_t5.config.pad_token_id,
    )
    my_acc: List[int] = []
    for ix, (tok_t, expected_tok) in enumerate(zip(gen, expected_output[0, 1:].tolist())):
        token_id: int = tok_t.cpu().squeeze().item()
        token_str: int = hf_tokenizer.convert_ids_to_tokens(token_id)
        print(token_str, flush=True, end="")
        my_acc.append(token_id)
        assert (
            token_id == expected_tok
        ), f"our sampler's output (sampling from {'NAI' if use_based else 'HF'}) diverged from HF output at prediction index {ix}. Expected token {expected_tok} ({hf_tokenizer.convert_ids_to_tokens(expected_tok)}). actual: {token_id} ({token_str})"
    else:
        print("", flush=True)
        assert (
            my_acc == generate_out[0, 1:].tolist()
        ), f"our sampler's output (sampling from {'NAI' if use_based else 'HF'}) diverged from HF output"
    pass  # somewhere to put your breakpoint


if __name__ == "__main__":
    main()
