import math
from typing import Optional, Protocol, TYPE_CHECKING, NamedTuple, Any
from itertools import chain
from functools import partial

import torch
from einops import rearrange
from torch import BoolTensor, FloatTensor, IntTensor, LongTensor, nn
from torch.nn import Linear, Embedding
from torch.nn.functional import scaled_dot_product_attention

from .t5_common import (
    RMSNormCast,
    T5AttnImpl,
    T5GEGLUFFN,
    T5Config,
    T5RelativeAttentionBias,
    T5ReLUFFN,
    ActAndResidual,
    flash_attention_flops,
    get_ffn_factory,
)
from .flex.flex_utils import ScoreMod, MaskMod

if TYPE_CHECKING:
    # avoid runtime import to prevent explosion on older PyTorch
    from torch.nn.attention.flex_attention import BlockMask
else:
    BlockMask = _mask_mod_signature = Any

####
#### T5 encoder self-attention
####


class T5EncoderSelfAttention(nn.Module):
    qkv_proj: Linear
    o_proj: Linear
    head_dim: int
    scale: float
    dropout: float
    config: T5Config

    def __init__(self, config: T5Config) -> None:
        super().__init__()
        assert config.n_head == config.kv_heads, "Q and KV heads must be equal; GQA not implemented yet."
        self.head_dim = config.head_dim
        qkv_heads: int = config.n_head + config.kv_heads * 2
        self.scale = self.head_dim**-0.5 if config.scale_qk else 1.0
        self.dropout = config.dropout
        self.qkv_proj = Linear(
            in_features=config.hidden_dim,
            out_features=config.head_dim * qkv_heads,
            bias=False,
            dtype=config.linear_weight_dtype,
        )
        self.o_proj = Linear(
            in_features=config.head_dim * config.n_head,
            out_features=config.hidden_dim,
            bias=False,
            dtype=config.linear_weight_dtype,
        )
        self.config = config

    def forward(
        self,
        x: FloatTensor,
        position_bias: FloatTensor,
        mask: Optional[BoolTensor] = None,
    ) -> FloatTensor:
        qkv: FloatTensor = self.qkv_proj(x)
        q, k, v = rearrange(
            qkv, "batch seq (proj heads head_dim) -> proj batch heads seq head_dim", proj=3, head_dim=self.head_dim
        ).unbind()
        position_bias = position_bias.type_as(q)
        # TODO: if training, then learn scales for Q and K as a proxy for learning rate
        if mask is not None:
            assert mask.ndim == 4, "Expected [batch, heads, q, k] attention mask"
            position_bias = position_bias.where(mask, -1e4)
        a = scaled_dot_product_attention(
            q,
            k,
            v,
            # fused kernel requires last dimension of input to have stride 1.
            attn_mask=position_bias.contiguous(),
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )
        a = rearrange(a, "batch heads seq head_dim -> batch seq (heads head_dim)")
        o = self.o_proj(a)
        return o

    def init_weights(self):
        nn.init.normal_(self.qkv_proj.weight, std=1 / math.sqrt(self.config.hidden_dim))
        nn.init.normal_(self.o_proj.weight, std=1 / math.sqrt(self.config.hidden_dim * self.config.num_layers))

class T5EncoderSelfAttentionFlex(nn.Module):
    score_mod: Optional[ScoreMod] = None
    qkv_proj: Linear
    o_proj: Linear
    head_dim: int
    scale: float
    dropout: float
    config: T5Config

    def __init__(self, config: T5Config) -> None:
        super().__init__()
        assert config.n_head == config.kv_heads, "Q and KV heads must be equal; GQA not implemented yet."
        self.config = config
        self.head_dim = config.head_dim
        qkv_heads: int = config.n_head + config.kv_heads * 2
        self.scale = self.head_dim**-0.5 if config.scale_qk else 1.0
        self.dropout = config.dropout
        # we don't have pos emb weights at the time of construction, so are not ready to bind the score_mod
        self.score_mod = None
        self.qkv_proj = Linear(
            in_features=config.hidden_dim,
            out_features=config.head_dim * qkv_heads,
            bias=False,
            dtype=config.linear_weight_dtype,
        )
        self.o_proj = Linear(
            in_features=config.head_dim * config.n_head,
            out_features=config.hidden_dim,
            bias=False,
            dtype=config.linear_weight_dtype,
        )
        self.config = config

    @staticmethod
    def make_score_mod(pos_bias: FloatTensor) -> ScoreMod:
        """
        Uses basically none of the functionality of flex attention but outperforms SDPA cuDNN/cutlassF
        and is also the fastest flex attention score_mod I was able to concoct
        https://github.com/pytorch/pytorch/issues/138493
        """
        def score_mod(
            score: FloatTensor,
            b: IntTensor,
            h: IntTensor,
            q_idx: IntTensor,
            kv_idx: IntTensor,
        ) -> FloatTensor:
            return score + pos_bias[h, q_idx, kv_idx]
        return score_mod

    @staticmethod
    def make_mask_mod(mask: BoolTensor) -> MaskMod:
        def mask_mod(
            batch: IntTensor,
            head: IntTensor,
            q_idx: IntTensor,
            kv_idx: IntTensor,
        ) -> BoolTensor:
            return mask[batch, kv_idx] & mask[batch, q_idx]
        return mask_mod

    def forward(
        self,
        x: FloatTensor,
        block_mask: Optional[BlockMask] = None,
    ) -> FloatTensor:
        # we refrain from binding score_mod on-demand because torch.compile would probably force you to do so repeatedly
        assert self.score_mod is not None, "score_mod has not yet been bound. Use T5EncoderStack#bind_score_mods after loading weights."
        qkv: FloatTensor = self.qkv_proj(x)
        q, k, v = rearrange(
            qkv, "batch seq (proj heads head_dim) -> proj batch heads seq head_dim", proj=3, head_dim=self.head_dim
        ).unbind()
        # NOTE: be sure to compile T5EncoderSelfAttentionFlex#forward! if you don't, then at least compile
        # this flex_attention operation (you could replace it with a call to .flex_utils.get_compiled_flex)
        from torch.nn.attention.flex_attention import flex_attention
        a = flex_attention(
            q,
            k,
            v,
            score_mod=self.score_mod,
            block_mask=block_mask,
            scale=self.scale,
            flex_kernel_options=self.config.flex_kernel_options,
        )
        a = rearrange(a, "batch heads seq head_dim -> batch seq (heads head_dim)")
        o = self.o_proj(a)
        return o

    def init_weights(self):
        nn.init.normal_(self.qkv_proj.weight, std=1 / math.sqrt(self.config.hidden_dim))
        nn.init.normal_(self.o_proj.weight, std=1 / math.sqrt(self.config.hidden_dim * self.config.num_layers))

class EncoderSelfAttentionFactory(Protocol):
    @staticmethod
    def __call__(config: T5Config) -> T5EncoderSelfAttention | T5EncoderSelfAttentionFlex: ...

####
#### T5 encoder layers
####

class SDPAArgs(NamedTuple):
    position_bias: FloatTensor
    mask: Optional[BoolTensor]

class FlexArgs(NamedTuple):
    block_mask: Optional[BlockMask]

class T5EncoderLayer(nn.Module):
    attn: T5EncoderSelfAttention | T5EncoderSelfAttentionFlex
    ln1: RMSNormCast
    """pre-attn layer norm"""
    ln2: RMSNormCast
    """post-attn layer norm"""
    ffn: T5ReLUFFN | T5GEGLUFFN
    dropout: nn.Dropout

    def __init__(
        self,
        config: T5Config,
        attn_ctor: EncoderSelfAttentionFactory = T5EncoderSelfAttention,
    ) -> None:
        super().__init__()
        self.attn = attn_ctor(config)
        ffn_factory = get_ffn_factory(config.ffn_type)
        self.ln1 = RMSNormCast(config.hidden_dim, eps=config.eps, dtype=config.norm_weight_dtype)
        self.ln2 = RMSNormCast(config.hidden_dim, eps=config.eps, dtype=config.norm_weight_dtype)
        self.ffn = ffn_factory(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x_r: ActAndResidual,
        attn_args: SDPAArgs | FlexArgs,
    ) -> ActAndResidual:
        x, residual = x_r
        x, residual = self.ln1(x, residual=residual)
        x = self.attn(x, *attn_args)
        x, residual = self.ln2(self.dropout(x), residual=residual)
        x = self.ffn(x)
        return ActAndResidual(self.dropout(x), residual)

    def init_weights(self):
        self.attn.init_weights()
        self.ln1.reset_parameters()
        self.ln2.reset_parameters()
        self.ffn.init_weights()


class T5EncoderStack(nn.Module):
    config: T5Config
    vocab_embed: Embedding
    relative_attention_bias: T5RelativeAttentionBias
    dropout: nn.Dropout
    layers: nn.ModuleList
    ln: RMSNormCast
    param_count: int
    non_emb_param_count: int

    def __init__(self, config: T5Config) -> None:
        super().__init__()
        self.config = config
        self.vocab_embed = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_dim,
            dtype=config.emb_weight_dtype,
        )
        self.relative_attention_bias = T5RelativeAttentionBias(config, bidirectional=True)
        match config.attn_impl:
            case T5AttnImpl.SDPA:
                attn_ctors = [T5EncoderSelfAttention]*config.num_layers
            case T5AttnImpl.Flex:
                bias_emb: Embedding = self.relative_attention_bias.bias_emb
                if config.pos_emb_per_layer:
                    emb_weights: list[FloatTensor] = bias_emb.weight.chunk(config.num_layers, dim=-1)
                else:
                    emb_weights: list[FloatTensor] = [bias_emb.weight]*config.num_layers
                attn_ctors = [partial(T5EncoderSelfAttentionFlex, emb_weight=emb_weight) for emb_weight in emb_weights]
            case _:
                raise ValueError(f"Unsupported attention implementation: {config.attn_impl}")
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([T5EncoderLayer(config, attn_ctor) for attn_ctor in attn_ctors])
        self.ln = RMSNormCast(config.hidden_dim, eps=config.eps, dtype=config.norm_weight_dtype)
        # we must calculate this at init, and not later, because FSDP may shard the params
        self.param_count = emb_param_count = 0
        for p in self.parameters():
            self.param_count += p.numel()
        for p in chain(self.vocab_embed.parameters(), self.relative_attention_bias.parameters()):
            emb_param_count += p.numel()
        self.non_emb_param_count = self.param_count - emb_param_count

    def forward(
        self,
        input_ids: LongTensor,
        input_mask: Optional[BoolTensor] = None,
        # SDPA: position_bias can be re-used across invocations; best computed outside
        position_bias: Optional[FloatTensor] = None,
        # Flex: block_mask creation seems to dislike being inside torch.compile, so I'ma ask you to pass it in
        block_mask: Optional[BlockMask] = None,
    ) -> FloatTensor:
        seq_len: int = input_ids.size(-1)
        input_embeds = self.vocab_embed(input_ids.flatten(end_dim=-2))
        
        attn_args: list[SDPAArgs|FlexArgs]
        match self.config.attn_impl:
            case T5AttnImpl.SDPA:
                if position_bias is None:
                    seq_len: int = input_ids.size(-1)
                    position_bias: FloatTensor = self.relative_attention_bias(seq_len)

                # UMT5 has a position embedding per layer, and we computed all of them simultaneously
                # note: [t]*n does not duplicate tensors, it just makes a list of n references to the same tensor
                biases: list[FloatTensor] = position_bias.unbind() if self.config.pos_emb_per_layer else [position_bias]*self.config.num_layers

                attn_mask: Optional[BoolTensor] = None if input_mask is None else self.broadcast_mask(input_mask, input_ids)
                    
                attn_args = [SDPAArgs(
                    position_bias=bias,
                    mask=attn_mask,
                ) for bias in biases]
            case T5AttnImpl.Flex:
                attn_args = [FlexArgs(block_mask=block_mask)]*self.config.num_layers

        x_r = ActAndResidual(self.dropout(input_embeds), None)
        for layer, attn_args_ in zip(self.layers, attn_args):
            layer: T5EncoderLayer
            x_r = layer(x_r, attn_args_)
        x, residual = x_r
        x = self.ln(x, residual=residual, prenorm=False)
        return x

    def bind_score_mods(self, seq_len=512) -> None:
        """
        [Only relevant if you're using flex attention]
        Run this function after you have loaded weights and transferred the model to device.
        You will probably want to run this within an inference_mode() or no_grad() context!
        """
        assert self.config.attn_impl == T5AttnImpl.Flex, "Only Flex attention has score_mods"
        position_bias: FloatTensor = self.relative_attention_bias(seq_len)
        biases: list[FloatTensor] = position_bias.unbind() if self.config.pos_emb_per_layer else [position_bias.squeeze(0)]*self.config.num_layers
        for layer, bias in zip(self.layers, biases):
            layer: T5EncoderLayer
            attn: T5EncoderSelfAttentionFlex = layer.attn
            attn.score_mod = attn.make_score_mod(bias)
    
    def broadcast_mask(self, input_mask: BoolTensor, input_ids: LongTensor) -> BoolTensor:
        seq_len: int = input_ids.size(-1)
        scores_shape = torch.Size((input_ids.size(0), self.config.n_head, seq_len, seq_len))
        match input_mask.ndim:
            case 2:
                assert (
                    input_mask.shape == input_ids.shape
                ), f"attn mask was 2-dim, so expected padding mask: (batch, seq_len). got {input_mask.shape} mask for {input_ids.shape} inputs."
                # broadcast the key-mask over all heads and queries (b h q k)
                attn_mask = rearrange(input_mask, "b k -> b 1 1 k")
            case 3:
                assert input_mask.shape == torch.Size(
                    [*input_ids.shape, input_ids.shape[-1]]
                ), f"attn mask was 3-dim, so expected packing mask: (batch, seq_len, seq_len). got {input_mask.shape} mask for {input_ids.shape} inputs."
                # broadcast over all heads (b h q k)
                attn_mask = rearrange(input_mask, "b q k -> b 1 q k")
            case 4:
                try:
                    torch.broadcast_shapes(input_mask.shape, scores_shape)
                except RuntimeError:
                    raise ValueError(
                        f"input_mask ({input_mask.shape}) was not broadcastable to scores, (batch, heads, seq_len, seq_len) ({scores_shape})"
                    )
            case _:
                raise ValueError(
                    f"Expected to broadcast 2~4-dim input mask onto {scores_shape} scores, got mask {input_mask.shape} for {input_ids.shape} inputs."
                )
        return attn_mask

    def flop_count_per_sequence(self, input_ids_len: int, labels_len: int) -> int:
        # note: encoder doesn't look at labels

        # encoder self-attn is non-causal
        return self.non_emb_param_count * input_ids_len * 6 + self.config.num_layers * flash_attention_flops(
            1, input_ids_len, input_ids_len, self.config.hidden_dim, 1, False, mode="fwd_bwd"
        )

    def init_weights(self):
        self.vocab_embed.reset_parameters()
        self.relative_attention_bias.init_weights()
        self.ln.reset_parameters()
        for layer in self.layers:
            layer.init_weights()
