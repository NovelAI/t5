import math
from typing import Optional, Protocol
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
from .flex.flex_utils import ScoreMod

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

# TODO:
class T5EncoderSelfAttentionFlex(nn.Module):
    score_mod: ScoreMod
    qkv_proj: Linear
    o_proj: Linear
    head_dim: int
    scale: float
    dropout: float
    config: T5Config

    def __init__(self, config: T5Config, emb_weight: FloatTensor) -> None:
        super().__init__()
        assert config.n_head == config.kv_heads, "Q and KV heads must be equal; GQA not implemented yet."
        self.head_dim = config.head_dim
        qkv_heads: int = config.n_head + config.kv_heads * 2
        self.scale = self.head_dim**-0.5 if config.scale_qk else 1.0
        self.dropout = config.dropout
        self.score_mod = self.make_self_attn_score_mod(
            emb_weight=emb_weight,
            num_buckets=config.relative_attention_num_buckets,
            max_distance=config.relative_attention_max_distance,
        )
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
    def make_self_attn_score_mod(
        emb_weight: FloatTensor,
        num_buckets=32,
        max_distance=128,
    ) -> ScoreMod:
        # encoder self-attn is bidirectional; each direction uses half the buckets
        half_buckets = num_buckets // 2

        max_half_bucket_ix = half_buckets - 1

        # (in each direction)
        # half of the buckets are allocated for exact increments in position
        # the other half are logarithmically growing bins in positions up to max_distance
        max_exact = half_buckets // 2

        relpos_coeff: float = (half_buckets - max_exact) / math.log(max_distance / max_exact)

        def score_mod(
            score: FloatTensor,
            b: IntTensor,
            h: IntTensor,
            q_idx: IntTensor,
            kv_idx: IntTensor,
        ) -> FloatTensor:
            mem_pos = kv_idx
            # NOTE: in decoder self-attn, if kv-cached decoding is used: we would see
            # only the final query element, and its position would need adjusting.
            # q_idx would not run from 0 to q_len-1,
            # the only query would be q_idx=0, and you'd want to add q_len-1 to it, e.g.
            #     ctx_pos = q_idx + q_len-1
            # or just:
            #     ctx_pos = q_len-1
            ctx_pos = q_idx
            relative_position = (mem_pos - ctx_pos).abs_()
            is_small = relative_position < max_exact

            # NOTE: in decoder self-attn, if kv-cached decoding is used: we would need
            # to move the diagonal of this upper-triangular mask, so that no key gains
            # a positive position (they'd all be behind the query).
            # maybe like this:
            #     excess_keys: int = k_len - q_len
            #     relative_buckets = torch.where(mem_pos > ctx_pos + excess_keys, half_buckets, 0)
            # errr or maybe just:
            #     relative_buckets = 0
            relative_buckets = torch.where(mem_pos > ctx_pos, half_buckets, 0)

            relpos_distant = (max_exact + relative_position.to(torch.float32, copy=True).div_(max_exact).log_().mul_(relpos_coeff).long()).clamp_max_(max_half_bucket_ix)

            relative_buckets.add_(torch.where(is_small, relative_position, relpos_distant))

            return score + emb_weight[relative_buckets, h]

        return score_mod

    def forward(
        self,
        x: FloatTensor,
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
            position_bias = position_bias.where(mask, -1e5)
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

class EncoderSelfAttentionFactory(Protocol):
    @staticmethod
    def __call__(config: T5Config) -> T5EncoderSelfAttention | T5EncoderSelfAttentionFlex: ...

####
#### T5 encoder layers
####


class T5EncoderLayer(nn.Module):
    attn: T5EncoderSelfAttention | T5EncoderSelfAttentionFlex
    ln1: RMSNormCast
    """pre-attn layer norm"""
    ln2: RMSNormCast
    """post-attn layer norm"""
    ffn: T5ReLUFFN | T5GEGLUFFN
    dropout: nn.Dropout
    # ln1_residual_scale: Optional[FloatTensor]
    # ln2_residual_scale: Optional[FloatTensor]
    ln1_residual_scale: Optional[float]
    ln2_residual_scale: Optional[float]

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
        # TODO: expose via config somehow
        # self.register_buffer('ln1_residual_scale', None)
        # self.register_buffer('ln2_residual_scale', None)
        self.ln1_residual_scale = None
        self.ln2_residual_scale = None

    def forward(
        self,
        x_r: ActAndResidual,
        position_bias: FloatTensor,
        attn_mask: Optional[BoolTensor] = None,
    ) -> ActAndResidual:
        x, residual = x_r
        x, residual = self.ln1(x, residual=residual)
        if self.ln1_residual_scale is not None:
            residual = residual * self.ln1_residual_scale
        x = self.attn(x, position_bias=position_bias, mask=attn_mask)
        x, residual = self.ln2(self.dropout(x), residual=residual)
        if self.ln2_residual_scale is not None:
            residual = residual * self.ln2_residual_scale
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
    ) -> FloatTensor:
        seq_len: int = input_ids.size(-1)
        input_embeds = self.vocab_embed(input_ids.flatten(end_dim=-2))
        position_bias: FloatTensor = self.relative_attention_bias(seq_len)
        attn_mask = None
        if input_mask is not None:
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
                case _:
                    raise ValueError(
                        f"Expected 2 or 3-dim input mask, got mask {input_mask.shape} for {input_ids.shape} inputs."
                    )

        # UMT5 has a position embedding per layer, and we computed all of them simultaneously
        # note: [t]*n does not duplicate tensors, it just makes a list of n references to the same tensor
        biases: list[FloatTensor] = position_bias.unbind() if self.config.pos_emb_per_layer else [position_bias]*self.config.num_layers

        x_r = ActAndResidual(self.dropout(input_embeds), None)
        for layer, bias in zip(self.layers, biases):
            assert isinstance(layer, T5EncoderLayer)
            x_r = layer(x_r, position_bias=bias, attn_mask=attn_mask)
        x, residual = x_r
        x = self.ln(x, residual=residual, prenorm=False)
        return x

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
