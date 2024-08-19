import math
from typing import Optional
from itertools import chain

import torch
from einops import rearrange
from torch import BoolTensor, FloatTensor, LongTensor, nn
from torch.nn import Linear, Embedding
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention

from .t5_common import RMSNorm_f32, T5GEGLUFFN, T5Config, T5RelativeAttentionBias, T5ReLUFFN, flash_attention_flops, get_ffn_factory

####
#### T5 encoder self-attention
####


class T5EncoderSelfAttention(nn.Module):
    qkv_proj: Linear
    o_proj: Linear
    head_dim: int
    scale: float
    dropout: float
    use_math_attn: bool
    config: T5Config

    def __init__(self, config: T5Config) -> None:
        super().__init__()
        assert config.n_head == config.kv_heads, "Q and KV heads must be equal; GQA not implemented yet."
        self.head_dim = config.head_dim
        qkv_heads: int = config.n_head + config.kv_heads * 2
        self.scale = self.head_dim**-0.5 if config.scale_qk else 1.0
        self.use_math_attn = config.use_math_attn
        self.dropout = config.dropout
        self.qkv_proj = Linear(
            in_features=config.hidden_dim,
            out_features=config.head_dim * qkv_heads,
            bias=False,
            dtype=config.dtype,
        )
        self.o_proj = Linear(
            in_features=config.head_dim * config.n_head,
            out_features=config.hidden_dim,
            bias=False,
            dtype=config.dtype,
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
        # TODO: if training, then learn scales for Q and K as a proxy for learning rate
        if mask is not None:
            assert mask.ndim == 4, "Expected [batch, heads, q, k] attention mask"
            position_bias = position_bias.where(mask, -1e5)
        # use math attn for HF parity instead of performance
        with sdpa_kernel(
            [SDPBackend.MATH]
            if self.use_math_attn
            else [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
            ]
        ):
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


####
#### T5 encoder layers
####


class T5EncoderLayer(nn.Module):
    attn: T5EncoderSelfAttention
    ln1: RMSNorm_f32
    """pre-attn layer norm"""
    ln2: RMSNorm_f32
    """post-attn layer norm"""
    ffn: T5ReLUFFN | T5GEGLUFFN
    dropout: nn.Dropout

    def __init__(self, config: T5Config) -> None:
        super().__init__()
        self.attn = T5EncoderSelfAttention(config=config)
        ffn_factory = get_ffn_factory(config.ffn_type)
        device = torch.get_default_device()
        self.ln1 = RMSNorm_f32(config.hidden_dim, eps=config.eps, device=device)
        self.ln2 = RMSNorm_f32(config.hidden_dim, eps=config.eps, device=device)
        self.ffn = ffn_factory(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        position_bias: FloatTensor,
        attn_mask: Optional[BoolTensor] = None,
    ) -> FloatTensor:
        residual = x
        x = self.ln1(x)
        attn_out: FloatTensor = self.attn(x, position_bias=position_bias, mask=attn_mask)

        x = residual + self.dropout(attn_out)

        residual = x
        x = self.ffn(self.ln2(x))

        x = residual + self.dropout(x)

        return x

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
    ln: RMSNorm_f32
    param_count: int
    non_emb_param_count: int

    def __init__(self, config: T5Config) -> None:
        super().__init__()
        self.config = config
        device = torch.get_default_device()
        self.vocab_embed = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_dim,
            device=device,
            dtype=config.dtype,
        )
        self.relative_attention_bias = T5RelativeAttentionBias(config, bidirectional=True)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([T5EncoderLayer(config) for _ in range(config.num_layers)])
        self.ln = RMSNorm_f32(config.hidden_dim, eps=config.eps, device=device)
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
        position_bias = self.relative_attention_bias(seq_len)
        attn_mask = None
        if input_mask is not None:
            match input_mask.ndim:
                case 2:
                    assert (
                        input_mask.shape == input_ids.shape
                    ), f"attn mask was 2-dim, so expected padding mask: (batch, seq_len). got {input_mask.shape} mask for {input_ids.shape} inputs."
                    # broadcast over all heads and keys (b h q k)
                    attn_mask = rearrange(input_mask, "b q -> b 1 q 1")
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
        x: FloatTensor = self.dropout(input_embeds)
        for layer in self.layers:
            assert isinstance(layer, T5EncoderLayer)
            x: FloatTensor = layer(x, position_bias=position_bias, attn_mask=attn_mask)
        normed: FloatTensor = self.ln(x)
        return normed

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
