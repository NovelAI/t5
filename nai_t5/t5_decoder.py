import math
from typing import List, Optional

from einops import rearrange
import torch
from torch import BoolTensor, FloatTensor, inference_mode, nn
from torch.nn import Linear
from torch.nn.functional import scaled_dot_product_attention

from .t5_common import (
    ActAndResidual,
    RMSNormCast,
    T5GEGLUFFN,
    T5Config,
    T5RelativeAttentionBias,
    T5ReLUFFN,
    get_ffn_factory,
)

####
#### T5 decoder cross-attention
####


class T5CrossAttention(nn.Module):
    q_proj: Linear
    kv_proj: Linear
    o_proj: Linear
    head_dim: int
    scale: float
    dropout: float
    config: T5Config

    def __init__(self, config: T5Config) -> None:
        super().__init__()
        assert config.n_head == config.kv_heads, "Q and KV heads must be equal; GQA not implemented yet."
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5 if config.scale_qk else 1.0
        self.dropout = config.dropout
        self.q_proj = Linear(
            in_features=config.hidden_dim,
            out_features=config.head_dim * config.n_head,
            bias=False,
            dtype=config.linear_weight_dtype,
        )
        self.kv_proj = Linear(
            in_features=config.hidden_dim,
            out_features=config.head_dim * config.kv_heads * 2,
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
        x: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[BoolTensor] = None,
        kv: Optional[FloatTensor] = None,
    ) -> FloatTensor:
        q: FloatTensor = self.q_proj(x)
        q = rearrange(q, "batch seq (heads head_dim) -> batch heads seq head_dim", head_dim=self.head_dim)
        if kv is None:
            kv: FloatTensor = self.kv_proj(y)
            kv = rearrange(
                kv, "batch seq (proj heads head_dim) -> proj batch heads seq head_dim", proj=2, head_dim=self.head_dim
            )
        k, v = kv.unbind()

        # TODO: if training, then learn scales for Q and K as a proxy for learning rate
        if mask is not None:
            assert mask.ndim == 4, "Expected [batch, heads, q, k] attention mask"
        a = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )
        a = rearrange(a, "batch heads seq head_dim -> batch seq (heads head_dim)")
        o = self.o_proj(a)
        return o

    def init_weights(self, generator: Optional[torch.Generator] = None) -> None:
        nn.init.normal_(self.q_proj.weight, std=1 / math.sqrt(self.config.hidden_dim), generator=generator)
        nn.init.normal_(self.kv_proj.weight, std=1 / math.sqrt(self.config.hidden_dim), generator=generator)
        nn.init.normal_(self.o_proj.weight, std=1 / math.sqrt(self.config.hidden_dim * self.config.num_layers), generator=generator)


####
#### T5 decoder self-attention
####


class T5DecoderSelfAttention(nn.Module):
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
        bias: FloatTensor,
        past_kv: Optional[FloatTensor] = None,
    ) -> FloatTensor:
        qkv: FloatTensor = self.qkv_proj(x)
        q, k, v = rearrange(
            qkv, "batch seq (proj heads head_dim) -> proj batch heads seq head_dim", proj=3, head_dim=self.head_dim
        ).unbind()
        bias = bias.type_as(q)

        if past_kv is not None:
            kv_in, kv_out = past_kv.split((past_kv.size(-2) - 1, 1), dim=-2)
            past_k, past_v = kv_in.unbind(0)
            k_out, v_out = kv_out.unbind(0)

            k_out.copy_(k)
            v_out.copy_(v)

            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        # TODO: if training, then learn scales for Q and K as a proxy for learning rate
        # NOTE: this _is_ a causal attention but we are not able to enable is_causal optimization because
        #       we are passing an arbitrary bias (T5 achieves relative position embedding via a learned bias)
        # NOTE: when kv-cache is in use (i.e. autoregressive inference), the diagonal of the mask ceases
        #       to be default, and instead becomes k_len_total-q_len
        # TODO: torch 2.4.0 FlexAttention could enable a simpler way to express the bias (and/or the causal mask
        #       with which it's modulated). moreover it would enable us to not pay the cost of materializing said mask.
        # TODO: torch nightly/2.5.0 FlexAttention enables blockwise sparsity over arbitrary masks, so could
        #       enable us to speed up this causal attention
        a = scaled_dot_product_attention(
            q,
            k,
            v,
            # fused kernel requires last dimension of input to have stride 1.
            attn_mask=bias,  # .contiguous(),
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )
        a = rearrange(a, "batch heads seq head_dim -> batch seq (heads head_dim)")
        o = self.o_proj(a)
        return o

    def init_weights(self, generator: Optional[torch.Generator] = None) -> None:
        nn.init.normal_(self.qkv_proj.weight, std=1 / math.sqrt(self.config.hidden_dim), generator=generator)
        nn.init.normal_(self.o_proj.weight, std=1 / math.sqrt(self.config.hidden_dim * self.config.num_layers), generator=generator)


####
#### T5 decoder layers
####


class T5DecoderLayer(nn.Module):
    self_attn: T5DecoderSelfAttention
    cross_attn: T5CrossAttention
    ln1: RMSNormCast
    """pre-attn layer norm"""
    ln2: RMSNormCast
    """pre-xattn layer norm"""
    ln3: RMSNormCast
    """post-attn layer norm"""
    ffn: T5ReLUFFN | T5GEGLUFFN
    dropout: nn.Dropout

    def __init__(self, config: T5Config) -> None:
        super().__init__()
        self.self_attn = T5DecoderSelfAttention(config=config)
        self.cross_attn = T5CrossAttention(config=config)
        ffn_factory = get_ffn_factory(config.ffn_type)
        self.ln1 = RMSNormCast(config.hidden_dim, eps=config.eps, dtype=config.norm_weight_dtype, elementwise_affine=config.elementwise_affine)
        self.ln2 = RMSNormCast(config.hidden_dim, eps=config.eps, dtype=config.norm_weight_dtype, elementwise_affine=config.elementwise_affine)
        self.ln3 = RMSNormCast(config.hidden_dim, eps=config.eps, dtype=config.norm_weight_dtype, elementwise_affine=config.elementwise_affine)
        self.ffn = ffn_factory(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x_r: ActAndResidual,
        encoding: FloatTensor,
        self_attn_bias: FloatTensor,
        cross_attn_mask: Optional[BoolTensor] = None,
        self_past_kv: Optional[FloatTensor] = None,
        cross_kv: Optional[FloatTensor] = None,
    ) -> ActAndResidual:
        x, residual = x_r
        x, residual = self.ln1(x, residual=residual)
        x = self.self_attn(
            x,
            bias=self_attn_bias,
            past_kv=self_past_kv,
        )
        x, residual = self.ln2(self.dropout(x), residual=residual)
        x = self.cross_attn(x, encoding, mask=cross_attn_mask, kv=cross_kv)
        x, residual = self.ln3(self.dropout(x), residual=residual)
        x = self.ffn(x)
        return ActAndResidual(self.dropout(x), residual)

    def init_weights(self, generator: Optional[torch.Generator] = None) -> None:
        self.self_attn.init_weights(generator)
        self.cross_attn.init_weights(generator)
        self.ln1.reset_parameters()
        self.ln2.reset_parameters()
        self.ln3.reset_parameters()
        self.ffn.init_weights(generator)


class T5DecoderStack(nn.Module):
    config: T5Config
    relative_attention_bias: T5RelativeAttentionBias
    dropout: nn.Dropout
    layers: nn.ModuleList
    ln: RMSNormCast
    param_count: int
    non_emb_param_count: int

    def __init__(self, config: T5Config) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.relative_attention_bias = T5RelativeAttentionBias(config, bidirectional=False)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([T5DecoderLayer(config) for _ in range(config.num_layers)])
        self.ln = RMSNormCast(config.hidden_dim, eps=config.eps, dtype=config.norm_weight_dtype, elementwise_affine=config.elementwise_affine)
        # we must calculate this at init, and not later, because FSDP may shard the params
        self.param_count = emb_param_count = 0
        for p in self.parameters():
            self.param_count += p.numel()
        for p in self.relative_attention_bias.parameters():
            emb_param_count += p.numel()
        self.non_emb_param_count = self.param_count - emb_param_count

    def forward(
        self,
        input_embeds: FloatTensor,
        encoding: FloatTensor,
        input_mask: Optional[BoolTensor] = None,
        self_past_kv: Optional[FloatTensor] = None,
        cross_kv: Optional[FloatTensor] = None,
        cross_mask: Optional[BoolTensor] = None,
    ) -> FloatTensor:
        q_len: int = input_embeds.size(-2)
        k_len_self_past: int = 0 if self_past_kv is None else self_past_kv.size(-2) - 1
        k_len_self_total: int = q_len + k_len_self_past
        self_position_bias = self.relative_attention_bias(
            k_len_self_total, cached_autoregressive=self_past_kv is not None
        )
        if self_past_kv is None:
            assert (
                k_len_self_total == q_len
            ), "when KV-cache is not provided: we assume typical self-attention (q_len == k_len)"
            if input_mask is None:
                # an everything-visible padding mask, broadcastable to all batch items
                input_mask = input_embeds.new_ones((1, q_len), dtype=torch.bool)
            # ignore channels dim from input_embeds to determine original input_ids shape
            inputs_shape = input_embeds.shape[:-1]
            match input_mask.ndim:
                case 2:
                    assert (
                        input_mask.shape == inputs_shape
                    ), f"attn mask was 2-dim, so expected padding mask: (batch, seq_len). got {input_mask.shape} mask for {inputs_shape} inputs."
                    # broadcast over all heads and queries (b h q k)
                    self_mask = rearrange(input_mask, "b k -> b 1 1 k")
                    self_mask = self_mask.repeat_interleave(k_len_self_total, dim=-2)
                    self_mask.tril_(k_len_self_total - q_len)
                case 3:
                    assert input_mask.shape == torch.Size(
                        [*inputs_shape, inputs_shape[-1]]
                    ), f"attn mask was 3-dim, so expected packing mask: (batch, seq_len, seq_len). got {input_mask.shape} mask for {inputs_shape} inputs."
                    # broadcast over all heads (b h q k)
                    self_mask = rearrange(input_mask, "b q k -> b 1 q k")
                    # user is responsible for per-key masking, so we assume they already causally-masked it
                case _:
                    raise ValueError(
                        f"Expected 2 or 3-dim input mask, got mask {input_mask.shape} for {inputs_shape} inputs."
                    )
            if self.config.pos_emb_per_layer:
                # broadcast over all layers
                self_mask.unsqueeze_(0)
            self_attn_bias: FloatTensor = self_position_bias.where(self_mask, -10000)
        else:
            assert (
                q_len == 1
            ), "when KV-cache is provided: we assume autoregressive inference, where causal self-attn means 1 query attending to itself + past KVs"
            assert (
                input_mask is None
            ), "KV-cache was provided, so we assume there are no future tokens or padding in your sequence that would need masking."
            self_attn_bias: FloatTensor = self_position_bias
            # if you really wanted to (maybe for speculative decoding?),
            # I believe you can narrow a causal mask like this (which for q_len=1 will be all-ones):
            # self_mask = input_embeds.new_ones((1, 1, q_len, k_len_self_total), dtype=torch.bool).tril_(k_len_self_total - q_len)
            # self_attn_bias: FloatTensor = self_position_bias.where(self_mask, -10000)

        # if you don't pass in a cross-mask, you're saying "all of decoder sequence should attend to all of encoding".
        # in other words: neither the encoder nor decoder sequence employs padding or packing.
        if cross_mask is not None:
            # ignore channels dim from input_embeds to determine original input_ids shape
            inputs_shape = input_embeds.shape[:-1]
            enc_len = encoding.size(-2)
            assert cross_mask.shape == torch.Size(
                [*inputs_shape, enc_len]
            ), "Expected 3-dim cross-attention mask: (batch, q_len, k_len) where q_len is the decoder sequence, k_len is the encoding."
            # broadcast over all heads (b h q k)
            cross_mask = rearrange(cross_mask, "b q k -> b 1 q k")

        # UMT5 has a position embedding per layer, and we computed all of them simultaneously
        # note: [e]*n does not duplicate tensors, it just makes a list of n references to the same tensor
        self_biases: list[FloatTensor] = self_attn_bias.unbind() if self.config.pos_emb_per_layer else [self_attn_bias]*self.config.num_layers

        layer_self_kvs: List[Optional[FloatTensor]] = (
            [None] * len(self.layers) if self_past_kv is None else self_past_kv.unbind()
        )
        layer_cross_kvs: List[Optional[FloatTensor]] = (
            [None] * len(self.layers) if cross_kv is None else cross_kv.unbind()
        )
        x_r = ActAndResidual(self.dropout(input_embeds), None)
        for layer, layer_self_kv, layer_cross_kv, self_bias in zip(self.layers, layer_self_kvs, layer_cross_kvs, self_biases):
            assert isinstance(layer, T5DecoderLayer)
            x_r = layer(
                x_r,
                encoding,
                self_attn_bias=self_bias,
                cross_attn_mask=cross_mask,
                self_past_kv=layer_self_kv,
                cross_kv=layer_cross_kv,
            )
        x, residual = x_r
        x = self.ln(x, residual=residual, prenorm=False)
        x = self.dropout(x)
        return x

    def get_kv_cache(
        self,
        batch_size=1,
        length: Optional[int] = None,
        dtype=torch.float32,
        device: Optional[torch.device | str] = None,
    ) -> FloatTensor:
        kv_max_len: int = self.config.n_tokens if length is None else length
        if device is None:
            device = torch.empty(0).device
        self_past_kv: FloatTensor = torch.zeros(
            (
                self.config.num_layers,
                2,  # k, then v
                batch_size,  # batch size
                self.config.kv_heads,
                kv_max_len,
                self.config.head_dim,
            ),
            dtype=dtype,
            device=device,
        )
        return self_past_kv

    @inference_mode()
    def get_cross_kv(self, encoding: FloatTensor) -> FloatTensor:
        expected_shape = (
            self.config.num_layers,
            2,  # k, then v
            encoding.size(0),  # batch size
            self.config.kv_heads,
            encoding.size(-2),
            self.config.head_dim,
        )
        cross_kv: FloatTensor = encoding.new_zeros(expected_shape)
        for layer, layer_kv in zip(self.layers, cross_kv.unbind()):
            assert isinstance(layer, T5DecoderLayer)
            kv = layer.cross_attn.kv_proj(encoding)
            kv = rearrange(
                kv,
                "batch seq (proj heads head_dim) -> proj batch heads seq head_dim",
                proj=2,
                head_dim=self.config.head_dim,
            )
            layer_kv.copy_(kv)
        return cross_kv

    def init_weights(self, generator: Optional[torch.Generator] = None) -> None:
        self.relative_attention_bias.init_weights(generator)
        self.ln.reset_parameters()
        for layer in self.layers:
            layer: T5DecoderLayer
            layer.init_weights(generator)
