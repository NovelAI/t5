import math
from enum import Enum
from typing import Optional, Type, Literal, Dict, Any, NamedTuple
from pydantic import BaseModel, field_validator, field_serializer

import torch
from einops import rearrange
from torch import FloatTensor, LongTensor, Tensor, nn
from torch.nn import Linear, Embedding
from torch.nn.modules.normalization import _shape_t
# from torch.nn.modules.normalization import RMSNorm, _shape_t
from .layer_norm import RMSNorm
from torch.amp import autocast

####
#### DType pydantic definitions (for persisting torch.dtype into a config)
####


class DType(str, Enum):
    Float16 = "float16"
    Float32 = "float32"
    BFloat16 = "bfloat16"
    None_ = "none"

dtype_map: Dict[DType, Optional[torch.dtype]] = {
    DType.Float16: torch.float16,
    DType.Float32: torch.float32,
    DType.BFloat16: torch.bfloat16,
    DType.None_: None,
}
# reverse map
dtype_map_reverse: Dict[Optional[torch.dtype], DType] = {v: k for k, v in dtype_map.items()}

class DTypeSerializer:
    @staticmethod
    def dtype_deserialize(val: str | torch.dtype) -> torch.dtype:
        if isinstance(val, torch.dtype):
            if val not in dtype_map_reverse.keys():
                raise KeyError(
                    f"Received <{str(val)}>, but we only support dtypes: [{', '.join([str(k) for k in dtype_map_reverse.keys() if k is not None])}]"
                )
            return val
        assert isinstance(val, str)
        return dtype_map[val]

    @staticmethod
    def dtype_serialize(val: torch.dtype) -> DType:
        assert isinstance(val, torch.dtype)
        assert val in dtype_map_reverse.keys()
        return dtype_map_reverse[val]


####
#### T5 config
####


class T5FFNType(str, Enum):
    ReLU = "ReLU"
    GEGLU = "GEGLU"

class GELUApprox(str, Enum):
    None_ = "none"
    Tanh = "tanh"

class T5AttnImpl(str, Enum):
    SDPA = "sdpa"
    Flex = "flex"


class T5Config(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    vocab_size: int
    hidden_dim: int
    num_layers: int
    n_head: int
    kv_heads: int
    head_dim: int
    ff_dim: int
    dropout: float = 0.1
    eps: float = 1e-6
    emb_weight_dtype: DType | torch.dtype = torch.float32
    linear_weight_dtype: DType | torch.dtype = torch.float32
    norm_weight_dtype: DType | torch.dtype = torch.float32
    ffn_type: T5FFNType = T5FFNType.GEGLU
    gelu_approx: GELUApprox = GELUApprox.None_
    attn_impl: T5AttnImpl = T5AttnImpl.SDPA
    flex_kernel_options: dict[str, Any] = {}
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    scale_qk: bool = True
    pad_token_id: int = 0
    decoder_start_token_id: int = 0
    label_ignore_index: int = -100

    pos_emb_per_layer: bool = False
    """
    True = UMT5-style (position embedding, per layer)
    False = T5-style (one position embedding, shared by all layers)
    """

    elementwise_affine: bool = True

    @field_validator("emb_weight_dtype", "linear_weight_dtype", "norm_weight_dtype")
    @classmethod
    def dtype_deserialize(cls, val: str | torch.dtype) -> torch.dtype:
        return DTypeSerializer.dtype_deserialize(val)

    @field_serializer("emb_weight_dtype", "linear_weight_dtype", "norm_weight_dtype")
    @classmethod
    def dtype_serialize(cls, val: torch.dtype) -> DType:
        return DTypeSerializer.dtype_serialize(val)


####
#### Residualized activation
####

class ActAndResidual(NamedTuple):
    x: FloatTensor
    residual: FloatTensor


####
#### T5 bias
####

def _relative_position(
    q_len: int,
    k_len: Optional[int] = None,
    cached_autoregressive=False,
    device = torch.device('cpu'),
) -> LongTensor:
    if k_len is None:
        k_len = q_len
    memory_position = torch.arange(k_len, dtype=torch.long, device=device).unsqueeze(0)
    if cached_autoregressive:
        # only the final query position will be kept, so that's the only one we'll compute
        context_position = q_len - 1
    else:
        context_position = torch.arange(q_len, dtype=torch.long, device=device).unsqueeze(-1)
    relative_position = memory_position - context_position  # shape (q_len, k_len)
    return relative_position

# based on HF implementation, Apache-licensed:
# https://github.com/huggingface/transformers/blob/9138935784583203fb5f61e8f581cdfdcd887e0f/src/transformers/models/t5/modeling_t5.py#L384
def _relative_position_bucket(
    relative_position: LongTensor, bidirectional: bool, num_buckets=32, max_distance=128
) -> Tensor:
    # in cached autoregressive inference, we have 1 query attending to n keys.
    # we move the diagonal to be equivalent to having n queries attending to n keys.
    *_, q_len, k_len = relative_position.shape
    excess_keys: int = k_len - q_len
    if bidirectional:
        num_buckets //= 2
        # I think the excess_keys offset here is never exercised in practice,
        # because the only bidirectional case is encoder self-attn, which doesn't need KV-caching.
        # still, it's probably the correct way to adjust the diagonal if you somehow had that use-case.
        relative_buckets = torch.triu(torch.full_like(relative_position, num_buckets), diagonal=1 + excess_keys)
        relative_position = torch.abs(relative_position)
    else:
        relative_buckets = torch.zeros_like(relative_position)
        relative_position = -torch.tril(relative_position, diagonal=excess_keys)
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_position_if_large = (
        max_exact
        + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()
    )
    relative_position_if_large = relative_position_if_large.min(relative_position_if_large.new_tensor(num_buckets - 1))

    relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets


class T5RelativeAttentionBias(nn.Module):
    bidirectional: bool
    relative_attention_num_buckets: int
    bias_emb: Embedding
    config: T5Config

    def __init__(self, config: T5Config, bidirectional: bool) -> None:
        nn.Module.__init__(self)
        self.bias_emb = Embedding(
            num_embeddings=config.relative_attention_num_buckets,
            embedding_dim=config.n_head * (config.num_layers if config.pos_emb_per_layer else 1),
            dtype=config.emb_weight_dtype,
        )
        # Encoder should be bidirectional
        self.bidirectional = bidirectional
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.config = config

    # based on HF compute_bias, Apache-licensed
    # https://github.com/huggingface/transformers/blob/9138935784583203fb5f61e8f581cdfdcd887e0f/src/transformers/models/t5/modeling_t5.py#L431
    def forward(self, q_len: int, k_len: Optional[int] = None, cached_autoregressive=False) -> FloatTensor:
        """Compute binned relative position bias"""
        relative_position: LongTensor = _relative_position(
            q_len,
            k_len,
            cached_autoregressive,
            device=self.bias_emb.weight.device,
        )
        relative_position_bucket = _relative_position_bucket(
            relative_position,  # shape (q_len, k_len)
            bidirectional=self.bidirectional,
            num_buckets=self.relative_attention_num_buckets,
        )
        values: FloatTensor = self.bias_emb(relative_position_bucket)  # shape (q_len, k_len, num_heads)
        # shape (1, num_heads, q_len, k_len)

        # UMT5 has a position embedding per layer, and we computed all of them simultaneously.
        if self.config.pos_emb_per_layer:
            # move layers to dim 0; we can unbind them from there and each chunk will still be contiguous
            values = rearrange(values, "q k (layers heads) -> layers 1 heads q k", layers=self.config.num_layers)
        else:
            values = rearrange(values, "q k heads -> 1 heads q k")

        # need stride of last dimension to be 1 in order to be eligible for torch sdp mem-eff kernels
        # for some reason values.contiguous() doesn't achieve this, but cloning with contiguous format does
        values = values.clone(memory_format=torch.contiguous_format)
        return values

    def init_weights(self):
        self.bias_emb.reset_parameters()


####
#### T5 FFN
####


# implements ReLU, (T5 v1.0)
class T5ReLUFFN(nn.Module):
    ff_in: Linear
    gate: nn.ReLU
    dropout: nn.Dropout
    ff_out: Linear
    config: T5Config

    def __init__(self, config: T5Config) -> None:
        super().__init__()
        self.ff_in = Linear(
            in_features=config.hidden_dim,
            out_features=config.ff_dim,
            bias=False,
            dtype=config.linear_weight_dtype,
        )
        self.ff_out = Linear(
            in_features=config.ff_dim,
            out_features=config.hidden_dim,
            bias=False,
            dtype=config.linear_weight_dtype,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.gate = nn.ReLU()
        self.config = config

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = self.ff_in(x)
        # TODO: should we checkpoint the gate, as Arda did?
        x = self.gate(x)
        x = self.dropout(x)
        x = self.ff_out(x)
        return x

    def init_weights(self):
        nn.init.normal_(self.ff_in.weight, std=1 / math.sqrt(self.config.hidden_dim))
        nn.init.normal_(self.ff_out.weight, std=1 / math.sqrt(self.config.hidden_dim * self.config.num_layers))


# implements GEGLU, (T5 v1.1)
# TODO: consider fusions
class T5GEGLUFFN(nn.Module):
    ff_in: Linear
    gate: nn.GELU
    dropout: nn.Dropout
    ff_out: Linear
    config: T5Config

    def __init__(self, config: T5Config) -> None:
        super().__init__()
        self.ff_in = Linear(
            in_features=config.hidden_dim,
            out_features=config.ff_dim * 2,
            bias=False,
            dtype=config.linear_weight_dtype,
        )
        self.ff_out = Linear(
            in_features=config.ff_dim,
            out_features=config.hidden_dim,
            bias=False,
            dtype=config.linear_weight_dtype,
        )
        self.dropout = nn.Dropout(config.dropout)
        # you can get closer HF parity with transformers.activations.NewGELUActivation,
        # but nn.GELU is faster and still predicts the same token in our testing
        self.gate = nn.GELU(approximate=config.gelu_approx.value)
        self.config = config

    # TODO: torch.compile
    def forward(self, x: FloatTensor) -> FloatTensor:
        x = self.ff_in(x)
        g, x = torch.chunk(x, 2, dim=-1)
        g = self.gate(g)
        x = g * x
        x = self.dropout(x)
        x = self.ff_out(x)
        return x

    def init_weights(self):
        nn.init.normal_(self.ff_in.weight, std=1 / math.sqrt(self.config.hidden_dim))
        nn.init.normal_(self.ff_out.weight, std=1 / math.sqrt(self.config.hidden_dim * self.config.num_layers))


def get_ffn_factory(ffn_type: T5FFNType) -> Type[T5ReLUFFN | T5GEGLUFFN]:
    match (ffn_type):
        case T5FFNType.ReLU:
            return T5ReLUFFN
        case T5FFNType.GEGLU:
            return T5GEGLUFFN
        case _:
            raise ValueError(f"Unknown T5FFNType: {ffn_type}")


####
#### RMSNorm
####

class RMSNormCast(RMSNorm):
    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: Optional[float] = None,
        elementwise_affine: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        residual_scale: Optional[float] = None,
    ) -> None:
        assert isinstance(normalized_shape, int)
        super().__init__(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
            residual_scale=residual_scale,
        )
    
    def forward(self, x: FloatTensor, residual: FloatTensor, prenorm=True) -> ActAndResidual | FloatTensor:
        out = super().forward(x, residual=residual, prenorm=prenorm, residual_in_fp32=True)
        if prenorm:
            x, residual = out
            return ActAndResidual(x=x, residual=residual)
        return out


# class RMSNormCast(RMSNorm):
#     def __init__(
#         self,
#         normalized_shape: _shape_t,
#         eps: Optional[float] = None,
#         elementwise_affine: bool = True,
#         device: str | torch.device | None = None,
#         dtype: torch.dtype = torch.float32,
#     ) -> None:
#         super().__init__(
#             normalized_shape,
#             eps=eps,
#             elementwise_affine=elementwise_affine,
#             device=device,
#             dtype=dtype,
#         )

#     @autocast(device_type='cuda', enabled=False)
#     def forward(self, input: Tensor) -> Tensor:
#         dtype = self.weight.dtype if self.elementwise_affine else torch.float32
#         return super().forward(input.type(dtype)).type_as(input)


####
#### FLOP counter
####


# Based on Dao-AILab's flops() function
# https://github.com/Dao-AILab/flash-attention/blob/32792d37ec66902e5d82e149971daacbee8b55d7/benchmarks/benchmark_flash_attention.py#L27
# License: BSD 3-clause
# https://github.com/Dao-AILab/flash-attention/blob/main/LICENSE
def flash_attention_flops(
    batch: int,
    q_len: int,
    kv_len: int,
    headdim: int,
    nheads: int,
    causal: bool,
    mode: Literal["fwd", "bwd", "fwd_bwd"] = "fwd",
) -> int | float:
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    if q_len != kv_len:
        assert not causal, "we don't know how well attention can take advantage of sparsity in causal cross-attention."
    f = 4 * batch * nheads * q_len * kv_len * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)