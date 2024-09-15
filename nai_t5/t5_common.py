import math
from enum import Enum
from typing import Optional, Type, Literal, Dict
from pydantic import BaseModel, field_validator, field_serializer

import torch
from einops import rearrange
from torch import FloatTensor, LongTensor, Tensor, nn
from torch.nn import Linear, Embedding
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


class DTypeModel:
    @field_validator("dtype")
    @classmethod
    def dtype_deserialize(cls, val: str | torch.dtype) -> torch.dtype:
        if isinstance(val, torch.dtype):
            if val not in dtype_map_reverse.keys():
                raise KeyError(
                    f"Received <{str(val)}>, but we only support dtypes: [{', '.join([str(k) for k in dtype_map_reverse.keys() if k is not None])}]"
                )
            return val
        assert isinstance(val, str)
        return dtype_map[val]

    @field_serializer("dtype")
    @classmethod
    def dtype_serialize(cls, val: torch.dtype) -> DType:
        assert isinstance(val, torch.dtype)
        assert val in dtype_map_reverse.keys()
        return dtype_map_reverse[val]


####
#### T5 config
####


class T5FFNType(str, Enum):
    ReLU = "ReLU"
    GEGLU = "GEGLU"


class T5Config(BaseModel, DTypeModel):
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
    dtype: DType | torch.dtype = torch.float32
    ffn_type: T5FFNType = T5FFNType.GEGLU
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    scale_qk: bool = True
    use_math_attn: bool = False
    pad_token_id: int = 0
    decoder_start_token_id: int = 0
    label_ignore_index: int = -100


####
#### T5 bias
####


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

    def __init__(self, config: T5Config, bidirectional: bool) -> None:
        nn.Module.__init__(self)
        self.bias_emb = Embedding(
            num_embeddings=config.relative_attention_num_buckets,
            embedding_dim=config.n_head,
            device=torch.get_default_device(),
            dtype=config.dtype,
        )
        # Encoder should be bidirectional
        self.bidirectional = bidirectional
        self.relative_attention_num_buckets = config.relative_attention_num_buckets

    # based on HF compute_bias, Apache-licensed
    # https://github.com/huggingface/transformers/blob/9138935784583203fb5f61e8f581cdfdcd887e0f/src/transformers/models/t5/modeling_t5.py#L431
    def forward(self, q_len: int, k_len: Optional[int] = None, cached_autoregressive=False) -> FloatTensor:
        """Compute binned relative position bias"""
        device = self.bias_emb.weight.device
        if k_len is None:
            k_len = q_len
        memory_position = torch.arange(k_len, dtype=torch.long, device=device).unsqueeze(0)
        if cached_autoregressive:
            # only the final query position will be kept, so that's the only one we'll compute
            context_position = q_len - 1
        else:
            context_position = torch.arange(q_len, dtype=torch.long, device=device).unsqueeze(-1)
        relative_position = memory_position - context_position  # shape (q_len, k_len)
        relative_position_bucket = _relative_position_bucket(
            relative_position,  # shape (q_len, k_len)
            bidirectional=self.bidirectional,
            num_buckets=self.relative_attention_num_buckets,
        )
        values: FloatTensor = self.bias_emb(relative_position_bucket)  # shape (q_len, k_len, num_heads)
        # shape (1, num_heads, q_len, k_len)
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
            dtype=config.dtype,
        )
        self.ff_out = Linear(
            in_features=config.ff_dim,
            out_features=config.hidden_dim,
            bias=False,
            dtype=config.dtype,
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
            dtype=config.dtype,
        )
        self.ff_out = Linear(
            in_features=config.ff_dim,
            out_features=config.hidden_dim,
            bias=False,
            dtype=config.dtype,
        )
        self.dropout = nn.Dropout(config.dropout)
        # you can get closer HF parity with transformers.activations.NewGELUActivation,
        # but nn.GELU is faster and still predicts the same token in our testing
        self.gate = nn.GELU()
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


@autocast(device_type='cuda', enabled=False)
def f_rms_norm(x, weight: Optional[torch.nn.Parameter], eps=1e-5):
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    if weight is None:
        return x
    return weight * x


# default eps seems to be 1e-5 for llama2
@autocast(device_type='cuda', enabled=False)
class RMSNorm_f32(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: str | torch.device | None = None,
        # dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(normalized_shape, device=device, dtype=torch.float32))

        else:
            self.register_parameter("weight", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)

    # @torch.compile
    @autocast(device_type='cuda', enabled=False)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return f_rms_norm(input.float(), self.weight, self.eps).to(input.dtype)

    def extra_repr(self) -> str:
        return "eps={eps}, elementwise_affine={elementwise_affine}".format(**self.__dict__)


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


####
#### Float16 clamper
####


f16_info = torch.finfo(torch.float16)
f16_safermin = f16_info.min + 1000
f16_safermax = f16_info.max - 1000

# based on HF transformers' T5 float16 adaptations
# Apache-licensed
# https://github.com/huggingface/transformers/blob/main/LICENSE
# credit: Suraj Patil
# https://github.com/huggingface/transformers/pull/9487
# credit: Prathik Rao
# https://github.com/huggingface/transformers/pull/22097
def clamp_inf_if_float16(x: FloatTensor) -> FloatTensor:
    """clamp activations to give """
    if x.dtype == torch.float16:
        x.clamp_(x, min=f16_safermin, max=f16_safermax)
    return x