from typing import Optional, Protocol, TYPE_CHECKING, Any
from contextlib import nullcontext
from functools import lru_cache
from torch import FloatTensor, IntTensor, BoolTensor
import torch

if TYPE_CHECKING:
    # avoid runtime import to prevent explosion on older PyTorch
    from torch.nn.attention.flex_attention import BlockMask
else:
    BlockMask = _mask_mod_signature = Any

# there's an official type _score_mod_signature for this but our protocol provides more detail
class ScoreMod(Protocol):
    """
    Same purpose as torch.nn.attention.flex_attention._score_mod_signature
    but documents parameter names and datatypes.
    """
    @staticmethod
    def __call__(
        score: FloatTensor,
        batch: IntTensor,
        head: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
    ) -> FloatTensor: ...

class MaskMod(Protocol):
    """
    Same purpose as torch.nn.attention.flex_attention._mask_mod_signature
    but documents parameter names and datatypes.
    """
    @staticmethod
    def __call__(
        batch: IntTensor,
        head: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
    ) -> BoolTensor: ...

# based on attention-gym, BSD-3-Clause License
# https://github.com/pytorch-labs/attention-gym/blob/f7c93ded4abf9fd8d7dc9d8bcbf57e420b891e2d/attn_gym/utils.py#L23
# https://github.com/pytorch-labs/attention-gym/blob/main/LICENSE
def create_score_mod(
    query: FloatTensor,
    key: FloatTensor,
    score_mod: ScoreMod,
    device: str = "cuda",
    _compile: bool = False,
    scale: Optional[float] = None,
    batch_idx: int = 0,
    head_idx: int = 0,
) -> FloatTensor:
    from torch.nn.attention.flex_attention import _vmap_for_bhqkv
    # TODO This was moved on nightly, this enables 2.5 and 2.6 | we should remove this once 2.5 is no longer supported
    try:
        from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex
    except ImportError:
        from torch._higher_order_ops.flex_attention import TransformGetItemToIndex

    B = 1
    H = 1
    M = query.shape[0]
    N = key.shape[0]

    b = torch.arange(0, B, device=device) + batch_idx
    h = torch.arange(0, H, device=device) + head_idx
    m = torch.arange(0, M, device=device)
    n = torch.arange(0, N, device=device)

    scale_factor = query.size(-1)**-.5 if scale is None else scale
    ctx = nullcontext() if _compile else TransformGetItemToIndex()

    with ctx:
        mod = _vmap_for_bhqkv(score_mod, prefix=(0,))
        scores = query @ key.mT
        scores *= scale_factor
        scores = scores.view(1, 1, M, N)
        out = mod(scores, b, h, m, n)

    return out

def create_bias(
    score_mod: ScoreMod,
    q_len: int,
    kv_len: int,
    heads = 1,
    batch = 1,
    device: str = "cuda",
    _compile: bool = False,
) -> FloatTensor:
    from torch.nn.attention.flex_attention import _vmap_for_bhqkv
    # TODO This was moved on nightly, this enables 2.5 and 2.6 | we should remove this once 2.5 is no longer supported
    try:
        from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex
    except ImportError:
        from torch._higher_order_ops.flex_attention import TransformGetItemToIndex

    b = torch.arange(0, batch, device=device)
    h = torch.arange(0, heads, device=device)
    m = torch.arange(0, q_len, device=device)
    n = torch.arange(0, kv_len, device=device)

    ctx = nullcontext() if _compile else TransformGetItemToIndex()
    scores = torch.zeros(batch, heads, q_len, kv_len, device=device)

    with ctx:
        mod = _vmap_for_bhqkv(score_mod, prefix=(0,))
        out = mod(scores, b, h, m, n)

    return out

# in practice we are no longer using this; tried using it inside T5EncoderSelfAttention#forward…
# it compiled the flex_attention operation successfully, but it caused failures when compiling
# T5EncoderSelfAttention#forward itself (double/nested compilation problems I suppose?).
@lru_cache(maxsize=1)
def get_compiled_flex():
    """
    Memoize the compilation of flex_attention.
    """
    from torch.nn.attention.flex_attention import flex_attention
    return torch.compile(flex_attention, dynamic=False)

# caches last-used-mask only (an unbounded cache could leak memory)
@lru_cache(maxsize=1)
def create_block_mask_cached(
    mask_mod: MaskMod,
    batch: int,
    heads: int,
    q_len: int,
    kv_len: int,
    device="cuda",
) -> BlockMask:
    from torch.nn.attention.flex_attention import create_block_mask
    block_mask = create_block_mask(mask_mod, batch, heads, q_len, kv_len, device=device)
    return block_mask