from typing import NamedTuple, Union

import torch
from torch import LongTensor

# The purpose of this file is to convince ourselves that our simplifications
# to HF's _relative_position_bucket are equivalent.


device = torch.device("cpu")


class BucketsAndPosition(NamedTuple):
    buckets: Union[LongTensor, int]
    position: LongTensor


# Excerpt from HF's _relative_position_bucket, Apache-licensed
# just the top part, which computes relative_position
# https://github.com/huggingface/transformers/blob/367a0dbd53cc1b826d986b166f3ac520d500db64/src/transformers/models/t5/modeling_t5.py#L384
def relative_buckets_hf(relative_position, bidirectional=True, num_buckets=32) -> BucketsAndPosition:
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    return BucketsAndPosition(relative_buckets, relative_position)


# Made it a bit more semantic
# - explicitly express our intention to apply a triangular mask
# - explicitly coerce relative_buckets to a tensor of zeros, giving it the same type in both branches
#   (making it less surprising when the downstream code does an element-wise addition to it)
def relative_buckets_v1(relative_position: LongTensor, bidirectional=True, num_buckets=32) -> BucketsAndPosition:
    if bidirectional:
        num_buckets //= 2
        relative_buckets = torch.triu(torch.full_like(relative_position, num_buckets), diagonal=1)
        relative_position = torch.abs(relative_position)
    else:
        relative_buckets = torch.zeros_like(relative_position)
        relative_position = -torch.tril(relative_position)
    return BucketsAndPosition(relative_buckets, relative_position)


# Eliminate wasted computation/allocation in the case of cached autoregressive inference,
# by supporting q_len != k_len
# This is the implementation we ultimately use in t5_bias.py.
def relative_buckets_v2(relative_position: LongTensor, bidirectional=True, num_buckets=32) -> BucketsAndPosition:
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
    return BucketsAndPosition(relative_buckets, relative_position)


def get_context_position(q_len: int, cached_autoregressive: bool, device: torch.device) -> Union[LongTensor, int]:
    if cached_autoregressive:
        # only the final query position will be kept, so that's the only one we'll compute
        return q_len - 1
    return torch.arange(q_len, dtype=torch.long, device=device).unsqueeze(-1)


ctx_len = 40
q_len = k_len = ctx_len

memory_position: LongTensor = torch.arange(k_len, dtype=torch.long, device=device).unsqueeze(0)
cp_all_q: LongTensor = get_context_position(q_len, cached_autoregressive=False, device=device)
cp_final_q: int = get_context_position(q_len, cached_autoregressive=True, device=device)
rp_all_q: LongTensor = memory_position - cp_all_q
rp_final_q: LongTensor = memory_position - cp_final_q

for directionality in [False, True]:
    b_hf, p_hf = relative_buckets_hf(rp_all_q, bidirectional=directionality)
    b_v1, p_v1 = relative_buckets_v1(rp_all_q, bidirectional=directionality)
    b_v2, p_v2 = relative_buckets_v2(rp_final_q, bidirectional=directionality)
    if isinstance(b_hf, int):
        b_hf = torch.tensor(b_hf, device=device).unsqueeze(0)
    assert b_v1.allclose(b_hf)
    assert p_v1.allclose(p_hf)
    # here we see how the v2 algorithm saves us from computing+discarding unused query positions
    assert b_v2.allclose(b_hf[-1:,])
    assert p_v2.allclose(p_hf[-1:,])
    pass  # nice place to put a breakpoint
pass  # nice place to put a breakpoint
