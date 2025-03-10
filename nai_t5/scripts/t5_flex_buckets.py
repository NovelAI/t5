#!/usr/bin/env python

from torch import LongTensor, FloatTensor, inference_mode
import torch
from torch.nn import Embedding
from einops import rearrange

from nai_t5.t5_common import _relative_position, _relative_position_bucket
from nai_t5.t5_encoder import T5EncoderSelfAttentionFlex
from nai_t5.flex_utils import ScoreMod, create_bias

device = torch.device('cuda')
ctx_len = 512
num_buckets = 32
max_distance = 128
heads = 6
torch.set_printoptions(edgeitems=20, linewidth=280)

relative_position: LongTensor = _relative_position(ctx_len, device=device)
buckets: LongTensor = _relative_position_bucket(
    relative_position,
    num_buckets=num_buckets,
    max_distance=max_distance,
    bidirectional=True,
)

# emb_weights: FloatTensor = torch.zeros(num_buckets, heads, device=device)
# emb_weights: FloatTensor = torch.arange(num_buckets, device=device).unsqueeze(-1).expand(-1, heads)
emb_weights: FloatTensor = torch.randn(num_buckets, heads, device=device)
emb = Embedding(num_buckets, heads, device=device).eval()
with inference_mode():
    emb.weight.copy_(emb_weights)
    ref_bias = emb(buckets)

score_mod: ScoreMod = T5EncoderSelfAttentionFlex.make_self_attn_score_mod(
    emb_weight=emb_weights,
    num_buckets=num_buckets,
    max_distance=max_distance,
)
bias: FloatTensor = create_bias(
    score_mod=score_mod,
    q_len=ctx_len,
    kv_len=ctx_len,
    heads=heads,
    batch=1,
    device=device,
    _compile=False,
)
print(bias[0,0])
# assert bias[0,0].allclose(buckets.float()) # will only work with the arange() emb weights
assert rearrange(bias, '1 h q k -> q k h').allclose(ref_bias)
pass