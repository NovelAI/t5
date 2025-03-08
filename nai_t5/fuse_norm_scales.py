from typing import NamedTuple

import torch
from torch import FloatTensor, inference_mode
from torch.nn import Linear

from nai_t5 import T5
from nai_t5.t5_encoder import T5EncoderStack, T5EncoderLayer
from nai_t5.t5_decoder import T5DecoderStack, T5DecoderLayer
from nai_t5.t5_common import RMSNormCast

class NormAndScale(NamedTuple):
    ln: RMSNormCast
    scale: FloatTensor
def extract_norm_scales(orig: RMSNormCast) -> NormAndScale:
    assert orig.elementwise_affine
    ln = RMSNormCast(
        normalized_shape=orig.normalized_shape,
        eps=orig.eps,
        elementwise_affine=False,
        device=orig.weight.device,
    )
    return NormAndScale(ln, orig.weight)


@inference_mode()
def fuse_scales_into_lin_weight(lin_weight: FloatTensor, scales: FloatTensor) -> None:
    scale_diag: FloatTensor = torch.eye(
        scales.size(-1),
        device=lin_weight.device,
        dtype=lin_weight.dtype,
    ) * scales.type_as(lin_weight).unsqueeze(-1)
    lin_weight.copy_(lin_weight @ scale_diag)

def fuse_ln_scales_into_lin(lin: Linear, affine_ln: RMSNormCast) -> RMSNormCast:
    """
    Fuses an affine RMSNorm's scales into a Linear's weight matrix.
    Returns a new instance of the RMSNorm with elementwise_affine=False.
    """
    ln, scale = extract_norm_scales(affine_ln)
    fuse_scales_into_lin_weight(lin.weight, scale)
    return ln


def fuse_norm_scales_enc(t5_enc: T5EncoderStack) -> None:
    for layer in t5_enc.layers:
        layer: T5EncoderLayer

        ln1: RMSNormCast = fuse_ln_scales_into_lin(layer.attn.qkv_proj, layer.ln1)
        setattr(layer, 'ln1', ln1)

        ln2: RMSNormCast = fuse_ln_scales_into_lin(layer.ffn.ff_in.weight, layer.ln2)
        setattr(layer, 'ln2', ln2)


def fuse_norm_scales_dec(t5_dec: T5DecoderStack) -> None:
    for layer in t5_dec.layers:
        layer: T5DecoderLayer

        ln1: RMSNormCast = fuse_ln_scales_into_lin(layer.self_attn.qkv_proj, layer.ln1)
        setattr(layer, 'ln1', ln1)

        ln2: RMSNormCast = fuse_ln_scales_into_lin(layer.cross_attn.kv_proj, layer.ln2)
        setattr(layer, 'ln2', ln2)

        ln3: RMSNormCast = fuse_ln_scales_into_lin(layer.ffn.ff_in.weight, layer.ln3)
        setattr(layer, 'ln3', ln3)

def fuse_norm_scales(t5: T5) -> None:
    fuse_norm_scales_enc(t5.encoder)
    fuse_norm_scales_dec(t5.decoder)