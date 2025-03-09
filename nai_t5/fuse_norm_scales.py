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
def fuse_scales_into_lin_weight(
    lin_weight: FloatTensor,
    scales: FloatTensor,
    fuse_via_f32: bool = False,
) -> None:
    # fusing via f32 probably doesn't matter much, but I didn't check.
    # might be worth it if you're just doing it once for the purpose of exporting a fused model.
    fusion_dtype = torch.float32 if fuse_via_f32 else torch.promote_types(lin_weight.dtype, scales.dtype)
    scale_diag: FloatTensor = torch.eye(
        scales.size(-1),
        device=lin_weight.device,
        dtype=fusion_dtype,
    ) * scales.type(fusion_dtype).unsqueeze(-1)
    lin_weight.copy_(lin_weight.type(fusion_dtype) @ scale_diag.type(fusion_dtype))

def fuse_ln_scales_into_lin(
    lin: Linear,
    affine_ln: RMSNormCast,
    fuse_via_f32: bool = False,
) -> RMSNormCast:
    """
    Fuses an affine RMSNorm's scales into a Linear's weight matrix.
    Returns a new instance of the RMSNorm with elementwise_affine=False.
    """
    ln, scale = extract_norm_scales(affine_ln)
    fuse_scales_into_lin_weight(
        lin.weight,
        scale,
        fuse_via_f32=fuse_via_f32,
    )
    return ln


def fuse_norm_scales_enc(
    t5_enc: T5EncoderStack,
    fuse_via_f32: bool = False,
) -> None:
    for layer in t5_enc.layers:
        layer: T5EncoderLayer

        ln1: RMSNormCast = fuse_ln_scales_into_lin(layer.attn.qkv_proj, layer.ln1, fuse_via_f32=fuse_via_f32)
        setattr(layer, 'ln1', ln1)

        ln2: RMSNormCast = fuse_ln_scales_into_lin(layer.ffn.ff_in, layer.ln2, fuse_via_f32=fuse_via_f32)
        setattr(layer, 'ln2', ln2)


def fuse_norm_scales_dec(
    t5_dec: T5DecoderStack,
    fuse_via_f32: bool = False,
) -> None:
    for layer in t5_dec.layers:
        layer: T5DecoderLayer

        ln1: RMSNormCast = fuse_ln_scales_into_lin(layer.self_attn.qkv_proj, layer.ln1, fuse_via_f32=fuse_via_f32)
        setattr(layer, 'ln1', ln1)

        ln2, scale = extract_norm_scales(layer.ln2)
        fuse_scales_into_lin_weight(layer.cross_attn.q_proj.weight, scale, fuse_via_f32=fuse_via_f32)
        fuse_scales_into_lin_weight(layer.cross_attn.kv_proj.weight, scale, fuse_via_f32=fuse_via_f32)
        setattr(layer, 'ln2', ln2)

        ln3: RMSNormCast = fuse_ln_scales_into_lin(layer.ffn.ff_in, layer.ln3, fuse_via_f32=fuse_via_f32)
        setattr(layer, 'ln3', ln3)

def fuse_norm_scales(
    t5: T5,
    fuse_via_f32: bool = False,
) -> None:
    fuse_norm_scales_enc(t5.encoder, fuse_via_f32=fuse_via_f32)
    fuse_norm_scales_dec(t5.decoder, fuse_via_f32=fuse_via_f32)