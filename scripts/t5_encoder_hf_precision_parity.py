from typing import Any, Callable, NamedTuple, TypeVar
import json
from pathlib import Path
from enum import Enum
from contextlib import nullcontext

import torch
from torch import FloatTensor, LongTensor, BoolTensor, Tensor, inference_mode
from torch.nn import RMSNorm
from torch.amp.autocast_mode import autocast
from torch.nn.attention import SDPBackend, sdpa_kernel
from tensorizer import TensorDeserializer
from sentencepiece import SentencePieceProcessor

from nai_t5 import T5Config, T5EncoderStack
from nai_t5.t5_common import RMSNormCast
from nai_t5.t5_encoder import T5EncoderLayer
from nai_t5.weight_load import FusingDeserializer
from nai_t5.replace_linear import replace_linear

from torch import Tensor
from typing import Optional
from torch.linalg import matrix_norm
def fmt_matrix_norm(t: Tensor) -> str:
    t = t.squeeze().cpu()
    if t.numel() == 1:
        return f'{t.item():.2f}'
    if t.numel() > 4:
        return f'avg {t.mean().item():.2f}'
    return str(t)
def stats(t: Tensor, label: Optional[str] = None) -> str:
    return ' '.join((str(val) for val in (f'{str(tuple(t.shape)):14s}', f"{str(t.dtype).removeprefix('torch.'):8s}", f'σ={t.std().item():g}', f'μ={t.mean().item():.2f}', f'norm={"N/A" if t.ndim < 2 else fmt_matrix_norm(matrix_norm(t.float(), ord=2))}', f'absmax={t.abs().max().item():g}', label or '')))
def stat(t: Tensor, label: Optional[str] = None) -> None:
    print(stats(t, label))

from functools import partial
from torch.utils.hooks import RemovableHandle
from contextlib import contextmanager

@contextmanager
def fin(dtor):
    try:
        yield
    finally:
        dtor()

class PrecisionMode(str, Enum):
    Float32 = 'f32'
    MixedBF16 = 'mixed-bf16'
    PureBF16 = 'pure-bf16'
    PureF16 = 'pure-f16'

class Checkpoint(str, Enum):
    T5v1_1Small = 't5-v1.1-small'
    T5v1_1XL = 't5-v1.1-xl'
    T5v1_1XXL = 't5-v1.1-xxl'
    T5v1Large = 't5-v1-large'
    PileT5Large = 'pile-t5-large'


class EncAndConfig(NamedTuple):
    enc: T5EncoderStack
    conf: T5Config


def get_model(
    dir: Path,
    dtype: Optional[torch.dtype] = None,
    fuse_norm_scales = False,
    norm_fusion_via_f32 = False,
    enc_attn_out_scales: Optional[list[float]] = None,
    enc_ffn_out_scales: Optional[list[float]] = None,
) -> EncAndConfig:
    with open(dir / 'config.json', 'r') as f:
        conf_dict: dict[str, Any] = json.load(f)
    config: T5Config = T5Config.model_validate(conf_dict)
    config.elementwise_affine = not fuse_norm_scales

    with torch.device('meta'):
        enc: T5EncoderStack = T5EncoderStack(config).eval()

    if enc_ffn_out_scales is not None or enc_attn_out_scales is not None:
        deserializer = FusingDeserializer(dir / 'enc.tensors', lazy_load=True, dtype=dtype)
        deserializer.load_with_fusions(
            enc,
            fuse_norm_scales=fuse_norm_scales,
            norm_fusion_via_f32=norm_fusion_via_f32,
            enc_attn_out_scales=enc_attn_out_scales,
            enc_ffn_out_scales=enc_ffn_out_scales,
        )
    else:
        deserializer = TensorDeserializer(dir / 'enc.tensors', lazy_load=True, dtype=dtype)
        deserializer.load_into_module(enc)
    deserializer.close()
    return EncAndConfig(enc, config)

def explain_diff(ref: FloatTensor, candidate: FloatTensor) -> FloatTensor:
    diff = ref.float().sub(candidate.float())
    qs = torch.tensor([.5, .75, .9, .95, .99, .999, .9999], device=ref.device)
    q = diff.abs().quantile(qs)
    print(str(q.cpu()).removeprefix("tensor(").removesuffix(")"))
    stat(diff, 'diff')


class NamedActivation(NamedTuple):
    name: str
    act: FloatTensor

class NormAndScale(NamedTuple):
    ln: RMSNormCast
    scale: FloatTensor
def extract_norm_scales(orig: RMSNorm) -> NormAndScale:
    # assert orig.elementwise_affine
    ln = RMSNormCast(
        # normalized_shape=orig.normalized_shape,
        normalized_shape=orig.weight.size(-1),
        eps=orig.eps,
        # elementwise_affine=False,
        device=orig.weight.device,
    )
    return NormAndScale(ln, orig.weight)

T = TypeVar('T')
class VoidList(list[T]):
    def append(self, _: T) -> None:
        pass

ffn_out_scale_dict: dict[Checkpoint, Optional[list[float]]] = {
    # 8 layers
    Checkpoint.T5v1_1Small: [*[1]*6, 1/8, 1],
    # 24 layers
    Checkpoint.T5v1_1XL: [*[1]*4, 1/8, *[1]*19],
    # 24 layers
    Checkpoint.T5v1_1XXL: [*[1]*10, 1/4, *[1]*13],
}

attn_out_scale_dict: dict[Checkpoint, Optional[list[float]]] = {
    Checkpoint.T5v1_1Small: None,
    Checkpoint.T5v1_1XL: None,
    Checkpoint.T5v1_1XXL: None,
}

def main():
    device = torch.device("cuda")

    ckpt = Checkpoint.T5v1_1XL
    match ckpt:
        case Checkpoint.T5v1_1Small:
            f32_needs_cast = f16_needs_cast = bf16_needs_cast = False
            f32_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog/t5-v1_1-small-f32')
            f16_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog/t5-v1_1-small-f16')
            bf16_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog/t5-v1_1-small-bf16')
        case Checkpoint.T5v1_1XL:
            f32_needs_cast = f16_needs_cast = bf16_needs_cast = False
            f32_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog/t5-v1_1-xl-f32')
            f16_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog/t5-v1_1-xl-f16')
            bf16_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog/t5-v1_1-xl-bf16')
        case Checkpoint.T5v1_1XXL:
            f32_needs_cast = f16_needs_cast = True
            bf16_needs_cast = False
            f32_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog/t5-v1_1-xxl-bf16')
            f16_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog/t5-v1_1-xxl-bf16')
            bf16_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog/t5-v1_1-xxl-bf16')
        case Checkpoint.T5v1Large:
            f32_needs_cast = f16_needs_cast = bf16_needs_cast = False
            f32_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog-v1/t5-large-f32')
            f16_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog-v1/t5-large-f16')
            bf16_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog-v1/t5-large-bf16')
        case Checkpoint.PileT5Large:
            f32_needs_cast = f16_needs_cast = bf16_needs_cast = False
            f32_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/eleuther/pile-t5-large-f32')
            f16_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/eleuther/pile-t5-large-f16')
            bf16_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/eleuther/pile-t5-large-bf16')
        case _:
            raise ValueError(f'unknown checkpoint: {ckpt}')

    fuse_norms = True

    do_autocast = False
    f32_enc: Optional[T5EncoderStack] = None
    f16_enc: Optional[T5EncoderStack] = None
    bf16_enc: Optional[T5EncoderStack] = None
    f32_config: Optional[T5Config] = None
    f16_config: Optional[T5Config] = None
    bf16_config: Optional[T5Config] = None
    # load if you intend to invoke the model, or if you intend to use it as a weight donor.
    # if you're loading it anyway, then we designate it as avaiable for weight donation too.
    if f32_enabled := True:
        dtype: Optional[torch.dtype] = torch.float32 if f32_needs_cast else None
        f32_enc, f32_config = get_model(f32_dir, dtype=dtype)
    if f16_enabled := True:
        dtype: Optional[torch.dtype] = torch.float16 if f16_needs_cast else None
        scaling_kwargs = {
            'fuse_norm_scales': fuse_norms,
            'norm_fusion_via_f32': True,
            'enc_attn_out_scales': attn_out_scale_dict[ckpt],
            'enc_ffn_out_scales': ffn_out_scale_dict[ckpt],
        }
        f16_enc, f16_config = get_model(
            f16_dir,
            dtype=dtype,
            **scaling_kwargs,
        )
        if f16_acc_gpupoor := False:
            from gpu_poor.modules import LowPrecisionLinear
            replace_linear(f16_enc, LowPrecisionLinear)
        if f16_acc_cublas_ops := False:
            from cublas_ops import CublasLinear
            replace_linear(f16_enc, CublasLinear)
    if bf16_enabled := False or (bf16_weight_donor := False):
        bf16_weight_donor = True
        dtype: Optional[torch.dtype] = torch.bfloat16 if bf16_needs_cast else None
        bf16_enc, bf16_config = get_model(bf16_dir)
    
    print_first_block_only = False

    retain_activations = False

    f32_activations: list[NamedActivation] = [] if retain_activations else VoidList()
    f16_activations: list[NamedActivation] = [] if retain_activations else VoidList()
    bf16_activations: list[NamedActivation] = [] if retain_activations else VoidList()

    def instrument_nai_t5(module: T5EncoderStack, config: T5Config, out_list: list[NamedActivation], model_name: str) -> Callable[[], None]:
        from torch.nn import GELU, Embedding, Linear
        from nai_t5.t5_common import RMSNormCast, T5GEGLUFFN
        from nai_t5.t5_encoder import T5EncoderLayer
        handles: list[RemovableHandle] = []
        for name, mod in module.named_modules():
            match mod:
                case Embedding():
                    def hook(mod, args, output, name: str):
                        if config.pos_emb_per_layer and name.endswith('bias_emb'):
                            for ix, b in enumerate(output.unflatten(-1, (config.num_layers, -1)).unbind(-2)):
                                out_list.append(NamedActivation(f'{name}.{ix}', b))
                                assert b.isfinite().all(), f'{model_name} {name}.{ix} has non-finite values'
                                print(f'{model_name} {f"{name}.{ix}":35s}:', stats(b))
                        else:
                            out_list.append(NamedActivation(name, output))
                            assert output.isfinite().all(), f'{model_name} {name} has non-finite values'
                            print(f'{model_name} {name:35s}:', stats(output))
                    handle: RemovableHandle = mod.register_forward_hook(partial(hook, name=name))
                    handles.append(handle)
                case Linear():
                    if print_first_block_only and not name.startswith('layers.0'): continue
                    def hook(mod, args, output, name: str):
                        if name.endswith('qkv_proj'):
                            q, k, v = output.chunk(3, dim=-1)
                            out_list.append(NamedActivation(f'{name}.q', q))
                            out_list.append(NamedActivation(f'{name}.k', k))
                            out_list.append(NamedActivation(f'{name}.v', v))
                            assert q.isfinite().all(), f'{model_name} {name}.q has non-finite values'
                            assert k.isfinite().all(), f'{model_name} {name}.k has non-finite values'
                            assert v.isfinite().all(), f'{model_name} {name}.v has non-finite values'
                            print(f'{model_name} {f"{name}.q":35s}:', stats(q))
                            print(f'{model_name} {f"{name}.k":35s}:', stats(k))
                            print(f'{model_name} {f"{name}.v":35s}:', stats(v))
                        elif name.endswith('ff_in'):
                            wi_0, wi_1 = output.chunk(2, dim=-1)
                            out_list.append(NamedActivation(f'{name}.wi_0', wi_0))
                            out_list.append(NamedActivation(f'{name}.wi_1', wi_1))
                            assert wi_0.isfinite().all(), f'{model_name} {name}.wi_0 has non-finite values'
                            assert wi_1.isfinite().all(), f'{model_name} {name}.wi_1 has non-finite values'
                            print(f'{model_name} {f"{name}.wi_0":35s}:', stats(wi_0))
                            print(f'{model_name} {f"{name}.wi_1":35s}:', stats(wi_1))
                        else:
                            if name.endswith('o_proj') or name.endswith('ff_out'):
                                (out,) = args
                                out_list.append(NamedActivation(f'{name} [input]', out))
                                assert out.isfinite().all(), f'{model_name} {name} [input] has non-finite values'
                                print(f'{model_name} {f"{name} [input]":35s}:', stats(out))
                            out_list.append(NamedActivation(name, output))
                            assert output.isfinite().all(), f'{model_name} {name} has non-finite values'
                            print(f'{model_name} {name:35s}:', stats(output))
                    handle: RemovableHandle = mod.register_forward_hook(partial(hook, name=name))
                    handles.append(handle)
                case T5EncoderLayer():
                    if print_first_block_only and name != 'layers.0': continue
                    def hook(mod, args, output, name: str):
                        # out_list.append(NamedActivation(name, output))
                        # assert output.isfinite().all(), f'{model_name} {name} has non-finite values'
                        # print(f'{model_name} {name:35s}:', stats(output))
                        (x, residual) = output
                        out_list.append(NamedActivation(f'{name}.x', x))
                        out_list.append(NamedActivation(f'{name}.residual', residual))
                        assert x.isfinite().all(), f'{model_name} {name}.x has non-finite values'
                        assert residual.isfinite().all(), f'{model_name} {name}.residual has non-finite values'
                        print(f'{model_name} {f"{name}.x":35s}:', stats(x))
                        print(f'{model_name} {f"{name}.residual":35s}:', stats(residual))
                    handle: RemovableHandle = mod.register_forward_hook(partial(hook, name=name))
                    handles.append(handle)
                case RMSNormCast():
                    if print_first_block_only and not name.startswith('layers.0'): continue
                    def hook(mod, args, kwargs: dict[str, Any], output, name: str):
                        # assert (name := kwargs.get('name', None)) is not None
                        (input,) = args
                        out_list.append(NamedActivation(f'{name} [input]', input))
                        assert input.isfinite().all(), f'{model_name} {name} [input] has non-finite values'
                        print(f'{model_name} {f"{name} [input]":35s}:', stats(input))
                        if (residual_in := kwargs.get('residual', None)) is not None:
                            out_list.append(NamedActivation(f'{name} [residual_in]', residual_in))
                            assert residual_in.isfinite().all(), f'{model_name} {name} [residual_in] has non-finite values'
                            print(f'{model_name} {f"{name} [residual_in]":35s}:', stats(residual_in))
                        if torch.is_tensor(output):
                            out_list.append(NamedActivation(name, output))
                            assert output.isfinite().all(), f'{model_name} {name} has non-finite values'
                            print(f'{model_name} {name:35s}:', stats(output))
                        else:
                            act, residual_out = output
                            out_list.append(NamedActivation(name, act))
                            out_list.append(NamedActivation(f'{name} [residual_out]', residual_out))
                            assert act.isfinite().all(), f'{model_name} {name} has non-finite values'
                            assert residual_out.isfinite().all(), f'{model_name} {name} [residual_out] has non-finite values'
                            print(f'{model_name} {name:35s}:', stats(act))
                            print(f'{model_name} {f"{name} [residual_out]":35s}:', stats(residual_out))

                    handle: RemovableHandle = mod.register_forward_hook(partial(hook, name=name), with_kwargs=True)
                    handles.append(handle)
                case GELU():
                    if print_first_block_only and not name.startswith('layers.0'): continue
                    def hook(mod, args, output, name: str):
                        out_list.append(NamedActivation(name, output))
                        assert output.isfinite().all(), f'{model_name} {name} has non-finite values'
                        print(f'{model_name} {name:35s}:', stats(output))
                    handle: RemovableHandle = mod.register_forward_hook(partial(hook, name=name))
                    handles.append(handle)
                case T5GEGLUFFN():
                    if print_first_block_only and not name.startswith('layers.0'): continue
                    def hook(mod, args, output, name: str):
                        out_list.append(NamedActivation(name, output))
                        assert output.isfinite().all(), f'{model_name} {name} has non-finite values'
                        print(f'{model_name} {name:35s}:', stats(output))
                    handle: RemovableHandle = mod.register_forward_hook(partial(hook, name=name))
                    handles.append(handle)
        def dtor():
            for handle in handles:
                handle.remove()
        return dtor

    tokenizer = SentencePieceProcessor(model_file=str(f32_dir / 'spiece.model'))
    
    prompts: list[str] = ['hello world']
    batch_size = len(prompts)

    toks: list[list[int]] = tokenizer.Encode(prompts, add_eos=True)
    # ctx_len = 512
    ctx_len = len(toks[0])
    input_ids: LongTensor = torch.full((batch_size, ctx_len), fill_value=tokenizer.pad_id(), dtype=torch.long, device='cpu')
    for seq, input_out in zip(toks, input_ids.unbind()):
        input_out[:len(seq)].copy_(torch.tensor(seq[:ctx_len], dtype=torch.long))
    input_ids = input_ids.to(device)
    mask: BoolTensor = input_ids != tokenizer.pad_id()

    seed = 42
    with (
        inference_mode(),
        fin(instrument_nai_t5(f32_enc, f32_config, f32_activations, ' f32')) if f32_enabled else nullcontext(),
        fin(instrument_nai_t5(f16_enc, f16_config, f16_activations, ' f16')) if f16_enabled else nullcontext(),
        fin(instrument_nai_t5(bf16_enc, bf16_config, bf16_activations, 'bf16')) if bf16_enabled else nullcontext(),
        # sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION),
        # sdpa_kernel(SDPBackend.CUDNN_ATTENTION),
    ):
        torch.manual_seed(seed)
        if f32_enabled:
            with autocast(device_type=device.type, dtype=torch.float16) if do_autocast else nullcontext():
                f32_out: FloatTensor = f32_enc(
                    input_ids=input_ids,
                    input_mask=mask,
                )
            assert f32_out.isfinite().all(), 'f32_out has non-finite values'
        if bf16_enabled:
            bf16_out: FloatTensor = bf16_enc(
                input_ids=input_ids,
                input_mask=mask,
            )
        if f16_enabled:
            f16_out: FloatTensor = f16_enc(
                input_ids=input_ids,
                input_mask=mask,
            )
            assert f16_out.isfinite().all(), 'f16_out has non-finite values'
    
    if f32_enabled:
        if bf16_enabled or f16_enabled:
            qs = torch.tensor([.5, .75, .9, .95, .99, .999, .9999], device=device)
            print("quantiles:")
            print(str(qs.cpu()).removeprefix("tensor(").removesuffix(")"))
        if f16_enabled:
            print('f32 vs f16:')
            explain_diff(f32_out, f16_out)
        if bf16_enabled:
            print('f32 vs bf16:')
            explain_diff(f32_out, bf16_out)
        if f32_activations and f16_activations:
            print("abs differences between f32, f16 layer activations...")
            torch.set_printoptions(linewidth=200)
            for f32_act, f16_act in zip(f32_activations, f16_activations):
                diff = f32_act.act.float().sub(f16_act.act.float())
                absdiff = diff.abs()
                print(f'{f32_act.name:35s}: {stats(absdiff):80s} {str(absdiff.quantile(qs).cpu()).removeprefix("tensor(").removesuffix(")")}')
    pass  # somewhere to put your breakpoint

if __name__ == "__main__":
    main()
