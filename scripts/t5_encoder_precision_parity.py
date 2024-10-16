from typing import Any, Callable, NamedTuple
import json
from pathlib import Path
from enum import Enum
from contextlib import nullcontext

import torch
from torch import FloatTensor, LongTensor, BoolTensor, Tensor, inference_mode
from torch.amp.autocast_mode import autocast
from torch.nn.attention import SDPBackend, sdpa_kernel
from tensorizer import TensorDeserializer
from sentencepiece import SentencePieceProcessor

from nai_t5 import T5Config, T5EncoderStack
from nai_t5.t5_encoder import T5EncoderLayer

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
    return ' '.join((str(val) for val in (f'{str(tuple(t.shape)):14s}', f"{str(t.dtype).removeprefix('torch.'):8s}", f'σ={t.std().item():g}', f'μ={t.mean().item():.2f}', f'norm={fmt_matrix_norm(matrix_norm(t.float(), ord=2))}', f'absmax={t.abs().max().item():g}', label or '')))
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


class EncAndConfig(NamedTuple):
    enc: T5EncoderStack
    conf: T5Config


def get_model(dir: Path) -> EncAndConfig:
    with open(dir / 'config.json', 'r') as f:
        conf_dict: dict[str, Any] = json.load(f)
    config: T5Config = T5Config.model_validate(conf_dict)

    with torch.device('meta'):
        enc: T5EncoderStack = T5EncoderStack(config).eval()

    deserializer = TensorDeserializer(dir / 'enc.tensors', lazy_load=True)
    deserializer.load_into_module(enc)
    deserializer.close()
    return EncAndConfig(enc, config)

def explain_diff(ref: FloatTensor, candidate: FloatTensor) -> FloatTensor:
    diff = ref.float().sub(candidate.float())
    qs = torch.tensor([.5, .75, .9, .95, .99, .999, .9999], device=ref.device)
    q = diff.abs().quantile(qs)
    print(f'quantiles ({qs.cpu()}):\n           {q.cpu()}')
    stat(diff, 'diff')


class NamedActivation(NamedTuple):
    name: str
    act: FloatTensor

def main():
    device = torch.device("cuda")

    f32_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog/t5-v1_1-small-f32')
    f16_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog/t5-v1_1-small-f16')
    bf16_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog/t5-v1_1-small-bf16')
    # f32_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog/t5-v1_1-xl-f32')
    # f16_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog/t5-v1_1-xl-f16')
    # bf16_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog/t5-v1_1-xl-bf16')
    # f32_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog-v1/t5-large-f32')
    # f16_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog-v1/t5-large-f16')
    # bf16_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog-v1/t5-large-bf16')
    # f32_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/eleuther/pile-t5-large-f32')
    # f16_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/eleuther/pile-t5-large-f16')
    # bf16_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/eleuther/pile-t5-large-bf16')

    do_autocast = False
    f32_enc: Optional[T5EncoderStack] = None
    f16_enc: Optional[T5EncoderStack] = None
    bf16_enc: Optional[T5EncoderStack] = None
    f32_config: Optional[T5Config] = None
    f16_config: Optional[T5Config] = None
    bf16_config: Optional[T5Config] = None
    if f32_enabled := True:
        f32_enc, f32_config = get_model(f32_dir)
    if f16_enabled := True:
        f16_enc, f16_config = get_model(f16_dir)
    if bf16_enabled := False:
        bf16_enc, bf16_config = get_model(bf16_dir)
    
    print_first_block_only = True

    f32_activations: list[NamedActivation] = []
    f16_activations: list[NamedActivation] = []
    bf16_activations: list[NamedActivation] = []

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
                            if name.endswith('o_proj'):
                                (attn,) = args
                                out_list.append(NamedActivation(f'{name} [input]', attn))
                                assert attn.isfinite().all(), f'{model_name} {name} [input] has non-finite values'
                                print(f'{model_name} {f"{name} [input]":35s}:', stats(attn))
                            out_list.append(NamedActivation(name, output))
                            assert output.isfinite().all(), f'{model_name} {name} has non-finite values'
                            print(f'{model_name} {name:35s}:', stats(output))
                    handle: RemovableHandle = mod.register_forward_hook(partial(hook, name=name))
                    handles.append(handle)
                case T5EncoderLayer():
                    if print_first_block_only and name != 'layers.0': continue
                    def hook(mod, args, output, name: str):
                        out_list.append(NamedActivation(name, output))
                        assert output.isfinite().all(), f'{model_name} {name} has non-finite values'
                        print(f'{model_name} {name:35s}:', stats(output))
                    handle: RemovableHandle = mod.register_forward_hook(partial(hook, name=name))
                    handles.append(handle)
                case RMSNormCast():
                    if print_first_block_only and not name.startswith('layers.0'): continue
                    def hook(mod, args, output, name: str):
                        (input,) = args
                        out_list.append(NamedActivation(f'{name} [input]', input))
                        assert input.isfinite().all(), f'{model_name} {name} [input] has non-finite values'
                        print(f'{model_name} {f"{name} [input]":35s}:', stats(input))
                        out_list.append(NamedActivation(name, output))
                        assert output.isfinite().all(), f'{model_name} {name} has non-finite values'
                        print(f'{model_name} {name:35s}:', stats(output))
                    handle: RemovableHandle = mod.register_forward_hook(partial(hook, name=name))
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
    
    # for (k32, v32), (k16, v16) in zip(f32_enc.state_dict().items(), f16_enc.state_dict().items()):
    #     assert k32 == k16, f'{k32} != {k16}'
    #     assert v16.allclose(v32.half()), f'{k32} differs'
    # from torch.linalg import norm
    # with inference_mode():
    #     for k32, v32 in f32_enc.state_dict().items():
    #         assert (v32 > torch.finfo(torch.float16).max).sum() == 0, f"{k32} exceeds f16 max"
    #         assert (v32 < torch.finfo(torch.float16).min).sum() == 0, f"{k32} under f16 min"
    # with inference_mode():
    #     for name, mod in f32_enc.named_modules():
    #         from torch.nn import Linear, Embedding
    #         match mod:
    #             case Embedding():
    #                 # print(f'{name:25s}: {norm(mod.weight, ord=2, dim=-1).div_(mod.weight.size(-1)**.5).max()}')
    #                 # print(f'{name:25s}: {norm(mod.weight, ord=2)}')
    #                 print(f'{name:25s}: {norm(mod.weight, ord=2).div_(mod.weight.numel()**.5)}')
    #             case Linear():
    #                 # print(f'{name:25s}: {norm(mod.weight, ord=2, dim=-2).div_(mod.weight.size(-2)**.5).max()}')
    #                 # print(f'{name:25s}: {norm(mod.weight, ord=2)}')
    #                 print(f'{name:25s}: {norm(mod.weight, ord=2).div_(mod.weight.numel()**.5)}')
    pass

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

    qk_rebalance = 1
    vo_rebalance = 1
    print('qk_rebalance:', qk_rebalance)
    print('vo_rebalance:', vo_rebalance)
    with inference_mode():
        for f16_layer, f32_layer in zip(f16_enc.layers, f32_enc.layers):
            f16_layer: T5EncoderLayer
            f32_layer: T5EncoderLayer
            q16, k16, v16 = f16_layer.attn.qkv_proj.weight.chunk(3, dim=-2)
            q32, k32, v32 = f32_layer.attn.qkv_proj.weight.chunk(3, dim=-2)
            o16 = f16_layer.attn.o_proj.weight
            o32 = f32_layer.attn.o_proj.weight
            if qk_rebalance != 1:
                q16.copy_(q32.div(qk_rebalance).half())
                k16.copy_(k32.mul(qk_rebalance).half())
            if vo_rebalance != 1:
                v16.copy_(v32.div(vo_rebalance).half())
                o16.copy_(o32.mul(vo_rebalance).half())
            break

    seed = 42
    with (
        inference_mode(),
        fin(instrument_nai_t5(f32_enc, f32_config, f32_activations, ' f32')) if f32_enabled else nullcontext(),
        fin(instrument_nai_t5(f16_enc, f16_config, f16_activations, ' f16')) if f16_enabled else nullcontext(),
        fin(instrument_nai_t5(bf16_enc, bf16_config, bf16_activations, 'bf16')) if bf16_enabled else nullcontext(),
        sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION),
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
        if f16_enabled:
            print('f32 vs f16:')
            explain_diff(f32_out, f16_out)
            qs = torch.tensor([.5, .75, .9, .95, .99, .999, .9999], device=device)
            print("abs differences between layer activations...")
            print("quantiles:")
            print(str(qs.cpu()).removeprefix("tensor(").removesuffix(")"))
            torch.set_printoptions(linewidth=200)
            for f32_act, f16_act in zip(f32_activations, f16_activations):
                diff = f32_act.act.float().sub(f16_act.act.float())
                absdiff = diff.abs()
                print(f'{f32_act.name:35s}: {stats(absdiff):80s} {str(absdiff.quantile(qs).cpu()).removeprefix("tensor(").removesuffix(")")}')
        if bf16_enabled:
            print('f32 vs bf16:')
            explain_diff(f32_out, bf16_out)
    pass  # somewhere to put your breakpoint

if __name__ == "__main__":
    main()
