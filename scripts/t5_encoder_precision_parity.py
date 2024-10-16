from typing import Any
import json
from pathlib import Path
from enum import Enum

import torch
from torch import FloatTensor, LongTensor, BoolTensor, Tensor, inference_mode
from torch.amp import autocast_mode
from tensorizer import TensorDeserializer
from sentencepiece import SentencePieceProcessor

from nai_t5 import T5Config, T5EncoderStack

from torch import Tensor
from typing import Optional
from torch.linalg import matrix_norm
def fmt_matrix_norm(t: Tensor) -> str:
    t = t.squeeze().cpu()
    if t.numel() == 1:
        return f'{t.item():.2f}'
    return str(t)
def stat(t: Tensor, label: Optional[str] = None) -> None:
    print(tuple(t.shape), str(t.dtype).removeprefix('torch.'), f'σ={t.std().item():g}', f'μ={t.mean().item():.2f}', f'norm={fmt_matrix_norm(matrix_norm(t.float(), ord=2))}', f'absmax={t.abs().max().item():g}', label or '')


class PrecisionMode(str, Enum):
    Float32 = 'f32'
    MixedBF16 = 'mixed-bf16'
    PureBF16 = 'pure-bf16'
    PureF16 = 'pure-f16'

def get_model(dir: Path) -> T5EncoderStack:
    with open(dir / 'config.json', 'r') as f:
        conf_dict: dict[str, Any] = json.load(f)
    config: T5Config = T5Config.model_validate(conf_dict)

    with torch.device('meta'):
        enc: T5EncoderStack = T5EncoderStack(config).eval()

    deserializer = TensorDeserializer(dir / 'enc.tensors', lazy_load=True)
    deserializer.load_into_module(enc)
    deserializer.close()
    return enc

def explain_diff(ref: FloatTensor, candidate: FloatTensor) -> FloatTensor:
    diff = ref.float().sub(candidate.float())
    qs = torch.tensor([.5, .75, .9, .95, .99, .999, .9999], device=ref.device)
    q = diff.abs().quantile(qs)
    print(f'quantiles ({qs.cpu()}):\n           {q.cpu()}')
    stat(diff, 'diff')


def main():
    device = torch.device("cuda")

    f32_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog/t5-v1_1-small-f32')
    f16_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog/t5-v1_1-small-f16')
    bf16_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog/t5-v1_1-small-bf16')

    f32_enc: Optional[T5EncoderStack] = None
    f16_enc: Optional[T5EncoderStack] = None
    bf16_enc: Optional[T5EncoderStack] = None
    if f32_enabled := False:
        f32_enc: T5EncoderStack = get_model(f32_dir)
    if f16_enabled := False:
        f16_enc: T5EncoderStack = get_model(f16_dir)
    if bf16_enabled := False:
        bf16_enc: T5EncoderStack = get_model(bf16_dir)

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
    with inference_mode():#, autocast_mode(device_type=device.type, dtype=torch.float16):
        torch.manual_seed(seed)
        if f32_enabled:
            f32_out: FloatTensor = f32_enc(
                input_ids=input_ids,
                input_mask=mask,
            )
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
        if bf16_enabled:
            print('f32 vs bf16:')
            explain_diff(f32_out, bf16_out)
    pass  # somewhere to put your breakpoint


if __name__ == "__main__":
    main()
