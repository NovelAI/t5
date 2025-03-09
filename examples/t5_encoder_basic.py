import json
from typing import Any, Optional
from pathlib import Path
import torch
from torch import BoolTensor, FloatTensor, IntTensor, inference_mode
from tensorizer import TensorDeserializer
from sentencepiece import SentencePieceProcessor

from nai_t5 import T5Config, T5EncoderStack

t5_dir = Path('ckpt/goog-t5-v1.1-small-bf16')

with open(t5_dir / 'config.json', 'r') as f:
    conf_dict: dict[str, Any] = json.load(f)
config: T5Config = T5Config.model_validate(conf_dict)

with torch.device('meta'):
    t5_enc: T5EncoderStack = T5EncoderStack(config).eval()

dtype = torch.bfloat16
device = torch.device('cuda')
deserializer = TensorDeserializer(t5_dir / 'enc.tensors', lazy_load=True, dtype=dtype, device=device)
deserializer.load_into_module(t5_enc)
deserializer.close()

tokenizer = SentencePieceProcessor(model_file=str(t5_dir / 'spiece.model'))

prompts: list[str] = ['illustration of a nice tree']
batch_size = len(prompts)

toks: list[list[int]] = tokenizer.Encode(prompts, add_eos=True)

fixed_ctx_len: Optional[int] = 512

ctx_len: int = max(len(t) for t in toks) if fixed_ctx_len is None else fixed_ctx_len

input_ids: IntTensor = torch.full((batch_size, ctx_len), fill_value=tokenizer.pad_id(), dtype=torch.int32, device='cpu')
for seq, input_out in zip(toks, input_ids.unbind()):
    input_out[:len(seq)].copy_(torch.tensor(seq[:ctx_len], dtype=torch.int32))
input_ids = input_ids.to(device)
mask: BoolTensor = input_ids != tokenizer.pad_id()

with inference_mode():
    emb: FloatTensor = t5_enc(
        input_ids=input_ids,
        input_mask=mask,
    )
