import json
from typing import Any, Optional
from pathlib import Path
import torch
from torch import BoolTensor, FloatTensor, IntTensor, inference_mode
from torch.nn.attention.flex_attention import BlockMask
from tensorizer import TensorDeserializer
from sentencepiece import SentencePieceProcessor

from nai_t5 import T5Config, T5EncoderStack
from nai_t5.t5_encoder import make_self_attn_block_mask
from nai_t5.t5_common import T5AttnImpl
from nai_t5.fuse_norm_scales import fuse_norm_scales_enc

t5_dir = Path('ckpt/goog-t5-v1.1-small-bf16')

with open(t5_dir / 'config.json', 'r') as f:
    conf_dict: dict[str, Any] = json.load(f)
config: T5Config = T5Config.model_validate(conf_dict)
config = config.model_copy(update={
    'attn_impl': T5AttnImpl.Flex,
    'flex_kernel_options': {
        'BLOCK_M': 128,
        'BLOCK_N': 64,
    },
})

with torch.device('meta'):
    t5_enc: T5EncoderStack = T5EncoderStack(config).eval()

dtype = torch.bfloat16
device = torch.device('cuda')
deserializer = TensorDeserializer(t5_dir / 'enc.tensors', lazy_load=True, dtype=dtype, device=device)
deserializer.load_into_module(t5_enc)
deserializer.close()

fuse_norm_scales_enc(t5_enc)

t5_enc.bind_score_mods(seq_len=512)

t5_enc = torch.compile(t5_enc, dynamic=False, fullgraph=True)

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
block_mask: BlockMask = make_self_attn_block_mask(
    mask=mask,
    mask_pad_queries=True,
)

with inference_mode():
    emb: FloatTensor = t5_enc(
        input_ids=input_ids,
        block_mask=block_mask,
    )
