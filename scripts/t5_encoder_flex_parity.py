from typing import Any
import json
from pathlib import Path
from torch import LongTensor, BoolTensor, FloatTensor
import torch
from torch import inference_mode
from tensorizer import TensorDeserializer
from sentencepiece import SentencePieceProcessor
from einops import rearrange
from nai_t5 import T5Config
from nai_t5.t5_common import T5AttnImpl
from nai_t5.t5_encoder import T5EncoderStack, T5EncoderLayer, T5EncoderSelfAttention, T5EncoderSelfAttentionFlex

device=torch.device('cuda')

tensorizer_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog/t5-v1_1-small-bf16')
config_path: Path = tensorizer_dir / 'config.json'
with open(config_path, 'r') as f:
    config_dict: dict[str, Any] = json.load(f)
sdpa_config: T5Config = T5Config.model_validate(config_dict)
flex_config: T5Config = sdpa_config.model_copy(update={'attn_impl': T5AttnImpl.Flex})

with torch.device('meta'):
    sdpa_enc: T5EncoderStack = T5EncoderStack(sdpa_config).eval()
    flex_enc: T5EncoderStack = T5EncoderStack(flex_config).eval()

deserializer = TensorDeserializer(tensorizer_dir / 'enc.tensors', lazy_load=True)
deserializer.load_into_module(sdpa_enc)
deserializer.load_into_module(flex_enc)
deserializer.close()

tokenizer = SentencePieceProcessor(model_file=str(tensorizer_dir / 'spiece.model'))

ctx_len = 512
prompts: list[str] = ['everything is great']
batch_size = len(prompts)

toks: list[list[int]] = tokenizer.Encode(prompts, add_eos=True)
input_ids: LongTensor = torch.full((batch_size, ctx_len), fill_value=tokenizer.pad_id(), dtype=torch.long, device='cpu')
for seq, input_out in zip(toks, input_ids.unbind()):
    input_out[:len(seq)].copy_(torch.tensor(seq[:ctx_len], dtype=torch.long))
input_ids = input_ids.to(device)
mask: BoolTensor = input_ids == tokenizer.pad_id()

with inference_mode():
    input_embeds: FloatTensor = sdpa_enc.vocab_embed(input_ids.flatten(end_dim=-2))
    position_bias: FloatTensor = sdpa_enc.relative_attention_bias(ctx_len)
    mask_broadcast = rearrange(mask, "b k -> b 1 1 k")
    sdpa_layer0: T5EncoderLayer = sdpa_enc.layers[0]
    flex_layer0: T5EncoderLayer = flex_enc.layers[0]
    normed_embs: FloatTensor = sdpa_layer0.ln1(input_embeds)
    sdpa_attn: T5EncoderSelfAttention = sdpa_layer0.attn
    flex_attn: T5EncoderSelfAttentionFlex = flex_layer0.attn
    sdpa_out: FloatTensor = sdpa_attn(normed_embs, position_bias=position_bias, mask=mask_broadcast)
    flex_out: FloatTensor = flex_attn(normed_embs, mask=mask_broadcast)
    assert sdpa_out.allclose(flex_out)
