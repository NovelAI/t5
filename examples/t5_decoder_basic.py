import json
from typing import Any, Optional, Generator
from pathlib import Path
from functools import partial
from einops import rearrange
import torch
from torch import BoolTensor, FloatTensor, IntTensor, inference_mode
from tensorizer import TensorDeserializer
from sentencepiece import SentencePieceProcessor

from nai_t5 import T5, T5Config
from nai_t5.t5_decoder import T5DecoderStack
from nai_t5.t5_encoder import T5EncoderStack
from nai_t5.sampling import BoundCachedDecode, MakeLogitGenerator, generate_greedy_cached, generate_until

t5_dir = Path('ckpt/goog-t5-v1.1-small-bf16')

with open(t5_dir / 'config.json', 'r') as f:
    conf_dict: dict[str, Any] = json.load(f)
config: T5Config = T5Config.model_validate(conf_dict)

with torch.device('meta'):
    t5 = T5(config).eval()
t5_dec: T5DecoderStack = t5.decoder
t5_enc: T5EncoderStack = t5.encoder

dtype = torch.bfloat16
device = torch.device('cuda')
deserializer = TensorDeserializer(t5_dir / 'encdec.tensors', lazy_load=True, dtype=dtype, device=device)
deserializer.load_into_module(t5)
deserializer.close()

tokenizer = SentencePieceProcessor(model_file=str(t5_dir / 'spiece.model'))

prompts: list[str] = ['Today is a fine<extra_id_0> on which to walk my<extra_id_1> in the park.']
batch_size = len(prompts)

toks: list[list[int]] = tokenizer.Encode(prompts, add_eos=True)

fixed_ctx_len: Optional[int] = 512

ctx_len: int = max(len(t) for t in toks) if fixed_ctx_len is None else fixed_ctx_len

input_ids: IntTensor = torch.full((batch_size, ctx_len), fill_value=tokenizer.pad_id(), dtype=torch.int32, device='cpu')
for seq, input_out in zip(toks, input_ids.unbind()):
    input_out[:len(seq)].copy_(torch.tensor(seq[:ctx_len], dtype=torch.int32))
input_ids = input_ids.to(device)

input_ids_mask: BoolTensor = input_ids != tokenizer.pad_id()

seed = 42
torch.manual_seed(seed)
with inference_mode():
    emb: FloatTensor = t5_enc(
        input_ids=input_ids,
        input_mask=input_ids_mask,
    )
    assert emb.isfinite().all()
    cross_kv: FloatTensor = t5_dec.get_cross_kv(emb)
    assert cross_kv.isfinite().all()

max_new_tokens = 64

batch_size = 1
cross_mask = rearrange(input_ids_mask.bool(), "b k -> b 1 k")
cross_mask = cross_mask.expand(-1, max_new_tokens, -1)

@inference_mode()
def decode(
    in_tokens: IntTensor,
    encoding: FloatTensor,
    cross_mask: BoolTensor,
    self_past_kv: FloatTensor,
    cross_kv: FloatTensor,
) -> FloatTensor:
    decoder_input_embeds: FloatTensor = t5_enc.vocab_embed(in_tokens)
    input_mask_: BoolTensor = input_ids_mask[:, : in_tokens.size(-1)]
    cross_mask_: BoolTensor = cross_mask[:, : in_tokens.size(-1), :]
    decoder_out: FloatTensor = t5_dec(
        decoder_input_embeds,
        encoding,
        input_mask=input_mask_,
        # input_mask=None,
        cross_mask=cross_mask_,
        self_past_kv=self_past_kv,
        cross_kv=cross_kv,
    )
    assert decoder_out.isfinite().all()
    logits: FloatTensor = t5.lm_head(decoder_out)
    assert logits.isfinite().all()
    return logits

self_past_kv: FloatTensor = t5_dec.get_kv_cache(
    length=max_new_tokens, device=emb.device, dtype=emb.dtype
)

bound_decode: BoundCachedDecode = partial(decode, encoding=emb, cross_mask=cross_mask, cross_kv=cross_kv)
make_gen: MakeLogitGenerator = partial(generate_greedy_cached, decode=bound_decode, self_past_kv=self_past_kv)

# in Google's T5 at least, PAD is the decoder start token.
dec_start_id: int = tokenizer.pad_id()
mask2: int = tokenizer.PieceToId('<extra_id_2>')
gen: Generator[IntTensor, None, IntTensor] = generate_until(
    make_gen=make_gen,
    device=device,
    batch_size=batch_size,
    max_tokens=max_new_tokens,
    stop_tokens={mask2, tokenizer.eos_id()},
    decoder_start_token_id=dec_start_id,
    pad_token_id=tokenizer.pad_id(),
)
my_acc: list[int] = []
for ix, tok_t in enumerate(gen):
    token_id: int = tok_t.cpu().squeeze().item()
    token_str: int = tokenizer.IdToPiece(token_id)
    print(token_str, flush=True, end="")
    my_acc.append(token_id)
print("", flush=True)