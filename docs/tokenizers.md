# Tokenization

In Google's T5 model training, sequences were tokenized via [Sentencepiece](https://github.com/google/sentencepiece).

Huggingface offers a library [tokenizers](https://github.com/huggingface/tokenizers) which offers two tokenizer implementations which both tokenize sequences differently than sentencepiece does.

We recommend to use Sentencepiece, in order to match how T5 was trained.

## Special tokens

T5 was trained on a masked language modelling objective, utilizing user-defined special tokens such `<extra_id_0>`, which get tokenized with high-precedence without being broken into subwords.

The `spiece.model` that HF distributes, does not include these special tokens.  
We offer a script [`tokenizer_hf_to_sentencepiece.py`](../nai_t5/scripts/tokenizer_hf_to_sentencepiece.py) which outputs a new `spiece.model` with the new special tokens included.

HF tokenizers instead uses custom tokenization logic to apply special tokens configured outside of `spiece.model`.  
The behaviour of tokenization is not the same though:

```python
from torch import IntTensor
from pathlib import Path
from transformers.models.t5 import T5TokenizerFast
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils.generic import TensorType
from sentencepiece import SentencePieceProcessor

prompt = 'Today is a fine <extra_id_0> on which to walk my <extra_id_1> in the park.'

hf_tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained('google/t5-v1_1-small', legacy=False)
input_encoding: BatchEncoding = hf_tokenizer(prompt, return_tensors=TensorType.PYTORCH, add_special_tokens=True)
input_ids_hf: IntTensor = input_encoding.input_ids

t5_dir = Path('ckpt/goog-t5-v1.1-small-bf16')
sp = SentencePieceProcessor(model_file=str(t5_dir / 'spiece.model'))
toks_sp: list[int] = sp.Encode(prompt, add_eos=True)

print(input_ids_hf[0].tolist())
# [1960, 19, 3, 9, 1399, 32099, 30, 84, 12, 1482, 82, 32098, 16, 8, 2447, 5, 1]

print(toks_sp)
# [1960, 19, 3, 9, 1399, 3, 32099, 30, 84, 12, 1482, 82, 3, 32098, 16, 8, 2447, 5, 1]
```

HF's tokenizer removes the space tokens preceding the special tokens.

HF and sentencepiece disagree on how to tokenize:

> `Today is a fine <extra_id_0> on which to walk my <extra_id_1> in the park.`

But if spaces before special tokens are removed, HF and sentencepiece agree on how to tokenize:

> `Today is a fine<extra_id_0> on which to walk my<extra_id_1> in the park.`

## Tie-breaks

```python
from pathlib import Path
from typing import NamedTuple
from transformers.models.t5 import T5TokenizerFast, T5Tokenizer
from sentencepiece import SentencePieceProcessor

hf_tokenizer_fast: T5TokenizerFast = T5TokenizerFast.from_pretrained('google/t5-v1_1-small', legacy=False)
hf_tokenizer_slow: T5Tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-small', legacy=False)

t5_dir = Path('/mnt/clusterstorage/models/nait5-tensorizer/goog/t5-v1_1-small-bf16')
sp = SentencePieceProcessor(model_file=str(t5_dir / 'spiece.model'))

class Tokenizations(NamedTuple):
    toks_hf_fast: list[int]
    toks_hf_slow: list[int]
    toks_sp: list[int]

def many_tokenize(prompt: str) -> Tokenizations:
    toks_hf_fast: list[int] = hf_tokenizer_fast(prompt, add_special_tokens=True).input_ids
    toks_hf_slow: list[int] = hf_tokenizer_slow(prompt, add_special_tokens=True).input_ids
    toks_sp: list[int] = sp.Encode(prompt, add_eos=True)
    return Tokenizations(toks_hf_fast, toks_hf_slow, toks_sp)

print(many_tokenize('zzz'))
# HF fast       [3, 172, 5271, 1]
# HF slow       [3, 5271, 172, 1]
# sentencepiece [3, 172, 5271, 1]

print(many_tokenize(':zzz'))
# HF fast       [3, 10, 172, 5271, 1]
# HF slow       [3, 10, 5271, 172, 1]
# sentencepiece [3, 10, 5271, 172, 1]
```

HF uses different precision to sentencepiece, which may lead to different results.  
Moreover, HF fast and HF slow get different results and it's not consistent which of them agrees with sentencepiece.

https://github.com/huggingface/tokenizers/issues/1213  
https://github.com/google/sentencepiece/issues/747

## Special token space continuations

Fast and slow tokenizers disagree on whether to use "▁on" (30, fast) or "on" (106, slow) after a special token. Sentencepiece uses 30.  
Same problem with "▁in" (16) and "in" (77). Sentencepiece uses 16.

```bash
print(many_tokenize('Today is a fine <extra_id_0> on which to walk my <extra_id_1> in the park.'))
# HF fast       [1960, 19, 3, 9, 1399, 32099, 30, 84, 12, 1482, 82, 32098, 16, 8, 2447, 5, 1]
# HF slow       [1960, 19, 3, 9, 1399, 32099, 106, 84, 12, 1482, 82, 32098, 77, 8, 2447, 5, 1]
# sentencepiece [1960, 19, 3, 9, 1399, 3, 32099, 30, 84, 12, 1482, 82, 3, 32098, 16, 8, 2447, 5, 1]
```

## Decoding

Fast and slow tokenizers disagree on how to decode non-space-prefixed tokens that follow special tokens.  
Fast agrees with sentencepiece (don't insert a space).

```python
hf_tokenizer_slow.decode([32099, 106, 84])
'<extra_id_0> on which'

hf_tokenizer_fast.decode([32099, 106, 84])
'<extra_id_0>on which'

sp.DecodeIds([32099, 106, 84])
'<extra_id_0>on which'
```