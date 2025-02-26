# NovelAI T5

Model code for T5. Designed to be fast and have good float16 support.  
Somewhat tidy.  
Somewhat tested.

## What's included

### Performance features

- torch SDPA attention in encoder + decoder
- Flex attention in encoder (optional)
  - ignores padding keys
  - ignores padding queries (uses safe_softmax to give these positions 0-probability)
- fused projections
  - QKV fusion in self-attention
  - KV fusion in cross-attention
  - in-projection fusion in GEGLU
- RMSNorm scales can be fused into subsequent linear projections
- KV cache support
  - just one big re-used tensor (avoids repeatedly reallocating a tensor as the sequence grows)
- UMT5 per-layer position embedding fusion (all layers computed concurrently)
- FFN out-proj is allowed to run in half-precision without the use of autocast

### PyTorch idioms

- RMSNorm built-in
- GELU built-in

### Compatibility

- masking
  - 3-dim packing mask or 2-dim padding mask
- support for v1.1 (GEGLU) and v1.0 (ReLU)
- support for UMT5 (e.g. EleutherAI's pile-t5) per-layer position embeddings
- supports SentencePiece tokenizer

### Training considerations

- weight init (basic attempt)
- supports disabling attention scale, for compatibility with Google checkpoints
  - Google burned the attention scale into the weights, which had no detriment to training dynamics because Adafactor optimizer scales param lr w.r.t the RMS of the params ([more detail here](https://x.com/Birchlabs/status/1821188959201845745))

### Float16 considerations

_A good write-up of prior approaches for float16 T5 support is available on the [Graphcore blog](https://www.graphcore.ai/posts/running-flan-t5-xl-inference-in-float16-for-ipu-how-we-did-it)._

**Not used: Activation clipping**  
Previous approaches (HuggingFace, Graphcore) have used clipping to keep activations within float16 range.  

**Not used: Single-precision FFN out**  
HuggingFace additionally casts out-projection weights to float32, which has the consequence that (except in mixed-precision contexts): out-projections would be run in float32.

**Not used: ReLU fallback**  
Graphcore has proposed a numerically-safer float16 GeLU (which falls back to ReLU for large numbers to avoid overflowing `x**3`).  
Instead we use PyTorch's [built-in GeLU](https://github.com/pytorch/pytorch/blob/35532fc477d66845a0c4ea468fd8cbaa312ae248/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L23), which uses [opmath](https://github.com/pytorch/pytorch/issues/63985) to specify that the cube operation be performed in float32.

**Float32 residual**  
We avoid accumulation error from float16/bfloat16 summation, by maintaining a residual in float32. This technique can also be seen in [flash attention's layernorm kernel](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/layer_norm.py).

**Activation scaling residual**  
Rather than _clipping_ activations: we _scale_ activations and residuals (and RMSNorm eps).

## Philosophy

Main objective was to modernize T5 with Torch SDPA attention and write in a clearer code style.

- type hints
- document return types via NamedTuple
- document tensor shapes via einops rearrange
- pass KV cache as a forward argument to be mutated; no impact on return types
- clearer separation of concerns between encoder/decoder/model
  - avoid weight-tying and shared references
- prefer to duplicate modules rather than add conditions to existing modules to make them multi-use
  - makes it clearer that there are 3 types of attention, and they can be optimized differently
  - makes it clearer that encoder does not use a KV cache
- eliminate unused configurables
  - for example we do not keep "tie emb to lm_head"
  - keep only what's needed for final shipped models (e.g. v1.1 and v1.0), not ablations

## Shelved ideas

We considered fusing the decoder's every cross-attention KV projection, but it's questionable whether this would provide any speedup (KV is work that can be done concurrently with Q anyway), and it would complicate FSDP (the very wide fused KV projection would need to be chunked to achieve good compute/communication overlap).

MaskedTensor could be used to exploit sparsity on padded fixed-length sequences. Fixed-length sequences help to enable torch.compile dynamic=False. This would be particularly beneficial when inferencing the decoder, as the sequence length keeps changing (but could be modelled as a fixed-length MaskedTensor).

## Setup

### Install

Install the `nai-t5` package.  
Currently distributed via GitHub only; we install via the repository URL.

```bash
pip install git+https://github.com/NovelAI/t5.git
# Sentencepiece tokenizer recommended, but you can use HF tokenizers too
pip install sentencepiece
# tensorizer recommended for weight-loading
pip install tensorizer async_timeout
```

### Get weights

We'll use HF transformers to download model weights, HF tokenizers to download the sentencepiece model:

```bash
pip install transformers tokenizers
# installing hf_transfer will enable us to download checkpoints faster
pip install huggingface_hub[hf_transfer]
```

Installing `nai-t5` should put the `t5_serialize.py` script should on your `PATH`.

You can export the t5 v1.1 small encoder like so:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 t5_serialize.py -m google/t5-v1_1-small \
--enc --weight-dtype bfloat16 --tensorizer -o ckpt/goog-t5-v1.1-small-bf16
```

This will output the following files:

```
ckpt/goog-t5-v1.1-small-bf16/enc.tensors  # weights of T5 encoder only
ckpt/goog-t5-v1.1-small-bf16/config.json
ckpt/goog-t5-v1.1-small-bf16/spiece.model # Sentencepiece model for T5 tokenization
```


### Basic usage (encoder)

Encode a batch of prompts like this:

```python
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

prompts: list[str] = ['hello world']
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
```

### Flex attention

**Why**  
Flex attention makes T5 a lot faster. T5 relies on an arbitrary attention bias to implement relative position.

Most attention backends don't implement support for arbitrary biases. cuDNN SDPA (available on H100 GPUs) supports arbitrary bias, but [might give incorrect results](https://github.com/pytorch/pytorch/issues/139298) in unknown circumstances.  
The most likely scenario is that SDPA will use the cutlassF (i.e. memory-efficient) attention backend, which is the slowest (except perhaps math).  
Flex attention helps you to avoid this, and enjoy fast attention.

We considered [a few approaches](https://github.com/pytorch/pytorch/issues/138493) for implementing T5 flex attention. The simplest (just add an arbitrary bias) was the fastest under the parameters we tried.

**How**  
Before constructing the model, update the config to use flex.

```diff
+ from nai_t5.t5_common import T5AttnImpl

  config: T5Config = T5Config.model_validate(conf_dict)
+ config = config.model_copy(update={
+     'attn_impl': T5AttnImpl.Flex,
+     'flex_kernel_options': {
+         'BLOCK_M': 128,
+         'BLOCK_N': 64,
+     },
+ })
```

After loading weights onto the model, initialize the model's flex attention score_mods:

```diff
  deserializer = TensorDeserializer(t5_dir / 'enc.tensors', lazy_load=True, dtype=dtype, device=device)
  deserializer.load_into_module(t5_enc)
  deserializer.close()

+ t5_enc.bind_score_mods(seq_len=512)
```

Now, every time you inference the model: construct a block mask, and pass that in.  
_You don't need to pass the regular boolean mask in any more; flex doesn't look at it._

```diff
+ from torch.nn.attention.flex_attention import BlockMask
+ from nai_t5.t5_encoder import make_self_attn_block_mask

  mask: BoolTensor = input_ids != tokenizer.pad_id()
+ block_mask: BlockMask = make_self_attn_block_mask(
+     mask=mask,
+     mask_pad_queries=True,
+ )

  with inference_mode():
    emb: FloatTensor = t5_enc(
        input_ids=input_ids,
-       input_mask=mask,
+       block_mask=block_mask,
    )
```

### Compilation

After weights are loaded, you can reassign the encoder with a compiled instance.

```python
t5_enc = torch.compile(t5_enc, dynamic=False, fullgraph=True)
```

This should make the model far faster.  
Ensure that you use a fixed input size (i.e. pad to a fixed context length to keep shapes consistent), otherwise you will incur recompiles.

## Run

```bash
python -m scripts.t5_encoder_parity
python -m scripts.t5_encdec_parity
python -m scripts.t5_sampling_hf_generate
python -m scripts.t5_sampling_parity_nocache
python -m scripts.t5_sampling_parity_cache
```

## Example scripts

- sampling code example
- FLOP counter demo (will work for SDPA but not Flex attention)