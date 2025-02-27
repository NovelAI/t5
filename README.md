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
HuggingFace casts out-projection weights to float32, which has the consequence that (except in mixed-precision contexts): out-projections would be run in float32.

**Not used: ReLU fallback**  
Graphcore has proposed a numerically-safer float16 GeLU (which falls back to ReLU for large numbers to avoid overflowing `x**3`).  
Instead we use PyTorch's [built-in GeLU](https://github.com/pytorch/pytorch/blob/35532fc477d66845a0c4ea468fd8cbaa312ae248/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L23), which uses [opmath](https://github.com/pytorch/pytorch/issues/63985) to specify that the cube operation be performed in float32.

**Float32 residual**  
We avoid accumulation error from float16/bfloat16 summation, by maintaining a residual in float32. This technique can also be seen in [flash attention's layernorm kernel](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/layer_norm.py).

**Scaling layer outputs and the residual**  
Rather than _clipping_ activations: we _scale_ the weights of output projections (attention out and FFN out) and the residual itself (and RMSNorm eps).  
This preserves the relative size difference between layer outputs and the residual, allowing outliers to be as large as they desire to be.  
At any point in the model where we scale down the residual, we also scale down all out-projections after that by the same amount.  
The process of selecting scales is manual. The script [`scripts/t5_encoder_precision_parity.py`](scripts/t5_encoder_precision_parity.py) tries to encode a sequence, and reports whether NaN is output by any layer. Should this happen, scales can be adjusted (i.e. by taking note of which layer encountered trouble and halving its scales). This process can be repeated until the test sequence succeeds. The script can also run the same sequence in float32 or bfloat16 in order to compare the absmax difference between the sequences, to determine whether the accuracy remains acceptable.

Originally our layer looked conventional:

```python
def forward(
    self,
    x: torch.Tensor,
    attn_args: SDPAArgs | FlexArgs,
) -> FloatTensor:
    residual = x
    x = self.ln1(x)
    attn_out: FloatTensor = self.attn(x, *attn_args)

    x = residual + self.dropout(attn_out)

    residual = x
    x = self.ffn(self.ln2(x))

    x = residual + self.dropout(x)

    return x
```

Now it looks like this:

```python
class ActAndResidual(NamedTuple):
    x: FloatTensor
    residual: FloatTensor

def forward(
    self,
    x_r: ActAndResidual,
    attn_args: SDPAArgs | FlexArgs,
) -> ActAndResidual:
    x, residual = x_r
    x, residual = self.ln1(x, residual=residual)
    x = self.attn(x, *attn_args)
    x, residual = self.ln2(self.dropout(x), residual=residual)
    x = self.ffn(x)
    return ActAndResidual(self.dropout(x), residual)
```

RMSNorm becomes responsible for adding layer outputs to the residual, which is maintained in float32.  
The RMSNorm can also scale the residual. When constructing the model, we can assign a residual_scale to each norm to make the residual smaller at blocks of the model that we find exceed the float16 range (this typically happens inside FFN out-projections).

```python
def forward(
    self,
    x: FloatTensor,
    residual: Optional[FloatTensor] = None,
    prenorm=True,
) -> ActAndResidual | FloatTensor:
    orig_dtype = x.dtype
    if residual is None:
        next_residual = x.float()
    else:
        next_residual = x + residual
    normed: FloatTensor = super().forward(next_residual).type(orig_dtype)
    if prenorm:
        if self.residual_scale is not None:
            next_residual = next_residual * self.residual_scale
        return ActAndResidual(normed, next_residual)
    return normed
```

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

### Norm scale fusion

Before constructing the model, modify the config to set `elementwise_affine=False`. This will construct RMSNorm without scale weights.  
_You can also enable flex attention here in the config if you want, as above._

```diff
  config: T5Config = T5Config.model_validate(conf_dict)
+ config = config.model_copy(update={
+     'elementwise_affine': False,
+ })
```

When loading weights onto the model, specify `fuse_norm_scales=True`.

```diff
- from tensorizer import TensorDeserializer
+ from nai_t5.weight_load import FusingDeserializer

- deserializer = TensorDeserializer(t5_dir / 'enc.tensors', lazy_load=True, dtype=dtype, device=device)
- deserializer.load_into_module(t5_enc)
+ deserializer = FusingDeserializer(t5_dir / 'enc.tensors', lazy_load=True, dtype=dtype, device=device)
+ deserializer.load_with_fusions(
+     t5_enc,
+     fuse_norm_scales=True,
+     norm_fusion_via_f32=True,
+ )
  deserializer.close()
```

RMSNorm scales will be fused into the weights of whatever Linear projection occurs after them, reducing latency and exposing you to fewer instances of floating-point rounding.

### Float16 usage (encoder)

_You can also fuse norm scales with the FusingDeserializer here, as above._

```diff
- from tensorizer import TensorDeserializer
+ from nai_t5.weight_load import FusingDeserializer

- deserializer = TensorDeserializer(t5_dir / 'enc.tensors', lazy_load=True, dtype=dtype, device=device)
- deserializer.load_into_module(t5_enc)
+ deserializer = FusingDeserializer(t5_dir / 'enc.tensors', lazy_load=True, dtype=dtype, device=device)
+ deserializer.load_with_fusions(
+     t5_enc,
+     enc_attn_out_scales=None,
+     # FFN out weight scales for the 8 layers of google/t5-v1_1-small's encoder
+     enc_ffn_out_scales=[*[1]*6, 1/2, 1/2],
+ )
  deserializer.close()
```

If you still encounter NaN outputs despite this: try using [`scripts/t5_encoder_precision_parity.py`](scripts/t5_encoder_precision_parity.py) to encode your prompt, and take note of the layer at which non-finite values are reported. Reduce scales at that layer and try again.

The same script includes suggested scales for T5v1.1 small, XL, and XXL.  

[`scripts/t5_encdec_precision_parity.py`](scripts/t5_encdec_precision_parity.py) likewise includes suggested _decoder_ scales for T5v1.1 small, XL, and XXL.

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