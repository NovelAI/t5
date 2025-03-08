## Usage

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

Nominally, float16 inference should accumulate less floating-point error than bfloat16, due to its extra precision. So long as we scale down the weights and the size of the residual to stay within float16 range, and do not scale it so far as to lose accuracy to underflow.

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
These scales are chosen to be the smallest power-of-2 changes that allow the test sequence to be encoded, with the priority being to preserve float16 accuracy by not shrinking more than necessary. Consequently no headroom has been reserved, so it is possible that other prompts could exceed float16 range. The hope is that exposing control of this enables exploration.

[`scripts/t5_encdec_precision_parity.py`](scripts/t5_encdec_precision_parity.py) likewise includes suggested _decoder_ scales for T5v1.1 small, XL, and XXL.

### Compilation

After weights are loaded, you can reassign the encoder with a compiled instance.

```python
t5_enc = torch.compile(t5_enc, dynamic=False, fullgraph=True)
```

This should make the model far faster.  
Ensure that you use a fixed input size (i.e. pad to a fixed context length to keep shapes consistent), otherwise you will incur recompiles.

### FSDP

[`scripts/t5_encoder_parity_fsdp.py`](scripts/scripts/t5_encoder_parity_fsdp.py) demonstrates how to load the model in FSDP or FSDP2 from a distributed checkpoint.

[`t5_serialize_dtensor.py`](scripts/t5_serialize_dtensor.py) can be used to convert a tensorizer checkpoint into a sharded distributed tensor checkpoint.