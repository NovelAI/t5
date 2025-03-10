## Usage

### Basic usage (encoder)

We will encode a prompt:

> `illustration of a nice tree`

The embedding that's returned can be a good condition for a text-to-image model such as Imagen or Flux.

See [`t5_encoder_basic.py`](../examples/t5_encoder_basic.py).

### Flex attention

_This optimization is included in [`t5_encoder_fast.py`](../examples/t5_encoder_fast.py)._

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

### Norm scale fusion (after weight-load)

_This optimization is included in [`t5_encoder_fast.py`](../examples/t5_encoder_fast.py)._

We can fuse RMSNorm scales into the weights of whatever Linear projection occurs after them, reducing latency and exposing you to fewer instances of floating-point rounding.

```diff
+ from nai_t5.fuse_norm_scales import fuse_norm_scales_enc

  deserializer = TensorDeserializer(t5_dir / 'enc.tensors', lazy_load=True, dtype=dtype, device=device)
  deserializer.load_into_module(t5_enc)
  deserializer.close()

+ fuse_norm_scales_enc(t5_enc)
```

If you want slightly more accurate fused weights and don't mind waiting slightly longer, you can pass `norm_fusion_via_f32=True`, but it probably doesn't make a big difference. Haven't checked.

**Do I have to fuse norm scales on every startup?**

It's pretty fast, so you could.

Alternatively you could save your fused model weights afterward so you don't need to fuse them next time.  
If you do that, you should save out a modified config with `elementwise_affine=False`, and use that config afterward.

### Norm scale fusion (during weight-load)

_This optimization is included in [`t5_encoder_fast_float16.py`](../examples/t5_encoder_fast_float16.py)._

Another supported way to fuse norm scales is to do so at the moment when weights are being loaded into the model. This is only really relevant if you're using the FusingDeserializer as your weight-loader already for other reasons (i.e. float16 weight load & scaling).

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

**Do I have to fuse norm scales on every startup?**

You could save the fused model and updated config and load those instead next time.

### Float16 usage (encoder)

_See [`t5_encoder_fast_float16.py`](../examples/t5_encoder_fast_float16.py)._

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

If you still encounter NaN outputs despite this: try using [`../nai_t5/scripts/t5_encoder_precision_parity.py`](../nai_t5/scripts/t5_encoder_precision_parity.py) to encode your prompt, and take note of the layer at which non-finite values are reported. Reduce scales at that layer and try again.

The same script includes suggested scales for T5v1.1 small, XL, and XXL.  
These scales are chosen to be the smallest power-of-2 changes that allow the test sequence to be encoded, with the priority being to preserve float16 accuracy by not shrinking more than necessary. Consequently no headroom has been reserved, so it is possible that other prompts could exceed float16 range. The hope is that exposing control of this enables exploration.

[`scripts/t5_encdec_precision_parity.py`](../nai_t5/scripts/t5_encdec_precision_parity.py) likewise includes suggested _decoder_ scales for T5v1.1 small, XL, and XXL.

### Compilation

_This optimization is included in [`t5_encoder_fast.py`](../examples/t5_encoder_fast.py)._

After weights are loaded, you can reassign the encoder with a compiled instance.

```python
t5_enc = torch.compile(t5_enc, dynamic=False, fullgraph=True)
```

This should make the model far faster.  
Ensure that you use a fixed input size (i.e. pad to a fixed context length to keep shapes consistent), otherwise you will incur recompiles.

### FSDP

[`t5_encoder_parity_fsdp.py`](../nai_t5/scripts/t5_encoder_parity_fsdp.py) demonstrates how to load the model in FSDP or FSDP2 from a distributed checkpoint.

[`t5_serialize_dtensor.py`](../nai_t5/scripts/t5_serialize_dtensor.py) can be used to convert a tensorizer checkpoint into a sharded distributed tensor checkpoint.