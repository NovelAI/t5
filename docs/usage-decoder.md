## Usage

The usage of these examples assumes you have already exported `--encdec` weights via tensorizer to a folder `ckpt/goog-t5-v1.1-small-bf16`. See [get weights](./get-weights.md) for more details, but here's an example that creates both `enc.tensors` (for encoder-only usage) and `encdec.tensors` (for encoder-decoder usage):

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 t5_serialize -m google/t5-v1_1-small \
--enc --encdec --weight-dtype bfloat16 --tensorizer -o ckpt/goog-t5-v1.1-small-bf16
```

### Basic usage (encoder+decoder, KV-cache)

We will encode a prompt:

> `Today is a fine<extra_id_0> on which to walk my<extra_id_1> in the park.`

We expect T5 to predict a sequence to fill the gaps (these will be printed):

> `<extra_id_0>▁day<extra_id_1>▁dog<extra_id_2>`

See [`t5_decoder_basic.py`](../examples/t5_decoder_basic.py).

### Basic usage (encoder+decoder, no KV-cache)

You can inference without a KV-cache if you prefer, but there's no advantage to doing so.

See [`t5_decoder_basic_nocache.py`](../examples/t5_decoder_basic_nocache.py).

### Fast usage (encoder+decoder, KV-cache)

See [`t5_decoder_fast.py`](../examples/t5_decoder_fast.py).

### Fast usage (encoder+decoder, KV-cache), float16

See [`t5_decoder_fast_float16.py`](../examples/t5_decoder_fast_float16.py).