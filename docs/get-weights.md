## Get weights

We'll use HF transformers to download model weights, HF tokenizers to download the sentencepiece model:

```bash
pip install transformers tokenizers
# installing hf_transfer will enable us to download checkpoints faster
pip install huggingface_hub[hf_transfer]
```

We assume you have already installed the `nai-t5` pip package, which should put the [`t5_serialize.py`](../nai_t5/scripts/t5_serialize.py) script should on your shell's `PATH`.

You can export the t5 v1.1 small encoder like so:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 t5_serialize -m google/t5-v1_1-small \
--enc --weight-dtype bfloat16 --tensorizer -o ckpt/goog-t5-v1.1-small-bf16
```

This will output the following files:

```
ckpt/goog-t5-v1.1-small-bf16/enc.tensors  # weights of T5 encoder only
ckpt/goog-t5-v1.1-small-bf16/config.json
ckpt/goog-t5-v1.1-small-bf16/spiece.model # Sentencepiece model for T5 tokenization
```

[`t5_serialize.py`](../nai_t5/scripts/t5_serialize.py) supports also an `--encdec` option to export encoder-decoder weights, 
and a `--dtensor` option to export the checkpoint as a distributed checkpoint (which can be loaded in pytorch without further dependencies).

Google's T5 checkpoints were originally distributed in bfloat16.  
Huggingface distributes them in float32, perhaps for compatibility reasons, but there is no extra precision in these larger checkpoints.  
The range of values in the weights is not excessive, so should cast to float16 comfortably.

**FSDP**:  
[`t5_serialize_dtensor.py`](../nai_t5/scripts/t5_serialize_dtensor.py) supports loading a tensorizer checkpoint and converting it into a _sharded_ distributed tensor checkpoint. This enables multi-device deployments to load just their own shard of the weights, for FSDP-sharded inference.