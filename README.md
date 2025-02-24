# NovelAI T5

Model code for T5. Designed to be fast and have good float16 support.  
Somewhat tidy.  
Somewhat tested.

## What's included

### Performance features

- torch SDPA attention in encoder + decoder
- Flex attention in encoder (optional)
- fused attention projections
  - QKV fusion in self-attention
  - KV fusion in cross-attention
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

## Setup

```bash
python -m venv venv
. ./venv/bin/activate
pip install -r requirements.txt
```

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