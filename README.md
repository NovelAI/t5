# NovelAI T5

Model code for T5. Designed to be fast and have good float16 support.  
Somewhat tidy.  
Somewhat tested.

## Install

Install the `nai-t5` package.  
Currently distributed via GitHub only; we install via the repository URL.

```bash
pip install git+https://github.com/NovelAI/t5.git

# Sentencepiece tokenizer recommended, but you can use HF tokenizers too
pip install sentencepiece
# tensorizer recommended for weight-loading
pip install tensorizer async_timeout
```

## Get weights

See [docs/get-weights.md](weight loading) docs.

## Usage

See [docs/usage.md](usage) docs.

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

See how we approached [docs/float16.md](float16 support).

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

## Run

TODO: delete this section after documenting the scripts

```bash
python -m scripts.t5_encoder_parity
python -m scripts.t5_encdec_parity
python -m scripts.t5_sampling_hf_generate
python -m scripts.t5_sampling_parity_nocache
python -m scripts.t5_sampling_parity_cache
```

## Example scripts

TODO: delete this section after documenting the scripts

- sampling code example
- FLOP counter demo (will work for SDPA but not Flex attention)