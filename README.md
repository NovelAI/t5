# T5

Model code for T5.

- Torch SDPA attention
- masking
  - 3-dim packing mask or 2-dim padding mask
- sampling code
- KV cache
  - not a repeatedly-reallocated list of tensors, just one big re-used tensor
- fused attention projections
- FLOP counter
- weight init (basic attempt)
- GEGLU and ReLU
- supports disabling attention scale, for compatibility with Google checkpoints
  - Google burned the attention scale into the weights, which had no detriment to training dynamics because Adafactor optimizer scales param lr w.r.t the RMS of the params ([more detail here](https://x.com/Birchlabs/status/1821188959201845745))

Main objective was to modernize it with Torch SDPA attention and write in a clearer code style.

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
  - keep only what's needed for v1.1 and v1.0

Performance considerations

- GEGLU and RMSNorm would appreciate a fused kernel
  - you could just slap a @torch.compile decorator on them
- we have not applied torch.compile but it would be a sensible idea
- FlexAttention is not used here, but could provide a way to speed up the decoder's causal self-attention with learned bias
- if you wish to train on packed sequences: FlexAttention or Flash Attention's varlen API would be a tidier (and faster) approach than the complex packing mask used here


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