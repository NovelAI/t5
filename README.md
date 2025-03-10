# NovelAI T5

Model code for T5. Designed to be fast and have good float16 support.  
Somewhat tidy.  
Somewhat tested.

## Install

Install the `nai-t5` package, available [on PyPI](https://pypi.org/project/nai-t5/).

```bash
# installs nai-t5 from PyPI
pip install nai_t5

# or install from GitHub source like so:
pip install git+https://github.com/NovelAI/t5.git
```

Other packages you'll probably want:

```bash
# Sentencepiece tokenizer recommended, but you can use HF tokenizers too
pip install sentencepiece
# tensorizer recommended for weight-loading
pip install tensorizer async_timeout
```

We recommend sentencepiece as tokenizer. See [tokenization](docs/tokenizers.md) docs for our reasoning.

## Get weights

See [weight loading](docs/get-weights.md) docs for how to convert HF weights to a format we can load, and to update the tokenizer model with the special tokens it's missing.

## Usage

See [encoder usage](docs/usage-encoder.md) or [decoder usage](docs/usage-decoder.md) docs.  
Typically you'll want to use the encoder, but the decoder can be useful for querying T5 to determine concepts it understands.

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
- supports disabling attention scale, for compatibility with Google checkpoints
  - Google burned the attention scale into the weights, which had no detriment to training dynamics because Adafactor optimizer scales param lr w.r.t the RMS of the params ([more detail here](https://x.com/Birchlabs/status/1821188959201845745))

### Training considerations

- weight init (basic attempt)
- supports conventional attention scale (`head_dim**-.5`)

### Float16 considerations

See how we approached [docs/float16.md](float16 support).

Float16 is appealing because the precision is better, and because it can enable better performance on devices such as 3090 and 4090. Ordinarily these consumer cards are speed-limited computing float16 matmuls with float32 accumulation, but they support float16 matmul with float16 accumulation at higher speeds (with 4090 being comparable to A100).

Support for float16 accumulation is [being added to pytorch](https://github.com/pytorch/pytorch/pull/144441).

Split-k matmul can be used to recover the accuracy that float16 matmuls lose with a float16 accumulator.  
See [CublasOps](https://github.com/aredden/torch-cublas-hgemm) for a cuBLAS implementation and [gpu_poor](https://github.com/sekstini/gpupoor) for a triton implementation of split-k matmul.

## Performance

nai-t5 supports two types of attention: [SDPA](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) and [Flex](https://pytorch.org/blog/flexattention/).

T5 is challenging to run performantly, because its relative position bias requires applying an arbitrary bias in the attention calculation.  
SDPA typically falls back to the slowest backend, cutlassF (memory-efficient attention), when arbitrary biases are required.  
On H100s, SDPA can use the cuDNN backend as a faster alternative. It's unknown how reliable is cuDNN's correctness. We encountered correctness problems with cuDNN SDPA [during training](https://github.com/pytorch/pytorch/issues/139298), but perhaps inference on fixed workloads could be different.  
HF computes attention manually via math operations. 

Flex attention is a far faster way to implement complex attention patterns. We measured [a few approaches](https://github.com/pytorch/pytorch/issues/138493), but "just add an arbitrary bias" was the fastest under the parameters we tried.  
Flex attention is only fast when compiled, because otherwise it falls back to a vmapped math attention.

### Benchmark

We measured the encoder performance of T5 v1.1 XXL at encoding a batch-of-1, 512-length context.

For uncompiled inference, nai-t5 SDPA (cutlassF) is 60% faster than HF, or 68% with cuDNN.

For compiled inference, nai-t5 SDPA (cutlassF) and HF are comparable. cuDNN is about 2% faster.
nai-t5 Flex is 10% faster at 0% sparsity (full 512-length prompt).  
nai-t5 Flex is 16% faster at 93.75% sparsity (1-token prompt).

```
Implementation        Compiled    FLOP/s           ms/iter    iter/s
--------------------  ----------  -------------  ---------  --------
hf                    False       226.6 TFLOP/s       21.4      46.8
nai_sdpa (cutlassF)   False       363.5 TFLOP/s       13.3      75.0
nai_sdpa (cuDNN)      False       381.5 TFLOP/s       12.7      78.7
nai_flex (512 tokens) False        23.3 TFLOP/s      208.0       4.8
nai_flex   (1 token)  False        23.2 TFLOP/s      208.9       4.8
hf                     True       499.4 TFLOP/s        9.7     103.1
nai_sdpa (cutlassF)    True       501.0 TFLOP/s        9.7     103.4
nai_sdpa (cuDNN)       True       513.2 TFLOP/s        9.4     105.9
nai_flex (512 tokens)  True       552.8 TFLOP/s        8.8     114.1
nai_flex   (1 token)   True       579.4 TFLOP/s        8.4     119.6
```

Performance measured via [`benchmark_encoder.py`](nai_t5/scripts/benchmark_encoder.py), using environment:

```
transformers 4.49.0
torch 2.6.0
CUDA 12.8
Nvidia Driver Version: 535.216.01
triton 3.2.0+git35c6c7c6
apex layernorm **not** used by transformers (commented-out of modeling_t5.py to avoid import)
NVIDIA H100 80GB HBM3
```

Typical invocation:

```python
python -m nai_t5.scripts.benchmark_encoder --ckpt v1_1-xxl --batch-size 1 --nai-fuse-norm-scales --bench-hf --bench-nai-sdpa --enable-cudnn-sdpa --bench-nai-flex
```

There are initiatives in HF transformers to introduce [Flex attention](https://github.com/huggingface/transformers/pull/36103) and [SDPA](https://github.com/huggingface/transformers/pull/31167). These are still in review at the time of transformers v4.49.0.

<!--
This uses a lot of space to say very little. Let's keep things snappy.

HF FLOP breakdown:

```
Module                        FLOP    % Total
-----------------------  ---------  ---------
T5EncoderModel           4844.723B    100.00%
 - aten.mm               4741.644B     97.87%
 - aten.bmm               103.079B      2.13%
 T5EncoderModel.encoder  4844.723B    100.00%
  - aten.mm              4741.644B     97.87%
  - aten.bmm              103.079B      2.13%
```

NAI SDPA (cutlassF) FLOP breakdown:

```
Module                                                FLOP    % Total
-----------------------------------------------  ---------  ---------
T5EncoderStack                                   4844.723B    100.00%
 - aten.mm                                       4741.644B     97.87%
 - aten._scaled_dot_product_efficient_attention   103.079B      2.13%
```

NAI SDPA (cuDNN) FLOP breakdown:

```
Module                                            FLOP    % Total
-------------------------------------------  ---------  ---------
T5EncoderStack                               4844.723B    100.00%
 - aten.mm                                   4741.644B     97.87%
 - aten._scaled_dot_product_cudnn_attention   103.079B      2.13%
```
-->

## Precision

On T5v1.1 XXL, we compare HF vs nai-t5 half-precision implementations to see how close each gets to HF float32.

nai-t5 half-precision implementations get closer than HF to the float32 reference in every quantile.

![Encoder precision](docs/encoder-precision.png)

```
absmax diff quantiles:
[0.5000, 0.7500, 0.9000, 0.9500, 0.9900, 0.9990, 0.9999]

HF float32 vs HF pure-bf16:
[0.0006, 0.0011, 0.0018, 0.0022, 0.0033, 0.0047, 0.0080]
HF float32 vs NAI bf16:
[0.0003, 0.0005, 0.0008, 0.0010, 0.0014, 0.0022, 0.0037]

HF float32 vs HF fp16:
[4.7763e-05, 9.2119e-05, 1.4318e-04, 1.7839e-04, 2.4898e-04, 3.5743e-04, 6.5168e-04]
HF float32 vs NAI f16:
[3.7434e-05, 7.2266e-05, 1.1227e-04, 1.3884e-04, 1.9671e-04, 2.7551e-04, 4.2810e-04]
```

Precision compared via [`t5_encoder_hf_precision_parity.py`](nai_t5/scripts/t5_encoder_hf_precision_parity.py), using environment:

```
transformers 4.49.0
torch 2.6.0
CUDA 12.8
Nvidia Driver Version: 535.216.01
triton 3.2.0+git35c6c7c6
apex layernorm **not** used by transformers (commented-out of modeling_t5.py to avoid import)
NVIDIA H100 80GB HBM3
nai-t5 using norm fusion but not using flex attention or torch compilation
```

<!--
if apex is allowed to be imported, which changes HF's layernorm:

HF float32 vs HF bf16:
[0.0006, 0.0012, 0.0018, 0.0023, 0.0033, 0.0051, 0.0094]
HF float32 vs NAI bf16:
[0.0003, 0.0005, 0.0008, 0.0010, 0.0014, 0.0022, 0.0037]

couldn't measure fp16 because apex seems to only work in mixed-precision.
apex layernorm didn't like receiving float32 input with float16 weight.
-->

nai-t5 has the advantage of a float32 residual.  
HF has the advantage of running FFN out-projections in float32, which is costly.  
Runtime performance should be compared too to understand the cost/benefit tradeoff of where extra precision was purchased.

The two implementations use entirely different approaches to float16. HF takes the risk that activation-clipping could impact outliers. nai-t5 takes a risk of float16 underflow in its residual stream. both are more accurate than their bfloat16 counterparts.

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

## References

T5:

```bibtex
@article{10.5555/3455716.3455856,
    author = {Raffel, Colin and Shazeer, Noam and Roberts, Adam and Lee, Katherine and Narang, Sharan and Matena, Michael and Zhou, Yanqi and Li, Wei and Liu, Peter J.},
    title = {Exploring the limits of transfer learning with a unified text-to-text transformer},
    year = {2020},
    issue_date = {January 2020},
    publisher = {JMLR.org},
    volume = {21},
    number = {1},
    issn = {1532-4435},
    abstract = {Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts all text-based language problems into a text-to-text format. Our systematic study compares pretraining objectives, architectures, unlabeled data sets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new "Colossal Clean Crawled Corpus", we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our data set, pre-trained models, and code.},
    journal = {J. Mach. Learn. Res.},
    month = jan,
    articleno = {140},
    numpages = {67},
    keywords = {transfer learning, natural language processing, multi-task learning, attention based models, deep learning}
}
```

UMT5:

```bibtex
@inproceedings{chung2023unimax,
    title={UniMax: Fairer and More Effective Language Sampling for Large-Scale Multilingual Pretraining},
    author={Hyung Won Chung and Xavier Garcia and Adam Roberts and Yi Tay and Orhan Firat and Sharan Narang and Noah Constant},
    booktitle={The Eleventh International Conference on Learning Representations },
    year={2023},
    url={https://openreview.net/forum?id=kXwdL1cWOAi}
}
```

pile-t5:

```bibtex
@misc{2024PileT5,
  author  = {Lintang Sutawika and Aran Komatsuzaki and Colin Raffel},
  title   = {Pile-T5},
  year    = {2024},
  url     = {https://blog.eleuther.ai/pile-t5/},
  note    = {Blog post},
}
```

Flex attention:

```bibtex
@misc{li2024flexattention,
      title={FlexAttention for Efficient High-Resolution Vision-Language Models}, 
      author={Junyan Li and Delin Chen and Tianle Cai and Peihao Chen and Yining Hong and Zhenfang Chen and Yikang Shen and Chuang Gan},
      year={2024},
      eprint={2407.20228},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.20228}, 
}
```

Huggingface Transformers:

```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```

Graphcore, [Running Flan-T5‌-XL Inference In Float16‍ Fo‌r IPU - Ho‌w We D‌id It](https://www.graphcore.ai/posts/running-flan-t5-xl-inference-in-float16-for-ipu-how-we-did-it)

## Contribution

This repository is open-source but not open-contribution. Be prepared that we may not have bandwidth to review pull requests or answer issues; we develop this during spare time. We are interested in knowing about breaks, but improving convenience or compatibility is a lower priority.

## License

Apache 2.0. Uses code from [HF transformers](https://github.com/huggingface/transformers), which is [also Apache 2.0](https://github.com/huggingface/transformers/blob/main/LICENSE).