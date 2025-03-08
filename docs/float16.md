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

Originally our encoder layer looked conventional:

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