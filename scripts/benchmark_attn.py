#!/usr/bin/env python

from torch import no_grad
import torch
from torch.utils.flop_counter import FlopCounterMode
from triton.testing import do_bench
from transformers.models.t5.configuration_t5 import T5Config as T5ConfigHF
from transformers.models.t5.modeling_t5 import T5Attention
from transformers.models.t5 import T5TokenizerFast
from nai_t5 import T5Config, to_based_config
from nai_t5.t5_encoder import T5EncoderSelfAttention

def get_flops_achieved(f):
  flop_counter = FlopCounterMode(display=True)
  with flop_counter, no_grad():
    f()
  total_flops = flop_counter.get_total_flops()
  ms_per_iter = do_bench(f)
  iters_per_second = 1e3/ms_per_iter
  print(f"{iters_per_second * total_flops / 1e12} TF/s")

device=torch.device('cuda')
dtype=torch.bfloat16

hf_model_name = "google/t5-v1_1-xl"
hf_config: T5ConfigHF = T5ConfigHF.from_pretrained(hf_model_name)
# just loading the tokenizer to learn the model_max_length, sigh
hf_tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained(hf_model_name, legacy=False)
ctx_len = hf_tokenizer.model_max_length

dtype = torch.bfloat16
my_config: T5Config = to_based_config(hf_config, n_tokens=ctx_len)
dim = my_config.hidden_dim
heads = my_config.n_head
my_config.linear_weight_dtype = torch.bfloat16
my_config.emb_weight_dtype = torch.bfloat16 # not relevant for this benchmark
my_config.norm_weight_dtype = torch.bfloat16 # not relevant for this benchmark

# don't bother with weight init, just construct
with device:
    hf_attn = T5Attention(hf_config, has_relative_attention_bias=False).eval().to(dtype)
    my_attn = T5EncoderSelfAttention(my_config).eval()

batch_size = 5
hidden_states = torch.randn(batch_size, ctx_len, dim, device=device, dtype=dtype)
bias = torch.randn(batch_size, heads, ctx_len, ctx_len, device=device, dtype=dtype)
nai_mask = torch.ones(batch_size, 1, 1, ctx_len, device=device, dtype=torch.bool)
hf_mask = nai_mask.type(dtype)

print('tracing HF...')
get_flops_achieved(lambda: hf_attn(hidden_states, mask=hf_mask, position_bias=bias)[0])
print('tracing NAI...')
get_flops_achieved(lambda: my_attn(hidden_states, bias, mask=nai_mask))

print('tracing compiled HF...')
hf_compiled = torch.compile(hf_attn)
get_flops_achieved(lambda: hf_compiled(hidden_states, mask=hf_mask, position_bias=bias)[0])
print('tracing compiled NAI...')
my_compiled = torch.compile(my_attn)
get_flops_achieved(lambda: my_compiled(hidden_states, bias, mask=nai_mask))