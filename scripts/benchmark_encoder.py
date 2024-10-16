from torch import no_grad
import torch
from torch.utils.flop_counter import FlopCounterMode
from triton.testing import do_bench
from transformers.models.t5.configuration_t5 import T5Config as T5ConfigHF
from transformers.models.t5 import T5EncoderModel as HFT5EncoderModel
from transformers.models.t5 import T5TokenizerFast
from nai_t5 import T5Config, to_based_config
from nai_t5.t5_encoder import T5EncoderStack

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
my_config.linear_weight_dtype = torch.bfloat16
my_config.emb_weight_dtype = torch.bfloat16
my_config.norm_weight_dtype = torch.bfloat16

# don't bother with weight init, just construct
with device:
    hf_enc = HFT5EncoderModel(hf_config).eval().to(dtype)
    my_enc = T5EncoderStack(my_config).eval()

batch_size = 5
input_ids = torch.arange(ctx_len, device=device, dtype=torch.long).unsqueeze(0).repeat_interleave(batch_size, dim=0)
nai_mask = torch.ones(batch_size, ctx_len, device=device, dtype=torch.bool)
hf_mask = nai_mask.unsqueeze(-1).type(dtype)

print('tracing HF...')
get_flops_achieved(lambda: hf_enc(input_ids, attention_mask=hf_mask).last_hidden_state)
print('tracing NAI...')
get_flops_achieved(lambda: my_enc(input_ids, input_mask=nai_mask))

print('tracing compiled HF...')
hf_compiled = torch.compile(hf_enc)
get_flops_achieved(lambda: hf_compiled(input_ids, attention_mask=hf_mask).last_hidden_state)
print('tracing compiled NAI...')
my_compiled = torch.compile(my_enc)
get_flops_achieved(lambda: my_compiled(input_ids, input_mask=nai_mask))
pass