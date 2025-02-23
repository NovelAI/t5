from typing import Dict, OrderedDict

import torch
from torch import FloatTensor, Tensor, inference_mode
from torch.nn import Module
from torch.amp import autocast
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5 import T5EncoderModel as HFT5EncoderModel
from transformers.models.t5 import T5TokenizerFast
from transformers.models.t5.configuration_t5 import T5Config as T5ConfigHF
from transformers.tokenization_utils_base import BatchEncoding
from enum import Enum
from contextlib import nullcontext

from nai_t5 import T5Config, T5EncoderStack
from nai_t5.t5_hf import (
    hf_to_based_t5_enc_state,
    to_based_config,
    replace_gates,
    replace_norms,
)

from torch import Tensor
from typing import Optional
from torch.linalg import matrix_norm
def fmt_matrix_norm(t: Tensor) -> str:
    t = t.squeeze().cpu()
    if t.numel() == 1:
        return f'{t.item():.2f}'
    return str(t)
def stat(t: Tensor, label: Optional[str] = None) -> None:
    print(tuple(t.shape), str(t.dtype).removeprefix('torch.'), f'σ={t.std().item():g}', f'μ={t.mean().item():.2f}', f'norm={fmt_matrix_norm(matrix_norm(t.float(), ord=2))}', f'absmax={t.abs().max().item():g}', label or '')


def explain_diff(ref: FloatTensor, candidate: FloatTensor) -> FloatTensor:
    diff = ref.float().sub(candidate.float())
    qs = torch.tensor([.5, .75, .9, .95, .99, .999, .9999], device=ref.device)
    q = diff.abs().quantile(qs)
    print(str(q.cpu()).removeprefix("tensor(").removesuffix(")"))
    stat(diff, 'diff')


class PrecisionMode(str, Enum):
    Float32 = 'float32'
    MixedBF16 = 'mixed-bf16'
    PureBF16 = 'pure-bf16'


def main():
    device = torch.device("cuda")

    # hf_model_name = "google/t5-v1_1-small"
    hf_model_name = "google/t5-v1_1-xl"

    hf_config: T5ConfigHF = T5ConfigHF.from_pretrained(hf_model_name)
    hf_tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained(hf_model_name, legacy=False)

    # precision_mode = PrecisionMode.Float32
    # precision_mode = PrecisionMode.MixedBF16
    precision_mode = PrecisionMode.PureBF16
    match precision_mode:
        case PrecisionMode.Float32 | PrecisionMode.MixedBF16:
            hf_dtype_kwargs = {}
        case PrecisionMode.PureBF16:
            hf_dtype_kwargs = {'torch_dtype': torch.bfloat16}
        case _:
            raise ValueError(f"Invalid precision mode: {precision_mode}")
    
    hf_encoder = HFT5EncoderModel.from_pretrained(hf_model_name, **hf_dtype_kwargs).eval()

    my_config: T5Config = to_based_config(hf_config, n_tokens=hf_tokenizer.model_max_length)
    if precision_mode == PrecisionMode.PureBF16:
        my_config.linear_weight_dtype = torch.bfloat16
        my_config.emb_weight_dtype = torch.bfloat16
        my_config.norm_weight_dtype = torch.bfloat16
    my_encoder = T5EncoderStack(my_config).eval()

    tokens: BatchEncoding = hf_tokenizer("hello world", return_tensors="pt")
    tokens.to(device)

    hf_state: OrderedDict[str, Tensor] = hf_encoder.state_dict()
    converted_enc_state: Dict[str, Tensor] = hf_to_based_t5_enc_state(hf_state, my_config)
    my_encoder.load_state_dict(converted_enc_state)

    # make HF's norms and gates match ours, so that we're only comparing "everything else" (we already know our norms and gates are subtly different)
    replace_norms(hf_encoder)
    replace_gates(hf_encoder)

    my_encoder.to(device)
    hf_encoder.to(device)

    match precision_mode:
        case PrecisionMode.Float32 | PrecisionMode.PureBF16:
            autocast_ctx = nullcontext()
        case PrecisionMode.MixedBF16:
            autocast_ctx = autocast(device_type=device.type, dtype=torch.bfloat16)
        case _:
            raise ValueError(f"Invalid precision mode: {precision_mode}")

    seed = 42
    with inference_mode(), autocast_ctx:
        # seed the random, so that we can parity-test things like dropout (if enabled)
        torch.manual_seed(seed)
        hf_enc_out: BaseModelOutputWithPastAndCrossAttentions = hf_encoder(
            input_ids=tokens.input_ids,  # [1, 3]
            attention_mask=tokens.attention_mask,  # [1, 3]
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )
        torch.manual_seed(seed)
        my_encoder_out: FloatTensor = my_encoder(
            input_ids=tokens.input_ids,
            input_mask=tokens.attention_mask.bool(),
        )
    compare_dtype = torch.promote_types(my_encoder_out.dtype, hf_enc_out["last_hidden_state"].dtype)
    hf_cast: FloatTensor = hf_enc_out["last_hidden_state"].type(compare_dtype)
    my_cast: FloatTensor = my_encoder_out.type(compare_dtype)
    qs = torch.tensor([.5, .75, .9, .95, .99, .999, .9999], device=device)
    print("quantiles:")
    print(str(qs.cpu()).removeprefix("tensor(").removesuffix(")"))
    explain_diff(hf_cast, my_cast)
    assert hf_cast.allclose(my_cast), "HF and NAI outputs do not match"
    pass  # somewhere to put your breakpoint


if __name__ == "__main__":
    main()
