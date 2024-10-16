from typing import Dict, OrderedDict

import torch
from torch import FloatTensor, Tensor, inference_mode
from torch.amp import autocast
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.umt5 import UMT5EncoderModel as UMT5EncoderHF
from transformers.models.umt5.configuration_umt5 import UMT5Config as UMT5ConfigHF
from transformers import LlamaTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding
from enum import Enum
from contextlib import nullcontext

from nai_t5 import T5Config, T5EncoderStack
from nai_t5.t5_hf import (
    to_based_config,
    hf_to_based_t5_enc_state,
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


class PrecisionMode(str, Enum):
    Float32 = 'float32'
    MixedBF16 = 'mixed-bf16'
    PureBF16 = 'pure-bf16'


def main():
    device = torch.device("cuda")

    hf_model_name = "EleutherAI/pile-t5-base"
    hf_config: UMT5ConfigHF = UMT5ConfigHF.from_pretrained(hf_model_name)
    hf_tokenizer: LlamaTokenizerFast = LlamaTokenizerFast.from_pretrained(hf_model_name)
    hf_tokenizer.pad_token = hf_tokenizer.unk_token

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
    
    hf_encoder: UMT5EncoderHF = UMT5EncoderHF.from_pretrained(hf_model_name, **hf_dtype_kwargs).eval()

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

    # NOTE: we are CHANGING the HF implementation!!
    # use torch built-in RMSNorm and GELU.
    # these are subtly but acceptably different; eliminate them as a confounding factor
    # so that we can measure parity of *everything else*
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
    with (
        inference_mode(),
        autocast_ctx,
        sdpa_kernel(SDPBackend.MATH),
    ):
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
    compare_dtype = torch.promote_types(my_encoder_out.dtype, hf_enc_out.last_hidden_state.dtype)
    hf_cast: FloatTensor = hf_enc_out.last_hidden_state.type(compare_dtype)
    my_cast: FloatTensor = my_encoder_out.type(compare_dtype)
    diff = hf_cast.float().sub(my_cast.float())
    qs = torch.tensor([.5, .75, .9, .95, .99, .999, .9999], device=device)
    q = diff.abs().quantile(qs)
    print(f'quantiles ({qs.cpu()}):\n           {q.cpu()}')
    stat(diff, 'diff')
    assert hf_cast.allclose(my_cast), "HF and NAI outputs do not match"
    pass  # somewhere to put your breakpoint


if __name__ == "__main__":
    main()
