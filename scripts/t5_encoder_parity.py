from typing import Dict, OrderedDict

import torch
from torch.nn import Module
from torch import FloatTensor, Tensor, inference_mode
from torch.amp import autocast
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5 import T5EncoderModel as HFT5EncoderModel
from transformers.models.t5 import T5TokenizerFast
from transformers.models.t5.configuration_t5 import T5Config as T5ConfigHF
from transformers.tokenization_utils_base import BatchEncoding
from transformers.activations import NewGELUActivation
from functools import partial

from nai_t5 import T5Config, T5EncoderStack, hf_to_based_t5_enc_state, to_based_config


from torch import Tensor
from typing import Optional
from torch.linalg import matrix_norm
def stat(t: Tensor, label: Optional[str] = None) -> None:
    print(tuple(t.shape), str(t.dtype).removeprefix('torch.'), f'σ={t.std().item():g}', f'μ={t.mean().item():g}', f'norm={matrix_norm(t.float(), ord=2).squeeze().cpu()}', label or '')


def main():
    device = torch.device("cuda")
    hf_model_name = "google/t5-v1_1-small"
    hf_config: T5ConfigHF = T5ConfigHF.from_pretrained(hf_model_name)
    hf_tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained(hf_model_name, legacy=False)
    hf_encoder = HFT5EncoderModel.from_pretrained(hf_model_name).eval()

    my_config: T5Config = to_based_config(hf_config, n_tokens=hf_tokenizer.model_max_length)
    my_encoder = T5EncoderStack(my_config).eval()

    def scrutinize_input(module: torch.nn.Module, input, name: str):
        if isinstance(input, (tuple, list)):
            for elem in input:
                scrutinize_input(module, elem, name)
        elif isinstance(input, dict):
            for elem in input.values():
                scrutinize_input(module, elem, name)
        elif torch.is_tensor(input):
            assert input.isfinite().all().item(), f"{module.__class__.__name__} {name}: non-finite input encountered"
    def scrutinize_output(module: torch.nn.Module, output, name: str):
        if isinstance(output, (tuple, list)):
            for elem in output:
                scrutinize_output(module, elem, name)
        elif isinstance(output, dict):
            for elem in output.values():
                scrutinize_output(module, elem, name)
        elif torch.is_tensor(output):
            assert output.isfinite().all().item(), f"{module.__class__.__name__} {name}: non-finite output encountered"

    def hook(module: Module, input, output, name: str):
        scrutinize_input(module, input, name)
        scrutinize_output(module, output, name)
    for name, mod in hf_encoder.named_modules():
        mod.register_forward_hook(partial(hook, name=f'hf/{name}'))
    for name, mod in my_encoder.named_modules():
        mod.register_forward_hook(partial(hook, name=f'my/{name}'))

    def replace_norms(mod: Module) -> None:
        from transformers.models.t5.modeling_t5 import T5LayerNorm
        from nai_t5.t5_common import RMSNorm_f32
        for child_name, child_mod in mod.named_children():
            # print(child_name, child_mod.__class__.__name__)
            if isinstance(child_mod, T5LayerNorm):
                # if mod.__class__.__module__.startswith('apex')
                norm = RMSNorm_f32(
                    child_mod.normalized_shape,
                    eps=child_mod.eps,
                    elementwise_affine=child_mod.elementwise_affine,
                    device=child_mod.weight.device,
                )
                with inference_mode():
                    norm.weight.copy_(child_mod.weight)
                setattr(mod, child_name, norm)
            else:
                replace_norms(child_mod)
    replace_norms(hf_encoder)

    def replace_gates(mod: Module) -> None:
        from transformers.activations import NewGELUActivation
        from torch.nn import GELU
        for child_name, child_mod in mod.named_children():
            # print(child_name, child_mod.__class__.__name__)
            if isinstance(child_mod, NewGELUActivation):
                gelu = GELU(approximate='tanh')
                setattr(mod, child_name, gelu)
            else:
                replace_gates(child_mod)
    replace_gates(hf_encoder)

    tokens: BatchEncoding = hf_tokenizer("hello world", return_tensors="pt")
    tokens.to(device)

    hf_state: OrderedDict[str, Tensor] = hf_encoder.state_dict()
    converted_enc_state: Dict[str, Tensor] = hf_to_based_t5_enc_state(hf_state, my_config)
    my_encoder.load_state_dict(converted_enc_state)

    my_encoder.to(device)
    hf_encoder.to(device)

    seed = 42
    with inference_mode(), autocast(device_type=device.type, dtype=torch.bfloat16):
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
        diff = hf_enc_out["last_hidden_state"].type_as(my_encoder_out).sub(my_encoder_out)
        stats: list[str] = [f'{label}: {t.item():g}' for t, label in zip(torch.std_mean(diff), ('std', 'mean'))]
        print('diff', ', '.join(stats))
        print(f'diff absmax: {diff.abs().max().item():g}')
        assert (
            hf_enc_out["last_hidden_state"].type_as(my_encoder_out).allclose(my_encoder_out)#, atol=0.5)
            # diff.allclose(my_encoder_out, atol=0.5)
        ), "HF and NAI outputs do not match"
    pass  # somewhere to put your breakpoint


if __name__ == "__main__":
    main()
