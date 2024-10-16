from typing import Dict, OrderedDict

import torch
from torch import FloatTensor, Tensor, inference_mode
from torch.amp import autocast
from transformers.models.t5 import T5ForConditionalGeneration, T5TokenizerFast
from transformers.models.t5.configuration_t5 import T5Config as T5ConfigHF
from transformers.tokenization_utils_base import BatchEncoding, TruncationStrategy
from transformers.utils.generic import PaddingStrategy, TensorType

from nai_t5 import (
    T5Config,
    hf_to_based_t5_state,
    label_mask_to_decoder_mask,
    labels_to_decoder_input_ids,
    to_based_config,
)
from nai_t5 import T5

from torch.nn import Module
from functools import partial


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
    hf_t5 = T5ForConditionalGeneration.from_pretrained(hf_model_name).eval()

    my_config: T5Config = to_based_config(hf_config, n_tokens=hf_tokenizer.model_max_length)
    my_t5 = T5(my_config).eval()
    input_encoding: BatchEncoding = hf_tokenizer(
        ["Today is a fine <extra_id_0> on which to walk my <extra_id_1> in the park."],
        return_tensors=TensorType.PYTORCH,
        padding=PaddingStrategy.LONGEST,
        truncation=TruncationStrategy.DO_NOT_TRUNCATE,
        add_special_tokens=False,
    ).to(device)
    label_encoding: BatchEncoding = hf_tokenizer(
        ["<pad><extra_id_0> day<extra_id_1> dog"],
        return_tensors=TensorType.PYTORCH,
        padding=PaddingStrategy.LONGEST,
        truncation=TruncationStrategy.DO_NOT_TRUNCATE,
        add_special_tokens=True,
    ).to(device)
    input_ids = input_encoding["input_ids"]
    input_ids_mask = input_encoding["attention_mask"]
    labels = label_encoding["input_ids"]
    labels_mask = label_encoding["attention_mask"]
    decoder_input_ids = labels_to_decoder_input_ids(
        labels,
        pad_token_id=my_config.pad_token_id,
        decoder_start_token_id=my_config.decoder_start_token_id,
        label_ignore_index=my_config.label_ignore_index,
    )
    # TODO: check whether this is correct/necessary
    #       (maybe we can/should rely on causal mask + loss-masking)
    decoder_mask = label_mask_to_decoder_mask(labels_mask)

    hf_state: OrderedDict[str, Tensor] = hf_t5.state_dict()
    converted_enc_state: Dict[str, Tensor] = hf_to_based_t5_state(hf_state, my_config)
    my_t5.load_state_dict(converted_enc_state)

    my_t5.to(device)
    hf_t5.to(device)

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
    for name, mod in hf_t5.named_modules():
        mod.register_forward_hook(partial(hook, name=f'hf/{name}'))
    for name, mod in my_t5.named_modules():
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
    replace_norms(hf_t5)

    def replace_gates(mod: Module) -> None:
        from transformers.activations import NewGELUActivation
        from torch.nn import GELU
        for child_name, child_mod in mod.named_children():
            if isinstance(child_mod, NewGELUActivation):
                gelu = GELU(approximate='tanh')
                setattr(mod, child_name, gelu)
            else:
                replace_gates(child_mod)
    replace_gates(hf_t5)

    seed = 42
    with inference_mode(), autocast(device_type=device.type, dtype=torch.bfloat16):
        # seed the random, so that we can parity-test things like dropout (if enabled)
        torch.manual_seed(seed)
        hf_out = hf_t5.forward(
            input_ids=input_ids,
            attention_mask=input_ids_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_mask,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=False,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )
        torch.manual_seed(seed)
        my_out: FloatTensor = my_t5.forward(
            encoder_input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            encoder_input_mask=input_ids_mask.bool(),
            decoder_input_mask=decoder_mask.bool(),
        )
        diff = hf_out.logits-my_out
        stat(diff)
        print('absmax:', f'{diff.abs().max().item()}:g')
        # assert hf_out.logits.allclose(my_out, atol=0.5), "HF and NAI logits do not match"
        assert hf_out.logits.allclose(my_out), "HF and NAI logits do not match"
    pass  # somewhere to put your breakpoint


if __name__ == "__main__":
    main()
