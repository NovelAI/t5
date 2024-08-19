from typing import Dict, OrderedDict

import torch
from torch import FloatTensor, Tensor, inference_mode
from torch.amp import autocast
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5 import T5EncoderModel as HFT5EncoderModel
from transformers.models.t5 import T5TokenizerFast
from transformers.models.t5.configuration_t5 import T5Config as T5ConfigHF
from transformers.tokenization_utils_base import BatchEncoding

from nai_t5 import T5Config, T5EncoderStack, hf_to_based_t5_enc_state, to_based_config


def main():
    device = torch.device("cuda")
    hf_model_name = "google/t5-v1_1-small"
    hf_config: T5ConfigHF = T5ConfigHF.from_pretrained(hf_model_name)
    hf_tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained(hf_model_name, legacy=False)
    hf_encoder = HFT5EncoderModel.from_pretrained(hf_model_name).eval()

    my_config: T5Config = to_based_config(hf_config, n_tokens=hf_tokenizer.model_max_length)
    my_encoder = T5EncoderStack(my_config).eval()

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
        hf_enc_out: BaseModelOutputWithPastAndCrossAttentions = hf_encoder.forward(
            input_ids=tokens.input_ids,  # [1, 3]
            attention_mask=tokens.attention_mask,  # [1, 3]
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )
        torch.manual_seed(seed)
        my_encoder_out: FloatTensor = my_encoder.forward(
            input_ids=tokens.input_ids,
            input_mask=tokens.attention_mask.bool(),
        )
        assert (
            hf_enc_out["last_hidden_state"].type_as(my_encoder_out).allclose(my_encoder_out, atol=0.5)
        ), "HF and NAI outputs do not match"
    pass  # somewhere to put your breakpoint


if __name__ == "__main__":
    main()
