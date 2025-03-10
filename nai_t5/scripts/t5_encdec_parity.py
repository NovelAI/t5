#!/usr/bin/env python

from typing import Dict, OrderedDict

import torch
from torch import FloatTensor, Tensor, inference_mode
from torch.amp import autocast
from transformers.models.t5 import T5ForConditionalGeneration, T5TokenizerFast
from transformers.models.t5.configuration_t5 import T5Config as T5ConfigHF
from transformers.tokenization_utils_base import BatchEncoding, TruncationStrategy
from transformers.utils.generic import PaddingStrategy, TensorType
from torch.nn.attention import SDPBackend, sdpa_kernel

from nai_t5 import (
    T5,
    T5Config,
    label_mask_to_decoder_mask,
    labels_to_decoder_input_ids,
)
from nai_t5.t5_hf import (
    to_based_config,
    hf_to_based_t5_state,
    replace_gates,
    replace_norms,
)


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
        add_special_tokens=True,
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

    # NOTE: we are CHANGING the HF implementation!!
    # use torch built-in RMSNorm and GELU.
    # these are subtlely but acceptably different; eliminate them as a confounding factor
    # so that we can measure parity of *everything else*
    replace_gates(hf_t5)
    replace_norms(hf_t5)

    my_t5.to(device)
    hf_t5.to(device)

    seed = 42
    with (
        inference_mode(),
        autocast(device_type=device.type, dtype=torch.bfloat16),
        sdpa_kernel(SDPBackend.MATH),
    ):
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
        assert hf_out.logits.allclose(my_out), "HF and NAI logits do not match"
    pass  # somewhere to put your breakpoint


if __name__ == "__main__":
    main()
