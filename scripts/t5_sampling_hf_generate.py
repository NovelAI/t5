#!/usr/bin/env python

from dataclasses import dataclass, field
from typing import List, Set

import torch
from torch import LongTensor, inference_mode
from torch.amp import autocast
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList, TensorType
from transformers.generation.streamers import BaseStreamer
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.t5 import T5ForConditionalGeneration, T5TokenizerFast
from transformers.tokenization_utils_base import BatchEncoding


@dataclass
class StopOnToken(StoppingCriteria):
    stop_tokens: Set[int]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids.squeeze()[-1].item() in self.stop_tokens


@dataclass
class TokenStreamer(BaseStreamer):
    tokenizer: T5TokenizerFast
    print_tokens: bool = False
    ids: List[int] = field(default_factory=list, init=False)
    tokens: List[str] = field(default_factory=list, init=False)

    def put(self, value: LongTensor) -> None:
        token_id: int = value.squeeze().item()
        token_str: int = self.tokenizer.convert_ids_to_tokens(token_id)
        if self.print_tokens:
            print(token_str, flush=True, end="")
        self.ids.append(token_id)
        self.tokens.append(token_str)

    def end(self):
        """Function that is called by `.generate()` to signal the end of generation"""
        if self.print_tokens:
            print("", flush=True)


def main():
    device = torch.device("cuda")
    hf_model_name = "google/t5-v1_1-small"
    hf_tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained(hf_model_name, legacy=False)
    hf_t5 = T5ForConditionalGeneration.from_pretrained(hf_model_name).eval().to(device)

    tokens: BatchEncoding = hf_tokenizer(
        "Today is a fine <extra_id_0> on which to walk my <extra_id_1> in the park.",
        return_tensors=TensorType.PYTORCH,
        # for some reason, ending the input with </s> results in a saner answer on t5 small.
        # t5 large is fine without </s>; we should generally prefer False, but it's nice to debug on small
        add_special_tokens=True,
    )
    tokens.to(device)
    input_ids = tokens["input_ids"]
    input_ids_mask = tokens["attention_mask"]
    mask2: int = hf_tokenizer.convert_tokens_to_ids("<extra_id_2>")

    streamer_factory = lambda: TokenStreamer(tokenizer=hf_tokenizer, print_tokens=True)
    streamer: TokenStreamer = streamer_factory()
    max_new_tokens = 16
    stopping_criteria = StoppingCriteriaList([StopOnToken(set((mask2, hf_t5.config.eos_token_id)))])

    seed = 42
    with inference_mode(), autocast(device_type=device.type, dtype=torch.bfloat16):
        # seed the random, so that we can parity-test things like dropout (if enabled)
        torch.manual_seed(seed)
        # Rest of your code...
        generate_out: LongTensor = hf_t5.generate(
            generation_config=GenerationConfig(
                max_new_tokens=max_new_tokens,
                decoder_start_token_id=hf_t5.config.decoder_start_token_id,
                eos_token_id=hf_t5.config.eos_token_id,
                pad_token_id=hf_t5.config.pad_token_id,
            ),
            input_ids=input_ids,
            attention_mask=input_ids_mask,
            streamer=streamer,
            stopping_criteria=stopping_criteria,
        )
    # gen_decoded_s: str = hf_tokenizer.decode(generate_out.squeeze())
    gen_decoded: List[str] = hf_tokenizer.convert_ids_to_tokens(generate_out.squeeze())

    ####
    #### and now we try to do the same greedy search as generate, manually:
    ####
    streamer: TokenStreamer = streamer_factory()
    model_kwargs = {
        "input_ids": input_ids,
        "attention_mask": input_ids_mask,
    }
    inputs_tensor, model_input_name, model_kwargs = hf_t5._prepare_model_inputs(
        inputs=None,
        bos_token_id=None,
        model_kwargs=model_kwargs,
    )
    model_kwargs["output_attentions"] = False
    model_kwargs["output_hidden_states"] = False
    model_kwargs["use_cache"] = True

    # encoder_outputs is not in model_kwargs, so we do this:
    with inference_mode(), autocast(device_type=device.type, dtype=torch.bfloat16):
        model_kwargs = hf_t5._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )

    input_ids, model_kwargs = hf_t5._prepare_decoder_input_ids_for_generation(
        batch_size=1,
        model_input_name=model_input_name,
        model_kwargs=model_kwargs,
        decoder_start_token_id=hf_t5.config.decoder_start_token_id,
        bos_token_id=None,
        device=inputs_tensor.device,
    )

    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    acc_tok: LongTensor = input_ids.clone()

    torch.manual_seed(seed)
    streamer.put(input_ids)
    for _ in range(max_new_tokens):
        model_inputs = hf_t5.prepare_inputs_for_generation(input_ids, **model_kwargs)

        with inference_mode(), autocast(device_type=device.type, dtype=torch.bfloat16):
            loop_outputs: Seq2SeqLMOutput = hf_t5.forward(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
        logits = loop_outputs.logits
        # outputs also includes past_key_values, encoder_last_hidden_state

        next_token_logits = logits[:, -1, :]

        next_tokens = torch.argmax(next_token_logits, dim=-1)
        # finished sentences should have their next token be a padding token
        next_tokens = next_tokens * unfinished_sequences + hf_t5.config.pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        acc_tok = torch.cat([acc_tok, next_tokens.unsqueeze(-1)], dim=-1)
        model_kwargs = hf_t5._update_model_kwargs_for_generation(loop_outputs, model_kwargs, is_encoder_decoder=True)

        should_end: bool = stopping_criteria(input_ids, None)
        streamer.put(next_tokens)
        if should_end:
            streamer.end()
            break
    else:
        streamer.end()

    # loop_decoded2: str = hf_tokenizer.decode(acc_tok.squeeze())
    loop_decoded2: List[str] = hf_tokenizer.convert_ids_to_tokens(acc_tok.squeeze())
    assert loop_decoded2 == gen_decoded, "Greedy loop and generate() output do not match"
    pass  # somewhere to put your breakpoint


if __name__ == "__main__":
    main()
