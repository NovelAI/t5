from dataclasses import dataclass, field
from einops import rearrange
from functools import partial
from typing import Dict, Generator, List, OrderedDict, Set

import torch
from torch import BoolTensor, FloatTensor, LongTensor, Tensor, inference_mode
from torch.amp import autocast
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList, TensorType
from transformers.generation.streamers import BaseStreamer
from transformers.models.t5 import T5ForConditionalGeneration, T5TokenizerFast
from transformers.models.t5.configuration_t5 import T5Config as T5ConfigHF
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils.generic import TensorType

from nai_t5 import T5, T5Config, hf_to_based_t5_state, to_based_config
from nai_t5.sampling import BoundCachedDecode, MakeLogitGenerator, generate_greedy_cached, generate_until

from torch.nn import Module
from functools import partial


from torch import Tensor
from typing import Optional
from torch.linalg import matrix_norm
def stat(t: Tensor, label: Optional[str] = None) -> None:
    print(tuple(t.shape), str(t.dtype).removeprefix('torch.'), f'σ={t.std().item():g}', f'μ={t.mean().item():g}', f'norm={matrix_norm(t.float(), ord=2).squeeze().cpu()}', label or '')


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
    hf_config: T5ConfigHF = T5ConfigHF.from_pretrained(hf_model_name)
    hf_tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained(hf_model_name, legacy=False)
    hf_t5 = T5ForConditionalGeneration.from_pretrained(hf_model_name).eval()

    my_config: T5Config = to_based_config(hf_config, n_tokens=hf_tokenizer.model_max_length)
    my_t5 = T5(my_config).eval()

    input_encoding: BatchEncoding = hf_tokenizer(
        "Today is a fine <extra_id_0> on which to walk my <extra_id_1> in the park.",
        return_tensors=TensorType.PYTORCH,
        # for some reason, ending the input with </s> results in a saner answer on t5 small.
        # t5 large is fine without </s>; we should generally prefer False, but it's nice to debug on small
        add_special_tokens=True,
    ).to(device)
    label_encoding: BatchEncoding = hf_tokenizer(
        "<pad><extra_id_0> day<extra_id_1> dog<extra_id_2>",
        return_tensors=TensorType.PYTORCH,
        add_special_tokens=False,
    ).to(device)
    input_ids = input_encoding["input_ids"]
    input_ids_mask = input_encoding["attention_mask"]
    expected_output = label_encoding["input_ids"]
    mask2: int = hf_tokenizer.convert_tokens_to_ids("<extra_id_2>")

    hf_state: OrderedDict[str, Tensor] = hf_t5.state_dict()
    converted_enc_state: Dict[str, Tensor] = hf_to_based_t5_state(hf_state, my_config)
    my_t5.load_state_dict(converted_enc_state)
    my_t5.to(device)

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

    max_new_tokens = 16
    seed = 42

    run_hf_baseline = True
    if run_hf_baseline:
        hf_t5.to(device)

        streamer_factory = lambda: TokenStreamer(tokenizer=hf_tokenizer, print_tokens=True)
        streamer: TokenStreamer = streamer_factory()
        stopping_criteria = StoppingCriteriaList([StopOnToken(set((mask2, hf_t5.config.eos_token_id)))])

        with inference_mode(), autocast(device_type=device.type, dtype=torch.bfloat16):
            # seed the random, so that we can parity-test things like dropout (if enabled)
            torch.manual_seed(seed)
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
        assert generate_out.squeeze().equal(
            expected_output.squeeze()
        ), "baseline HF generation no longer outputs the fixture we originally documented"
        # gen_decoded_s: str = hf_tokenizer.decode(generate_out.squeeze())
        gen_decoded: List[str] = hf_tokenizer.convert_ids_to_tokens(generate_out.squeeze())
    else:
        generate_out: LongTensor = expected_output

    ####
    #### and now we try to do the same greedy search as generate, manually:
    ####
    torch.manual_seed(seed)

    with inference_mode(), autocast(device_type=device.type, dtype=torch.bfloat16):
        encoding: FloatTensor = my_t5.encoder(input_ids, input_mask=input_ids_mask.bool())
        cross_kv: FloatTensor = my_t5.decoder.get_cross_kv(encoding)

    batch_size = 1
    cross_mask = rearrange(input_ids_mask.bool(), "b k -> b 1 k")
    cross_mask = cross_mask.expand(-1, max_new_tokens, -1)

    # input_mask = encoding.new_ones((batch_size, max_new_tokens), dtype=torch.bool)

    def decode(
        in_tokens: LongTensor,
        encoding: FloatTensor,
        cross_mask: BoolTensor,
        self_past_kv: FloatTensor,
        cross_kv: FloatTensor,
    ) -> FloatTensor:
        with inference_mode(), autocast(device_type=device.type, dtype=torch.bfloat16):
            decoder_input_embeds: FloatTensor = my_t5.encoder.vocab_embed(in_tokens)
            # input_mask_: BoolTensor = input_mask[:, : in_tokens.size(-1)]
            cross_mask_: BoolTensor = cross_mask[:, : in_tokens.size(-1), :]
            decoder_out: FloatTensor = my_t5.decoder(
                decoder_input_embeds,
                encoding,
                # input_mask=input_mask_,
                input_mask=None,
                cross_mask=cross_mask_,
                self_past_kv=self_past_kv,
                cross_kv=cross_kv,
            )
            logits: FloatTensor = my_t5.lm_head(decoder_out)
            return logits

    self_past_kv: FloatTensor = my_t5.decoder.get_kv_cache(
        length=max_new_tokens, device=encoding.device, dtype=encoding.dtype
    )

    bound_decode: BoundCachedDecode = partial(decode, encoding=encoding, cross_mask=cross_mask, cross_kv=cross_kv)
    make_gen: MakeLogitGenerator = partial(generate_greedy_cached, decode=bound_decode, self_past_kv=self_past_kv)

    gen: Generator[LongTensor, None, LongTensor] = generate_until(
        make_gen=make_gen,
        device=device,
        batch_size=batch_size,
        max_tokens=max_new_tokens,
        stop_tokens={mask2, hf_t5.config.eos_token_id},
        decoder_start_token_id=hf_t5.config.decoder_start_token_id,
        pad_start_token_id=hf_t5.config.pad_token_id,
    )
    my_acc: List[int] = []
    for ix, (tok_t, expected_tok) in enumerate(zip(gen, expected_output[0, 1:].tolist())):
        token_id: int = tok_t.cpu().squeeze().item()
        token_str: int = hf_tokenizer.convert_ids_to_tokens(token_id)
        print(token_str, flush=True, end="")
        my_acc.append(token_id)
        assert (
            token_id == expected_tok
        ), f"based output diverged from HF output at prediction index {ix}. Expected token {expected_tok} ({hf_tokenizer.convert_ids_to_tokens(expected_tok)}). actual: {token_id} ({token_str})"
    else:
        print("", flush=True)
        assert my_acc == generate_out[0, 1:].tolist(), f"based output diverged from HF output"
    pass  # somewhere to put your breakpoint


if __name__ == "__main__":
    main()
