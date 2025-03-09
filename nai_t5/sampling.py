import torch
from torch import LongTensor, FloatTensor, inference_mode
from typing import Protocol, Set, Generator


class BoundDecode(Protocol):
    @staticmethod
    def __call__(in_tokens: LongTensor) -> FloatTensor: ...


class BoundCachedDecode(Protocol):
    @staticmethod
    def __call__(in_tokens: LongTensor, self_past_kv: FloatTensor) -> FloatTensor: ...


class MakeLogitGenerator(Protocol):
    @staticmethod
    def __call__(decoder_start_prompt: LongTensor) -> Generator[FloatTensor, LongTensor, None]: ...


@inference_mode()
def generate_greedy(
    decoder_start_prompt: LongTensor,
    decode: BoundDecode,
) -> Generator[FloatTensor, LongTensor, None]:
    prompt: LongTensor = decoder_start_prompt
    del decoder_start_prompt
    while True:
        logits: FloatTensor = decode(prompt)
        prompt = yield logits


@inference_mode()
def generate_greedy_cached(
    decoder_start_prompt: LongTensor,
    self_past_kv: FloatTensor,
    decode: BoundCachedDecode,
) -> Generator[FloatTensor, LongTensor, None]:
    prompt: LongTensor = decoder_start_prompt
    del decoder_start_prompt
    while True:
        logits = decode(prompt[:, -1:], self_past_kv=self_past_kv[:, :, :, :, : prompt.size(-1), :])
        prompt = yield logits


def generate_until(
    make_gen: MakeLogitGenerator,
    device: torch.device,
    batch_size=1,
    max_tokens=49,
    raise_on_overflow=True,
    stop_tokens: Set[int] = set(),
    decoder_start_token_id=0,
    pad_token_id=0,
) -> Generator[LongTensor, None, LongTensor]:
    buffer: LongTensor = torch.full(
        (batch_size, max_tokens + 1),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )
    if decoder_start_token_id != pad_token_id:
        buffer[:, 0] = decoder_start_token_id

    proceed = True
    out_ix = 1
    gen: Generator[FloatTensor, LongTensor, None] = make_gen(buffer[:, 0:1])
    logits: FloatTensor = next(gen)
    for _ in range(max_tokens):
        logits = logits[:, -1, :]
        prediction: LongTensor = logits.argmax(dim=-1, keepdim=True)
        prediction_cpu: LongTensor = prediction.cpu().squeeze()
        for stop_token in stop_tokens:
            if torch.any(prediction_cpu == stop_token):
                proceed = False
                break
        buffer[:, out_ix] = prediction
        out_ix += 1
        yield prediction
        if not proceed:
            break
        logits: FloatTensor = gen.send(buffer[:, :out_ix])
    else:
        if raise_on_overflow:
            raise OverflowError(f"exceeded {max_tokens} token limit")
    return buffer[:, :out_ix]
