from typing import Optional, Callable, TYPE_CHECKING, Any
from functools import partial
from argparse import ArgumentParser, BooleanOptionalAction
from enum import Enum
from dataclasses import dataclass
from torch import FloatTensor, LongTensor, BoolTensor, no_grad, inference_mode
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch
from torch.utils.flop_counter import FlopCounterMode
from triton.testing import do_bench
from transformers.models.t5.configuration_t5 import T5Config as T5ConfigHF
from transformers.models.t5 import T5EncoderModel as HFT5EncoderModel
from nai_t5 import T5Config, to_based_config
from nai_t5.t5_encoder import T5EncoderStack
from nai_t5.t5_common import T5AttnImpl
from contextlib import nullcontext

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask
else:
    BlockMask = Any

def get_flops_achieved(f):
  flop_counter = FlopCounterMode(display=True)
  with flop_counter, no_grad():
    f()
  total_flops = flop_counter.get_total_flops()
  ms_per_iter = do_bench(f)
  iters_per_second = 1e3/ms_per_iter
  print(f"{iters_per_second * total_flops / 1e12} TF/s")

NiladicModelFwd = Callable[[], FloatTensor]

class BenchSubject(str, Enum):
    HF = 'hf'
    NAI_SDPA = 'nai_sdpa'
    NAI_Flex = 'nai_flex'

class Checkpoint(str, Enum):
    V1_1Small = 'v1_1-small'
    V1_1Base = 'v1_1-base'
    V1_1Large = 'v1_1-large'
    V1_1XL = 'v1_1-xl'
    V1_1XXL = 'v1_1-xxl'

ckpt_to_hf_model_name: dict[Checkpoint, str] = {
    Checkpoint.V1_1Small: 'google/t5-v1_1-small',
    Checkpoint.V1_1Base: 'google/t5-v1_1-base',
    Checkpoint.V1_1Large: 'google/t5-v1_1-large',
    Checkpoint.V1_1XL: 'google/t5-v1_1-xl',
    Checkpoint.V1_1XXL: 'google/t5-v1_1-xxl',
}

@dataclass
class Args:
    ckpt: Checkpoint
    batch_size: int
    ctx_len: int
    visible_tokens: int
    flex_block_m: Optional[int]
    flex_block_n: Optional[int]
    seed: int
    flex_mask_pad_queries: bool
    nai_fuse_norm_scales: bool
    disable_mask: bool
    disable_block_mask: bool
    enable_cudnn_sdpa: bool
    bench_hf: bool
    bench_nai_sdpa: bool
    bench_nai_flex: bool

def main(args: Args):
    device=torch.device('cuda')
    dtype=torch.bfloat16

    hf_model_name: str = ckpt_to_hf_model_name[args.ckpt]
    hf_config: T5ConfigHF = T5ConfigHF.from_pretrained(hf_model_name)

    dtype = torch.bfloat16
    nai_config_sdpa: T5Config = to_based_config(hf_config, n_tokens=args.ctx_len)
    nai_config_sdpa.elementwise_affine = not args.nai_fuse_norm_scales
    nai_config_sdpa.linear_weight_dtype = torch.bfloat16
    nai_config_sdpa.emb_weight_dtype = torch.bfloat16
    nai_config_sdpa.norm_weight_dtype = torch.bfloat16
    nai_config_flex = nai_config_sdpa.model_copy(update={
        'attn_impl': T5AttnImpl.Flex,
        'flex_kernel_options': {
            'BLOCK_M': args.flex_block_m,
            'BLOCK_N': args.flex_block_n,
        },
    })

    input_ids: LongTensor = torch.arange(args.ctx_len, device=device, dtype=torch.long).unsqueeze(0).repeat_interleave(args.batch_size, dim=0)
    bool_mask: BoolTensor = (torch.arange(args.ctx_len, device=device, dtype=torch.long) > args.visible_tokens).unsqueeze(0).repeat_interleave(args.batch_size, dim=0)

    bench_subjects: dict[BenchSubject, NiladicModelFwd] = {}
    bench_subjects_c: dict[BenchSubject, NiladicModelFwd] = {}

    # don't bother with weight init, just construct
    if args.bench_hf:
        torch.manual_seed(args.seed)
        with device:
            hf_enc = HFT5EncoderModel(hf_config).eval().to(dtype)
        bind_hf_fwd: Callable[[HFT5EncoderModel], NiladicModelFwd] = lambda hf_enc: lambda: hf_enc(
            input_ids,
            attention_mask=(None if args.disable_mask else bool_mask),
        ).last_hidden_state
        hf_enc_c = torch.compile(hf_enc, dynamic=False, fullgraph=True)
        hf_fwd: NiladicModelFwd = bind_hf_fwd(hf_enc)
        hf_fwd_c: NiladicModelFwd = bind_hf_fwd(hf_enc_c)
        bench_subjects[BenchSubject.HF] = hf_fwd
        bench_subjects_c[BenchSubject.HF] = hf_fwd_c

    if args.bench_nai_sdpa:
        torch.manual_seed(args.seed)
        with device:
            nai_enc_sdpa = T5EncoderStack(nai_config_sdpa).eval()
        bind_nai_sdpa_fwd: Callable[[T5EncoderStack], NiladicModelFwd] = lambda nai_enc_sdpa: lambda: nai_enc_sdpa(
            input_ids,
            input_mask=(None if args.disable_mask else bool_mask),
        )
        nai_enc_sdpa_c = torch.compile(nai_enc_sdpa, dynamic=False, fullgraph=True)
        nai_sdpa_fwd: NiladicModelFwd = bind_nai_sdpa_fwd(nai_enc_sdpa)
        nai_sdpa_fwd_c: NiladicModelFwd = bind_nai_sdpa_fwd(nai_enc_sdpa_c)
        bench_subjects[BenchSubject.NAI_SDPA] = nai_sdpa_fwd
        bench_subjects_c[BenchSubject.NAI_SDPA] = nai_sdpa_fwd_c

    if args.bench_nai_flex:
        torch.manual_seed(args.seed)
        with device:
            nai_enc_flex = T5EncoderStack(nai_config_flex).eval()
        nai_enc_flex.bind_score_mods(args.ctx_len)
        def bind_nai_flex_fwd(nai_enc_sdpa: T5EncoderStack) -> NiladicModelFwd:
            from nai_t5.t5_encoder import make_self_attn_block_mask
            def fwd() -> FloatTensor:
                if args.disable_block_mask:
                    block_mask: Optional[BlockMask] = None
                else:
                    block_mask: BlockMask = make_self_attn_block_mask(
                        mask=bool_mask,
                        mask_pad_queries=args.flex_mask_pad_queries,
                    )
                return nai_enc_sdpa(
                    input_ids,
                    block_mask=block_mask,
                )
            return fwd
        nai_enc_flex_c = torch.compile(nai_enc_flex, dynamic=False, fullgraph=True)
        nai_flex_fwd: NiladicModelFwd = bind_nai_flex_fwd(nai_enc_flex)
        nai_flex_fwd_c: NiladicModelFwd = bind_nai_flex_fwd(nai_enc_flex_c)
        bench_subjects[BenchSubject.NAI_Flex] = nai_flex_fwd
        bench_subjects_c[BenchSubject.NAI_Flex] = nai_flex_fwd_c

    if args.enable_cudnn_sdpa:
        torch.backends.cuda.enable_cudnn_sdp(True)
        get_sdpa_ctx = partial(sdpa_kernel, SDPBackend.CUDNN_ATTENTION)
    else:
        torch.backends.cuda.enable_cudnn_sdp(False)
        get_sdpa_ctx = nullcontext

    result_msi: dict[BenchSubject, float] = {}
    result_c_msi: dict[BenchSubject, float] = {}
    
    for subjects, compiled, result_out in zip(
        (bench_subjects, bench_subjects_c),
        (False, True),
        (result_msi, result_c_msi),
    ):
        qualifier = ' compiled' if compiled else ''
        for subject, fwd in subjects.items():
            disclaimer = ' [NOTE: Flex requires compilation to be fast]' if subject == BenchSubject.NAI_Flex and not compiled else ''
            print(f'benchmarking {subject}{qualifier}...{disclaimer}')
            with get_sdpa_ctx(), inference_mode():
                ms_per_iter: float = do_bench(fwd)
            iter_per_s: float = 1000 / ms_per_iter
            result_out[subject] = ms_per_iter
            print(f'''{subject}{qualifier}:
{ms_per_iter:7.1f}ms
{iter_per_s:7.1f}it/sec
''')
    pass

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ckpt', type=Checkpoint, default=Checkpoint.V1_1XL, choices=[t.value for t in Checkpoint])
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--ctx-len', type=int, default=512)
    parser.add_argument('--visible-tokens', type=int, default=512)
    parser.add_argument('--nai-fuse-norm-scales', action='store_true')
    parser.add_argument('--flex-block-m', type=int, default=128, help='kernel option BLOCK_M for flex attention')
    parser.add_argument('--flex-block-n', type=int, default=64, help='kernel option BLOCK_N for flex attention')
    parser.add_argument('--disable-block-mask', action='store_true', help="when visible_tokens==ctx_len, you can disable the block mask to compare SDPA vs Flex without the advantage of sparsity. this is also useful for testing whether block mask brings its own cost in the no-op case.")
    parser.add_argument('--disable-mask', action='store_true')
    parser.add_argument('--flex-mask-pad-queries', default=True, action=BooleanOptionalAction, help="in Flex, pad queries will attend to nothing, and rely on safe_softmax to prevent inf probabilities. this improves sparsity but may make parity tests fail (outputs in pad positions will be 0-valued).")
    parser.add_argument('--bench-hf', action='store_true')
    parser.add_argument('--bench-nai-sdpa', action='store_true')
    parser.add_argument('--bench-nai-flex', action='store_true')
    parser.add_argument('--enable-cudnn-sdpa', action='store_true', help="cuDNN SDPA backend is faster than the default 'memory-efficient' backend but is not widely available and may be buggy.")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(Args(**vars(args)))