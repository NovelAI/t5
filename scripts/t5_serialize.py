#!/usr/bin/env python

import argparse
from pathlib import Path
from typing import OrderedDict, Optional

from enum import Enum
from torch import Tensor
import torch

from transformers.models.umt5 import UMT5ForConditionalGeneration, UMT5EncoderModel
from transformers.models.umt5.configuration_umt5 import UMT5Config
from transformers import LlamaTokenizerFast
from transformers import AutoConfig
from transformers.models.t5 import T5ForConditionalGeneration, T5TokenizerFast, T5EncoderModel
from transformers.models.t5.configuration_t5 import T5Config as T5ConfigHF

from nai_t5 import T5, T5EncoderStack, T5Config, hf_to_based_t5_state, hf_to_based_t5_enc_state, to_based_config

class DType(str, Enum):
    Float16 = 'float16'
    Float32 = 'float32'
    BFloat16 = 'bfloat16'

dtype_map: dict[DType, torch.dtype] = {
    DType.Float16: torch.float16,
    DType.Float32: torch.float32,
    DType.BFloat16: torch.bfloat16,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model-name", type=str, required=True, help="HF model name. example: google/t5-v1_1-small"
    )
    parser.add_argument("--enc", action='store_true')
    parser.add_argument("--encdec", action='store_true')
    parser.add_argument(
        "--weight-dtype", default=None, type=DType, choices=[t.value for t in DType],
    )
    parser.add_argument("--tensorizer", action='store_true')
    parser.add_argument("--dtensor", action='store_true')
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        required=True,
        help="output directory. example: /mnt/clusterstorage/models/t5-goog/t5-v1_1-small",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    assert args.enc or args.encdec, "must specify at least one of --enc or --encdec for export"
    assert args.dtensor or args.tensorizer, "must specify at least one of --dtensor or --tensorizer"

    wants_enc: bool = args.enc
    wants_encdec: bool = args.encdec

    hf_config: T5ConfigHF | UMT5Config = AutoConfig.from_pretrained(args.model_name)
    assert isinstance(hf_config, (T5ConfigHF, UMT5Config))

    is_umt5: bool = hf_config.model_type == 'umt5'

    if is_umt5:
        hf_tokenizer: LlamaTokenizerFast = LlamaTokenizerFast.from_pretrained(args.model_name)
    else:
        hf_tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained(args.model_name, legacy=False)

    weight_dtype: Optional[torch.dtype] = None if args.weight_dtype is None else dtype_map[args.weight_dtype]
    hf_dtype_kwargs = {} if weight_dtype is None else {'torch_dtype': weight_dtype}

    if wants_encdec:
        if is_umt5:
            hf_t5: UMT5ForConditionalGeneration = UMT5ForConditionalGeneration.from_pretrained(
                args.model_name,
                **hf_dtype_kwargs,
            ).eval()
        else:
            hf_t5: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
                args.model_name,
                **hf_dtype_kwargs,
            ).eval()
    elif wants_enc:
        if is_umt5:
            hf_t5_enc: UMT5EncoderModel = UMT5EncoderModel.from_pretrained(
                args.model_name,
                **hf_dtype_kwargs,
            ).eval()
        else:
            hf_t5_enc: T5EncoderModel = T5EncoderModel.from_pretrained(
                args.model_name,
                **hf_dtype_kwargs,
            ).eval()
    else:
        raise ValueError(f"No weights targeted for export, with target {args.target}")

    my_config: T5Config = to_based_config(hf_config, n_tokens=hf_tokenizer.model_max_length)
    if weight_dtype is not None:
        my_config.emb_weight_dtype = weight_dtype
        my_config.linear_weight_dtype = weight_dtype
        my_config.norm_weight_dtype = weight_dtype

    if wants_encdec:
        print('Constructing NAI encdec')
        my_t5 = T5(my_config).eval()
        print('Loading HF weights into NAI encdec')

        # yes I know we could stream it instead of materializing the whole state dict
        # but you have to do more than just key-mapping (have to accumulate QKV projections to fuse them)
        # ain't nobody got time for that
        hf_state: OrderedDict[str, Tensor] = hf_t5.state_dict()
        converted_state: dict[str, Tensor] = hf_to_based_t5_state(hf_state, my_config)
        my_t5.load_state_dict(converted_state)
        del hf_state, converted_state

        if args.tensorizer:
            from tensorizer import TensorSerializer
            out_encdec_tensors: Path = args.out_dir / 'encdec.tensors'
            print(f"Writing encdec tensors to {out_encdec_tensors}...")
            serializer = TensorSerializer(out_encdec_tensors)
            serializer.write_module(my_t5, include_non_persistent_buffers=False)
            serializer.close()
            print(f"Wrote encdec tensors to   {out_encdec_tensors}.")
        if args.dtensor:
            import torch.distributed.checkpoint as dcp
            out_encdec_dcp: Path = args.out_dir / 'encdec'
            out_encdec_dcp.mkdir(parents=True, exist_ok=True)
            print(f"Writing encdec DCP to     {out_encdec_dcp}...")
            encdec_writer = dcp.FileSystemWriter(out_encdec_dcp)
            dcp.state_dict_saver.save(
                state_dict={"model": my_t5},
                storage_writer=encdec_writer,
            )
            print(f"Wrote encdec DCP to       {out_encdec_dcp}.")

        my_t5_enc: Optional[T5EncoderStack] = my_t5.encoder if wants_enc else None
    elif wants_enc:
        print('Constructing NAI enc')
        my_t5_enc = T5EncoderStack(my_config).eval()
        print('Loading HF weights into NAI enc')

        hf_enc_state: OrderedDict[str, Tensor] = hf_t5_enc.state_dict()
        converted_enc_state: dict[str, Tensor] = hf_to_based_t5_enc_state(hf_enc_state, my_config)
        my_t5_enc.load_state_dict(converted_enc_state)
        del hf_enc_state, converted_enc_state
    if wants_enc:
        assert my_t5_enc is not None
        if args.tensorizer:
            from tensorizer import TensorSerializer
            out_enc_tensors: Path = args.out_dir / 'enc.tensors'
            print(f"Writing enc tensors to    {out_enc_tensors}...")
            enc_serializer = TensorSerializer(out_enc_tensors)
            enc_serializer.write_module(my_t5_enc, include_non_persistent_buffers=False)
            enc_serializer.close()
            print(f"Wrote enc tensors to      {out_enc_tensors}.")
        if args.dtensor:
            import torch.distributed.checkpoint as dcp
            out_enc_dcp: Path = args.out_dir / 'encdec'
            out_enc_dcp.mkdir(parents=True, exist_ok=True)
            print(f"Writing enc DCP to        {out_enc_dcp}...")
            enc_writer = dcp.FileSystemWriter(out_enc_dcp)
            dcp.state_dict_saver.save(
                state_dict={"model": my_t5_enc},
                storage_writer=enc_writer,
            )
            print(f"Wrote enc DCP to          {out_enc_dcp}.")
        

    config_json: str = my_config.model_dump_json(indent=2)
    config_out: Path = args.out_dir / 'config.json'
    with open(config_out, 'w') as f:
        f.write(config_json)
    print(f"Wrote NAI config to       {config_out}.")

    hf_spiece_model = Path(hf_tokenizer.vocab_file)
    hf_tok_assets_dir = hf_spiece_model.parent
    hf_tok_config_file = hf_tok_assets_dir / 'tokenizer_config.json'

    cmd = f"python -m scripts.tokenizer_hf_to_sentencepiece" if __package__ else "tokenizer_hf_to_sentencepiece.py"

    print(f"""To convert HF's spiece.model into one which includes all the mask tokens, use:

{cmd} \\
--tokenizer-in     {hf_tokenizer.vocab_file} \\
--tokenizer-config {hf_tok_config_file} \\
--tokenizer-out    {args.out_dir / 'spiece.model'}

(We would've done that for you, but needs to be in a separate runtime to avoid a clash of sentencepiece filename in descriptor pool)
""")

if __name__ == "__main__":
    main()
