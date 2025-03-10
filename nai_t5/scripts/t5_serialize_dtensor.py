#!/usr/bin/env python

import argparse
from pathlib import Path
from typing import Any
from shutil import copyfile
import json
from itertools import chain
from os import environ

from enum import Enum
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor import init_device_mesh
from torch.distributed.device_mesh import DeviceMesh
from tensorizer import TensorDeserializer

from nai_t5 import T5, T5EncoderStack, T5Config

class ExportTarget(str, Enum):
    Enc = 'enc'
    EncDec = 'encdec'
    EncAndEncDec = 'enc_and_encdec'

# from https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/__init__.py
def create_mesh(world_size: int, tp=1, dp=-1, pp=1, fsdp_hybrid=1) -> DeviceMesh:
    # fsdp_hybrid 1 means full shard, high number means replicate
    if dp == -1:
        dp = world_size // (tp * pp)
    assert tp * dp * pp == world_size, f"Mesh dimensions {tp} * {dp} * {pp} don't fit {world_size}"
    assert pp == 1, "Pipeline parallel not implemented yet"

    dims = []
    names = []
    for d, name in zip([pp, dp, tp], ["pp", "dp", "tp"], strict=True):
        if d > 1:
            dims.append(d)
            names.append(name)
    if torch.distributed.get_rank() == 0:
        print(f"{len(dims)}D device mesh with {names}, {dims}")
    names = tuple(names)
    return init_device_mesh("cpu", dims, mesh_dim_names=names)

def main(
    local_rank: int,
    world_size: int,
    in_dir_tensorizer: Path,
    out_dir: Path,
    encoder_only: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path: Path = in_dir_tensorizer / 'config.json'
    with open(config_path, 'r') as f:
        config_dict: dict[str, Any] = json.load(f)
    my_config: T5Config = T5Config.model_validate(config_dict)

    print(f"Constructing model...")
    with torch.device('meta'):
        model: T5EncoderStack | T5
        if encoder_only:
            tensor_fname = 'enc.tensors'
            out_subdir = 'enc'
            my_t5_enc: T5EncoderStack = T5EncoderStack(my_config).eval()
            model = my_t5_enc
        else:
            tensor_fname = 'encdec.tensors'
            out_subdir = 'encdec'
            my_t5: T5 = T5(my_config).eval()
            model = my_t5
    print(f"Constructed model.")
    for tensor in chain(model.parameters(), model.buffers()):
        assert tensor.device == torch.device("meta")

    print(f"Loading tensorizer weights {in_dir_tensorizer / tensor_fname} into model...")
    deserializer = TensorDeserializer(in_dir_tensorizer / tensor_fname, lazy_load=True)
    deserializer.load_into_module(model)
    deserializer.close()
    print(f"Loaded tensorizer weights  {in_dir_tensorizer / tensor_fname} into model.")
    
    print(f"Initializing process group")
    dist.init_process_group("gloo", rank=local_rank, world_size=world_size)

    print(f"Initializing device mesh")
    mesh: DeviceMesh = create_mesh(world_size)

    print(f"Sharding model")
    model = fully_shard(model, mesh=mesh)

    print(f"Writing DTensor ckpt to {out_dir}...")
    fs_storage_writer = dcp.FileSystemWriter(out_dir / out_subdir)
    dcp.state_dict_saver.save(
        state_dict={"model": model},
        storage_writer=fs_storage_writer,
    )
    dist.barrier()
    print(f"Wrote DTensor ckpt to   {out_dir}.")

    if local_rank == 0:
        config_out: Path = out_dir / 'config.json'
        copyfile(config_path, config_out)
        print(f"Copied config to        {config_out}.")
        spiece_in: Path = in_dir_tensorizer / 'spiece.model'
        spiece_out: Path = out_dir / 'spiece.model'
        copyfile(spiece_in, spiece_out)
        print(f"Copied spiece.model to  {spiece_out}.")
    dist.barrier()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--in-dir-tensorizer",
        type=Path,
        required=True,
        help="input directory. example: /mnt/clusterstorage/models/nait5-tensorizer/goog/t5-v1_1-small-f32",
    )
    parser.add_argument("--encoder-only", action='store_true')
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        required=True,
        help="output directory. example: /mnt/clusterstorage/models/nait5-dtensor/goog/t5-v1_1-small-f32",
    )
    args = parser.parse_args()
    main(
        local_rank=int(environ['LOCAL_RANK']),
        world_size=int(environ['WORLD_SIZE']),
        in_dir_tensorizer=args.in_dir_tensorizer,
        out_dir=args.out_dir,
        encoder_only=args.encoder_only,
    )
