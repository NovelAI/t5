#!/usr/bin/env python

from typing import Any
import argparse
from os import environ
from pathlib import Path
import json
from contextlib import nullcontext
from itertools import chain

import torch
from torch import FloatTensor, LongTensor, BoolTensor, Tensor, inference_mode
from torch.amp import autocast
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._composable.fsdp._fsdp_api import MixedPrecisionPolicy
from torch.distributed._tensor import init_device_mesh
from torch.distributed.device_mesh import DeviceMesh
from tensorizer import TensorDeserializer
from sentencepiece import SentencePieceProcessor

from nai_t5 import T5Config, T5EncoderStack


from torch import Tensor
from typing import Optional
from torch.linalg import matrix_norm
def stat(t: Tensor, label: Optional[str] = None) -> None:
    print(tuple(t.shape), str(t.dtype).removeprefix('torch.'), f'σ={t.std().item():g}', f'μ={t.mean().item():g}', f'norm={matrix_norm(t.float(), ord=2).squeeze().cpu()}', label or '')


# from https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/__init__.py
def create_mesh_1d(world_size: int, tp=1, dp=-1, pp=1, fsdp_hybrid=1) -> DeviceMesh:
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
    return init_device_mesh("cuda", dims, mesh_dim_names=names)

def create_mesh_2d(global_rank: int, world_size: int, gpus_per_node: int) -> DeviceMesh:
    mesh = []
    # We will assume each node will have 8 GPUs, so dim=1 will be inter-node, and dim-0 will be intra node.
    if global_rank == 0:
        if world_size % gpus_per_node != 0 and world_size != 1:
            raise ValueError("World size must be a multiple of gpus_per_node")

    if world_size == 1:
        mesh.append([0])

    else:
        for i in range(world_size // gpus_per_node):
            mesh.append([j for j in range(i * gpus_per_node, (i + 1) * gpus_per_node)])

    print(f"Device Mesh Constructed: {mesh}")

    device_mesh = DeviceMesh(
        device_type="cuda",
        mesh=mesh,
        mesh_dim_names=["fsdp", "dp"],
    )
    return device_mesh

def main(
    global_rank: int,
    local_rank: int,
    world_size: int,
    local_world_size: int,
    in_dir_tensorizer: Path,
    in_dir_dcp: Optional[Path],
    mesh_2d: bool,
    mixed_precision: bool,
    fsdp2: bool,
):
    dist.init_process_group("nccl", rank=global_rank, world_size=world_size)
    if mesh_2d:
        mesh: DeviceMesh = create_mesh_2d(global_rank, world_size, local_world_size)
    else:
        mesh: DeviceMesh = create_mesh_1d(world_size)

    config_path: Path = in_dir_tensorizer / 'config.json'
    with open(config_path, 'r') as f:
        config_dict: dict[str, Any] = json.load(f)
    my_config: T5Config = T5Config.model_validate(config_dict)

    with torch.device('meta'):
        model = T5EncoderStack(my_config).eval()
        dmodel = T5EncoderStack(my_config).eval()

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    sp = SentencePieceProcessor(model_file=str(in_dir_tensorizer / 'spiece.model'))
    in_ids: list[int] = sp.encode('hello world')
    in_ids.append(sp.eos_id())
    input_ids: LongTensor = torch.tensor(in_ids, device=device).unsqueeze(0)
    attn_mask: BoolTensor = torch.ones_like(input_ids, dtype=torch.bool)

    model.to_empty(device=device)

    deserializer = TensorDeserializer(in_dir_tensorizer / 'enc.tensors', lazy_load=True, device=device)
    deserializer.load_into_module(model)
    if in_dir_dcp is None:
        # tensorizer loading, if used, has to be done *before* sharding
        # because it reassigns tensors instead of reusing existing storage.
        deserializer.load_into_module(dmodel)
    deserializer.close()

    if mixed_precision:
        autocast_ctx = autocast(device_type=device.type, dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext()

    if fsdp2:
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            cast_forward_inputs=False,
        ) if mixed_precision else MixedPrecisionPolicy()
        dmodel = fully_shard(
            dmodel,
            mesh=mesh,
            mp_policy=mp_policy,
        )
        if in_dir_dcp is not None:
            for tensor in chain(dmodel.parameters(), dmodel.buffers()):
                assert tensor.device == torch.device("meta")
            dmodel.to_empty(device=device)
    else:
        mp_kwargs = {
            'mixed_precision': MixedPrecision(
                param_dtype=torch.bfloat16,
            )
        } if mixed_precision else {}
        dmodel = FSDP(
            dmodel,
            device_id=device,
            device_mesh=mesh,
            auto_wrap_policy=size_based_auto_wrap_policy,
            use_orig_params=False,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD if mesh_2d else ShardingStrategy.FULL_SHARD,
            **mp_kwargs,
        )

    if in_dir_dcp is not None:
        fs_storage_reader = dcp.FileSystemReader(in_dir_dcp / 'enc')
        dcp.state_dict_loader.load(
            state_dict={"model": dmodel},
            storage_reader=fs_storage_reader,
        )
        dist.barrier()

    seed = 42
    with inference_mode():
        # seed the random, so that we can parity-test things like dropout (if enabled)
        torch.manual_seed(seed)
        sharded: FloatTensor = dmodel(
            input_ids=input_ids,
            input_mask=attn_mask,
        )
        dist.barrier() # barrier isn't strictly necessary, but will make logging less interleaved
        torch.manual_seed(seed)
        with autocast_ctx:
            unsharded: FloatTensor = model(
                input_ids=input_ids,
                input_mask=attn_mask,
            )
    compare_dtype = torch.promote_types(sharded.dtype, unsharded.dtype)
    diff = sharded.float().sub(unsharded.float())
    qs = torch.tensor([.5, .75, .9, .95, .99, .999, .9999], device=device)
    q = diff.abs().quantile(qs)
    print(f'quantiles ({qs.cpu()}):\n           {q.cpu()}')
    ref_std = unsharded.std()
    print(f'ref std    {ref_std.cpu():g}')
    print(f'q ratio   ({qs.cpu()}):\n           {qs.div(ref_std).cpu()}')
    stat(diff, 'diff')
    assert (
        sharded.type(compare_dtype).allclose(unsharded.type(compare_dtype))
    ), "sharded and unsharded outputs do not match"
    print("sharded and unsharded outputs matched")
    pass  # somewhere to put your breakpoint
    dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir-tensorizer",
        type=Path,
        required=True,
        help="naiT5 tensorizer weights directory. example: /mnt/clusterstorage/models/nait5-tensorizer/goog/t5-v1_1-small-f32",
    )
    parser.add_argument(
        "--in-dir-dcp",
        default=None,
        type=Path,
        required=False,
        help="[Not compatible with 2d-meshed FSDP2: 2D state dicts not supported] naiT5 DTensor weights directory. example: /mnt/clusterstorage/models/nait5-dtensor/goog/t5-v1_1-small-f32",
    )
    parser.add_argument(
        "--mesh-2d",
        action='store_true',
        help="[Not compatible with FSDP2] 2D mesh resembles how we'll run in practice, but will break FSDP2 (weight loading will give you zero-weights)",
    )
    parser.add_argument(
        "--fsdp2",
        action='store_true',
        help="Only known to work with 1D mesh",
    )
    args = parser.parse_args()
    main(
        global_rank=int(environ['RANK']),
        local_rank=int(environ['LOCAL_RANK']),
        world_size=int(environ['WORLD_SIZE']),
        local_world_size=int(environ['LOCAL_WORLD_SIZE']),
        in_dir_tensorizer=args.in_dir_tensorizer,
        in_dir_dcp=args.in_dir_dcp,
        mesh_2d=args.mesh_2d,
        mixed_precision=False,
        fsdp2=args.fsdp2,
    )
