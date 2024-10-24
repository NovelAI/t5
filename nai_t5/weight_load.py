from nai_t5 import T5, T5EncoderStack, T5Config
from typing import Optional, OrderedDict, TYPE_CHECKING, Any, Protocol
from dataclasses import dataclass
from functools import partial
import contextlib
import torch
from torch import FloatTensor, inference_mode
from torch.nn import Linear
from tensorizer import TensorDeserializer, TensorType
import re

if TYPE_CHECKING:
    from tensorizer._tensor_path import _TensorPath
else:
    _TensorPath = Any

@dataclass
class Scales:
    attn_out_scales: list[float]
    attn_out_scales_cp_hat: list[float]
    ffn_out_scales: list[float]
    ffn_out_scales_cp_hat: list[float]
    ln1_eps_scales: list[float]
    ln2_eps_scales: list[float]
    final_norm_eps_scale: float

def resolve_scales(
    attn_out_scales: Optional[list[float]] = None,
    ffn_out_scales: Optional[list[float]] = None,
) -> Scales:
    attn_out_scales: FloatTensor = torch.tensor(attn_out_scales, dtype=torch.float32)
    attn_out_scales_cp: FloatTensor = attn_out_scales.cumprod(-1)

    ffn_out_scales: FloatTensor = torch.tensor(ffn_out_scales, dtype=torch.float32)
    ffn_out_scales_cp: FloatTensor = ffn_out_scales.cumprod(-1)

    attn_out_scales_cp_hat = attn_out_scales_cp.clone()
    attn_out_scales_cp_hat[1:].mul_(ffn_out_scales_cp[:-1])

    ffn_out_scales_cp_hat = ffn_out_scales_cp.clone()
    ffn_out_scales_cp_hat.mul_(attn_out_scales_cp)

    ln1_eps_scales = ffn_out_scales_cp_hat.roll(1, dims=-1)
    ln1_eps_scales[0].copy_(1)

    ln2_eps_scales = attn_out_scales_cp_hat

    final_norm_eps_scale = ffn_out_scales_cp_hat[-1].item()

    return Scales(
        attn_out_scales=attn_out_scales.tolist(),
        attn_out_scales_cp_hat=attn_out_scales_cp_hat.tolist(),
        ffn_out_scales=ffn_out_scales.tolist(),
        ffn_out_scales_cp_hat=ffn_out_scales_cp_hat.tolist(),
        ln1_eps_scales=ln1_eps_scales.tolist(),
        ln2_eps_scales=ln2_eps_scales.tolist(),
        final_norm_eps_scale=final_norm_eps_scale,
    )

class AcceptFusion(Protocol):
    @staticmethod
    def __call__(attr: str, tensor: FloatTensor) -> None: ...

def fuse_norm_scale(
    w: FloatTensor,
    scale: FloatTensor,
    scale_dtype: Optional[torch.dtype] = None,
) -> FloatTensor:
    dtype = scale_dtype or scale.dtype
    scale_diag: FloatTensor = torch.eye(
        scale.size(-1),
        device=scale.device,
        dtype=dtype,
    ) * scale.type(dtype).unsqueeze(-1)
    higher_type: torch.dtype = torch.promote_types(w.dtype, scale_diag.dtype)
    w.copy_(w.type(higher_type) @ scale_diag)

class FusingDeserializer(TensorDeserializer):
    def load_with_fusions(
        self,
        m: T5 | T5EncoderStack,
        fusion_dtype: Optional[torch.dtype] = None,
        scale_dtype: Optional[torch.dtype] = None,
        fuse_norm_scales = True,
        attn_out_scales: Optional[list[float]] = None,
        ffn_out_scales: Optional[list[float]] = None,
        verify_hash: Optional[bool] = None,
    ) -> int:
        """
        Load weights into a model, fusing or scaling layers as we go.
        """
        config: T5Config = m.config
        scales: Scales = resolve_scales(
            attn_out_scales or [1.] * config.num_layers,
            ffn_out_scales or [1.] * config.num_layers,
        )

        modules: OrderedDict[str, torch.nn.Module] = OrderedDict()

        if verify_hash is None:
            verify_hash = self._verify_hash

        for name, module in m.named_modules():
            modules[name] = module
        
        keys: tuple[str, ...] = tuple(self._metadata.keys())

        is_ln1 = re.compile(r'layers\.\d+\.ln1.weight$')
        is_ln2 = re.compile(r'layers\.\d+\.ln2.weight$')
        is_ln1or2 = re.compile(r'layers\.\d+\.ln[12].weight$')
        ln_wants_lin: dict[str, str] = {
            **{k: k.replace('ln1', 'attn.qkv_proj') for k, *_ in keys if re.search(is_ln1, k)},
            **{k: k.replace('ln2', 'ffn.ff_in') for k, *_ in keys if re.search(is_ln2, k)},
        }
        lin_wants_ln: dict[str, str] = {v: k for k, v in ln_wants_lin.items()}
        wants_fusion = ln_wants_lin.keys() | lin_wants_ln.keys()

        pending_fusion: dict[str, AcceptFusion] = {}

        def fuse_me(
            w_name: str,
            w_mod: Linear,
            w_attr: str,
            w: FloatTensor,
            s_name: str,
            scale: FloatTensor,
        ) -> None:
            fuse_norm_scale(w, scale, scale_dtype=scale_dtype)
            w_mod.register_parameter(w_attr, w)
            wants_fusion.remove(s_name)
            wants_fusion.remove(w_name)

        tensor_ct = len(keys)

        buffer_type = TensorType.BUFFER
        param_type = TensorType.PARAM
        state_dict_type = TensorType.STATE_DICT

        bulk_loader = self._bulk_load(keys, verify_hash=verify_hash)
        with contextlib.closing(bulk_loader), inference_mode():
            for copied_data in bulk_loader:
                path: _TensorPath = copied_data.header.name
                entry = self._metadata[path]
                if entry.type is state_dict_type:
                    raise NotImplementedError(
                        "This was serialized using"
                        " TensorSerializer.write_state_dict(), so it cannot be"
                        " loaded using TensorDeserializer.load_into_module()."
                        " Use the TensorDeserializer object directly as a"
                        " state_dict mapping instead."
                    )
                elif (
                    entry.type is not buffer_type
                    and entry.type is not param_type
                ):
                    raise RuntimeError(f"Invalid tensor type: {entry.type}")
                elif not path.is_str_:
                    raise NotImplementedError(
                        "Cannot deserialize structured tensor keys as a module;"
                        " try using the TensorDeserializer directly"
                        " as a state_dict mapping instead."
                    )
                tensor = copied_data.parameter
                name: str = path.normalize_()
                obj_path, attr = name.rsplit(".", 1)
                module: torch.nn.Module = modules[obj_path]

                if entry.type is param_type:
                    if name in wants_fusion:
                        if re.search(is_ln1or2, name):
                            counterpart_dict: dict[str, str] = ln_wants_lin
                            fuse_kwargs = { 's_name': name, 'scale': tensor }
                        else:
                            counterpart_dict: dict[str, str] = lin_wants_ln
                            fuse_kwargs = { 'w_mod': module, 'w_name': name, 'w_attr': attr, 'w': tensor }
                        if name in pending_fusion:
                            pending_fusion[name](**fuse_kwargs)
                            del pending_fusion[name]
                        else:
                            counterpart: str = counterpart_dict[name]
                            pending_fusion[counterpart] = partial(fuse_me, **fuse_kwargs)
                    else:
                        module.register_parameter(attr, tensor)
                elif entry.type is buffer_type:
                    module.register_buffer(attr, tensor)

        self._file.close()
        assert not wants_fusion, f"Unfused: {wants_fusion}"
        return tensor_ct

    
    
