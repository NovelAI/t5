from nai_t5 import T5, T5Config
from nai_t5.t5_encoder import T5EncoderStack, T5EncoderLayer
from nai_t5.t5_decoder import T5DecoderStack, T5DecoderLayer
from typing import Optional, OrderedDict, TYPE_CHECKING, Any, Protocol, Literal, Callable
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
class EncScales:
    attn_out_scales: list[float]
    attn_out_scales_cp_hat: list[float]
    ffn_out_scales: list[float]
    ffn_out_scales_cp_hat: list[float]
    ln1_eps_scales: list[float]
    ln2_eps_scales: list[float]
    final_norm_eps_scale: float

@dataclass
class DecScales:
    self_attn_out_scales: list[float]
    self_attn_out_scales_cp_hat: list[float]
    cross_attn_out_scales: list[float]
    cross_attn_out_scales_cp_hat: list[float]
    ffn_out_scales: list[float]
    ffn_out_scales_cp_hat: list[float]
    ln1_eps_scales: list[float]
    ln2_eps_scales: list[float]
    ln3_eps_scales: list[float]
    final_norm_eps_scale: float

def resolve_enc_scales(
    attn_out_scales: Optional[list[float]] = None,
    ffn_out_scales: Optional[list[float]] = None,
) -> EncScales:
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

    return EncScales(
        attn_out_scales=attn_out_scales.tolist(),
        attn_out_scales_cp_hat=attn_out_scales_cp_hat.tolist(),
        ffn_out_scales=ffn_out_scales.tolist(),
        ffn_out_scales_cp_hat=ffn_out_scales_cp_hat.tolist(),
        ln1_eps_scales=ln1_eps_scales.tolist(),
        ln2_eps_scales=ln2_eps_scales.tolist(),
        final_norm_eps_scale=final_norm_eps_scale,
    )

def resolve_dec_scales(
    self_attn_out_scales: Optional[list[float]] = None,
    cross_attn_out_scales: Optional[list[float]] = None,
    ffn_out_scales: Optional[list[float]] = None,
) -> DecScales:
    self_attn_out_scales: FloatTensor = torch.tensor(self_attn_out_scales, dtype=torch.float32)
    self_attn_out_scales_cp: FloatTensor = self_attn_out_scales.cumprod(-1)

    cross_attn_out_scales: FloatTensor = torch.tensor(cross_attn_out_scales, dtype=torch.float32)
    cross_attn_out_scales_cp: FloatTensor = cross_attn_out_scales.cumprod(-1)

    ffn_out_scales: FloatTensor = torch.tensor(ffn_out_scales, dtype=torch.float32)
    ffn_out_scales_cp: FloatTensor = ffn_out_scales.cumprod(-1)

    self_attn_out_scales_cp_hat = self_attn_out_scales_cp.clone()
    self_attn_out_scales_cp_hat[1:].mul_(cross_attn_out_scales_cp[:-1])
    self_attn_out_scales_cp_hat[1:].mul_(ffn_out_scales_cp[:-1])

    cross_attn_out_scales_cp_hat = cross_attn_out_scales_cp.clone()
    cross_attn_out_scales_cp_hat.mul_(self_attn_out_scales_cp)
    cross_attn_out_scales_cp_hat[1:].mul_(ffn_out_scales_cp[:-1])

    ffn_out_scales_cp_hat = ffn_out_scales_cp.clone()
    ffn_out_scales_cp_hat.mul_(self_attn_out_scales_cp)
    ffn_out_scales_cp_hat.mul_(cross_attn_out_scales_cp)

    ln1_eps_scales = ffn_out_scales_cp_hat.roll(1, dims=-1)
    ln1_eps_scales[0].copy_(1)

    ln2_eps_scales = self_attn_out_scales_cp_hat

    ln3_eps_scales = cross_attn_out_scales_cp_hat

    final_norm_eps_scale = ffn_out_scales_cp_hat[-1].item()

    return DecScales(
        self_attn_out_scales=self_attn_out_scales.tolist(),
        self_attn_out_scales_cp_hat=self_attn_out_scales_cp_hat.tolist(),
        cross_attn_out_scales=cross_attn_out_scales.tolist(),
        cross_attn_out_scales_cp_hat=cross_attn_out_scales_cp_hat.tolist(),
        ffn_out_scales=ffn_out_scales.tolist(),
        ffn_out_scales_cp_hat=ffn_out_scales_cp_hat.tolist(),
        ln1_eps_scales=ln1_eps_scales.tolist(),
        ln2_eps_scales=ln2_eps_scales.tolist(),
        ln3_eps_scales=ln3_eps_scales.tolist(),
        final_norm_eps_scale=final_norm_eps_scale,
    )

class AcceptFusion(Protocol):
    @staticmethod
    def __call__(attr: str, tensor: FloatTensor) -> None: ...

def fuse_norm_scale(
    w: FloatTensor,
    ln_scale: FloatTensor,
    scale_via_f32 = False,
) -> FloatTensor:
    higher_type: torch.dtype = torch.float32 if scale_via_f32 else torch.promote_types(w.dtype, ln_scale.dtype)
    scale_diag: FloatTensor = torch.eye(
        ln_scale.size(-1),
        device=ln_scale.device,
        dtype=higher_type,
    ) * ln_scale.type(higher_type).unsqueeze(-1)
    matmul_type = torch.float32 if scale_via_f32 else higher_type
    w.copy_(w.type(matmul_type) @ scale_diag.type(matmul_type))

class FusingDeserializer(TensorDeserializer):
    def load_with_fusions(
        self,
        m: T5 | T5EncoderStack,
        norm_fusion_via_f32 = False,
        fuse_norm_scales = False,
        enc_attn_out_scales: Optional[list[float]] = None,
        enc_ffn_out_scales: Optional[list[float]] = None,
        dec_self_attn_out_scales: Optional[list[float]] = None,
        dec_cross_attn_out_scales: Optional[list[float]] = None,
        dec_ffn_out_scales: Optional[list[float]] = None,
        verify_hash: Optional[bool] = None,
    ) -> int:
        """
        Load weights into a model, fusing or scaling layers as we go.
        """
        config: T5Config = m.config
        enc_scales: EncScales = resolve_enc_scales(
            enc_attn_out_scales or [1.] * config.num_layers,
            enc_ffn_out_scales or [1.] * config.num_layers,
        )
        dec_scales: DecScales = resolve_dec_scales(
            dec_self_attn_out_scales or [1.] * config.num_layers,
            dec_cross_attn_out_scales or [1.] * config.num_layers,
            dec_ffn_out_scales or [1.] * config.num_layers,
        )

        def receives_residual(obj_path: str, qualifier: Optional[Literal['encoder', 'decoder']] = None) -> bool:
            prefix = f"{qualifier}." if qualifier else ''
            return obj_path == f'{prefix}ln' or obj_path.startswith(f'{prefix}layers.')

        match m:
            case T5():
                receives_enc_residual: Callable[[str], bool] = partial(receives_residual, qualifier='encoder')
                receives_dec_residual: Callable[[str], bool] = partial(receives_residual, qualifier='decoder')
                enc: T5EncoderStack = m.encoder
                dec: T5DecoderStack = m.decoder
            case T5EncoderStack():
                receives_enc_residual: Callable[[str], bool] = receives_residual
                receives_dec_residual: Callable[[str], bool] = lambda _: False
                enc: T5EncoderStack = m
                dec: Optional[T5DecoderStack] = None
            case _:
                raise ValueError(f"Unsupported model type: {type(m)}")
        
        for (
            layer,
            ln1_eps_scale,
            ln2_eps_scale,
            ln1_residual_scale,
            ln2_residual_scale,
        ) in zip(
            enc.layers,
            enc_scales.ln1_eps_scales,
            enc_scales.ln2_eps_scales,
            enc_scales.attn_out_scales,
            enc_scales.ffn_out_scales,
        ):
            layer: T5EncoderLayer
            layer.ln1.eps *= ln1_eps_scale
            layer.ln2.eps *= ln2_eps_scale
            # make residual smaller at the same time as we make a layer output smaller
            layer.ln1.residual_scale = ln1_residual_scale
            layer.ln2.residual_scale = ln2_residual_scale
        enc.ln.eps *= enc_scales.final_norm_eps_scale

        if dec is not None:
            for (
                layer,
                ln1_eps_scale,
                ln2_eps_scale,
                ln3_eps_scale,
                ln1_residual_scale,
                ln2_residual_scale,
                ln3_residual_scale,
            ) in zip(
                dec.layers,
                dec_scales.ln1_eps_scales,
                dec_scales.ln2_eps_scales,
                dec_scales.ln3_eps_scales,
                dec_scales.self_attn_out_scales,
                dec_scales.cross_attn_out_scales,
                dec_scales.ffn_out_scales,
            ):
                layer: T5DecoderLayer
                layer.ln1.eps *= ln1_eps_scale
                layer.ln2.eps *= ln2_eps_scale
                layer.ln3.eps *= ln3_eps_scale
                # make residual smaller at the same time as we make a layer output smaller
                layer.ln1.residual_scale = ln1_residual_scale
                layer.ln2.residual_scale = ln2_residual_scale
                layer.ln3.residual_scale = ln3_residual_scale
            dec.ln.eps *= enc_scales.final_norm_eps_scale

        modules: OrderedDict[str, torch.nn.Module] = OrderedDict()

        if verify_hash is None:
            verify_hash = self._verify_hash

        for name, module in m.named_modules():
            modules[name] = module
        
        keys: tuple[str, ...] = tuple(self._metadata.keys())

        is_ln1 = re.compile(r'layers\.(\d+)\.ln1\.weight$')
        is_ln2 = re.compile(r'layers\.(\d+)\.ln2\.weight$')
        is_ln3 = re.compile(r'layers\.(\d+)\.ln3\.weight$')
        is_o_proj = re.compile(r'layers\.(\d+)\.attn\.o_proj\.weight$')
        is_self_o_proj = re.compile(r'layers\.(\d+)\.self_attn\.o_proj\.weight$')
        is_cross_o_proj = re.compile(r'layers\.(\d+)\.cross_attn\.o_proj\.weight$')
        is_ff_out = re.compile(r'layers\.(\d+)\.ffn\.ff_out\.weight$')
        if fuse_norm_scales:
            is_ln1or2or3 = re.compile(r'layers\.(\d+)\.ln[123]\.weight$')
            enc_keys = keys if dec is None else tuple(k for k, *_ in keys if k.startswith('encoder.'))
            dec_keys = () if dec is None else tuple(k for k, *_ in keys if k.startswith('decoder.'))
            enc_ln_wants_lin: dict[str, str] = {
                **{k: k.replace('ln1', 'attn.qkv_proj') for k, *_ in enc_keys if re.search(is_ln1, k)},
                **{k: k.replace('ln2', 'ffn.ff_in') for k, *_ in enc_keys if re.search(is_ln2, k)},
            }
            dec_ln_wants_lin: dict[str, str] = {} if dec is None else {
                **{k: k.replace('ln1', 'self_attn.qkv_proj') for k, *_ in dec_keys if re.search(is_ln1, k)},
                # TODO: presumably we need to fuse ln2's norm scale into cross_attn.kv_proj also, but algorithm doesn't support one-to-many yet
                **{k: k.replace('ln2', 'cross_attn.q_proj') for k, *_ in dec_keys if re.search(is_ln2, k)},
                **{k: k.replace('ln3', 'ffn.ff_in') for k, *_ in dec_keys if re.search(is_ln3, k)},
            }
            ln_wants_lin: dict[str, str] = {
                **enc_ln_wants_lin,
                **dec_ln_wants_lin,
            }
            lin_wants_ln: dict[str, str] = {v: k for k, v in ln_wants_lin.items()}
            wants_norm_fusion: set[str] = ln_wants_lin.keys() | lin_wants_ln.keys()
        else:
            wants_norm_fusion: set[str] = {}

        pending_fusion: dict[str, AcceptFusion] = {}

        def fuse_me(
            w_name: str,
            w_mod: Linear,
            w_attr: str,
            w: FloatTensor,
            s_name: str,
            ln_scale: FloatTensor,
        ) -> None:
            fuse_norm_scale(
                w=w,
                ln_scale=ln_scale,
                scale_via_f32=norm_fusion_via_f32,
            )
            w_mod.register_parameter(w_attr, w)
            wants_norm_fusion.remove(s_name)
            wants_norm_fusion.remove(w_name)

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
                
                # make layer outputs smaller in proportion to how much smaller we made their corresponding residual
                out_scale: Optional[float] = None
                if receives_enc_residual(obj_path):
                    if match := re.search(is_o_proj, name):
                        layer_idx: int = int(match.group(1))
                        out_scale: float = enc_scales.attn_out_scales_cp_hat[layer_idx]
                    elif match := re.search(is_ff_out, name):
                        layer_idx: int = int(match.group(1))
                        out_scale: float = enc_scales.ffn_out_scales_cp_hat[layer_idx]
                if receives_dec_residual(obj_path):
                    if match := re.search(is_self_o_proj, name):
                        layer_idx: int = int(match.group(1))
                        out_scale: float = dec_scales.self_attn_out_scales_cp_hat[layer_idx]
                    elif match := re.search(is_cross_o_proj, name):
                        layer_idx: int = int(match.group(1))
                        out_scale: float = dec_scales.cross_attn_out_scales_cp_hat[layer_idx]
                    elif match := re.search(is_ff_out, name):
                        layer_idx: int = int(match.group(1))
                        out_scale: float = dec_scales.ffn_out_scales_cp_hat[layer_idx]

                if entry.type is param_type:
                    if name in wants_norm_fusion:
                        if re.search(is_ln1or2or3, name):
                            counterpart_dict: dict[str, str] = ln_wants_lin
                            fuse_kwargs = { 's_name': name, 'ln_scale': tensor }
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
                        if out_scale is not None and out_scale != 1:
                            tensor.mul_(out_scale)
                        module.register_parameter(attr, tensor)
                elif entry.type is buffer_type:
                    module.register_buffer(attr, tensor)

        self._file.close()
        assert not wants_norm_fusion, f"Unfused: {wants_norm_fusion}"
        return tensor_ct

    
    
