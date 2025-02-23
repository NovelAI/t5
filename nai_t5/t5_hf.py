from typing import TYPE_CHECKING, Dict, Any

import torch
from torch.nn import Module, GELU
from torch import Tensor, cat, inference_mode

from .t5_common import T5Config, T5FFNType, GELUApprox, RMSNormCast

####
#### Construction of T5 config from HF config
####

if TYPE_CHECKING:
    from transformers.models.t5.configuration_t5 import T5Config as T5ConfigHF
else:
    T5ConfigHF = Any


def to_based_config(hf_config: T5ConfigHF, n_tokens=512, device=torch.device("cpu")) -> T5Config:
    return T5Config(
        vocab_size=hf_config.vocab_size,
        hidden_dim=hf_config.d_model,
        num_layers=hf_config.num_layers,
        n_head=hf_config.num_heads,
        kv_heads=hf_config.num_heads,
        head_dim=hf_config.d_kv,
        ff_dim=hf_config.d_ff,
        dropout=hf_config.dropout_rate,
        eps=hf_config.layer_norm_epsilon,
        ffn_type=T5FFNType.ReLU if hf_config.dense_act_fn == "relu" else T5FFNType.GEGLU,
        gelu_approx=GELUApprox.Tanh,
        relative_attention_max_distance=hf_config.relative_attention_max_distance,
        relative_attention_num_buckets=hf_config.relative_attention_num_buckets,
        # scale_qk disabled because importing Google weights
        scale_qk=False,
        pad_token_id=hf_config.pad_token_id,
        decoder_start_token_id=hf_config.pad_token_id,
        label_ignore_index=-100,
        n_tokens=n_tokens,
        device=device,
        pos_emb_per_layer=hf_config.model_type == 'umt5',
    )


####
#### Loading of T5 weights from HF model
####


def hf_to_based_t5_enc_state(hf_state: Dict[str, Tensor], config: T5Config) -> Dict[str, Tensor]:
    match config.ffn_type:
        case T5FFNType.GEGLU:
            ffn_state: Dict[str, Tensor] = {
                **{
                    f"layers.{ix}.ffn.ff_in.weight": cat(
                        [
                            hf_state[f"encoder.block.{ix}.layer.1.DenseReluDense.wi_0.weight"],
                            hf_state[f"encoder.block.{ix}.layer.1.DenseReluDense.wi_1.weight"],
                        ],
                        dim=-2,
                    )
                    for ix in range(config.num_layers)
                },
                **{
                    f"layers.{ix}.ffn.ff_out.weight": hf_state[f"encoder.block.{ix}.layer.1.DenseReluDense.wo.weight"]
                    for ix in range(config.num_layers)
                },
            }
        case T5FFNType.ReLU:
            ffn_state: Dict[str, Tensor] = {
                **{
                    f"layers.{ix}.ffn.ff_in.weight": hf_state[f"encoder.block.{ix}.layer.1.DenseReluDense.wi.weight"]
                    for ix in range(config.num_layers)
                },
                **{
                    f"layers.{ix}.ffn.ff_out.weight": hf_state[f"encoder.block.{ix}.layer.1.DenseReluDense.wo.weight"]
                    for ix in range(config.num_layers)
                },
            }
        case _:
            raise NotImplementedError(f"FFN type {config['ffn_type']} not implemented.")

    based_state: Dict[str, Tensor] = {
        "vocab_embed.weight": hf_state["shared.weight"],
        **{
            f"layers.{ix}.attn.qkv_proj.weight": cat(
                [
                    hf_state[f"encoder.block.{ix}.layer.0.SelfAttention.q.weight"],
                    hf_state[f"encoder.block.{ix}.layer.0.SelfAttention.k.weight"],
                    hf_state[f"encoder.block.{ix}.layer.0.SelfAttention.v.weight"],
                ],
                dim=-2,
            )
            for ix in range(config.num_layers)
        },
        f"relative_attention_bias.bias_emb.weight": (
            cat(
                [
                    hf_state[f"encoder.block.{ix}.layer.0.SelfAttention.relative_attention_bias.weight"]
                    for ix in range(config.num_layers)
                ],
                dim=-1,
            ) if config.pos_emb_per_layer else hf_state[
                f"encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
            ]
        ),
        **{
            f"layers.{ix}.attn.o_proj.weight": hf_state[f"encoder.block.{ix}.layer.0.SelfAttention.o.weight"]
            for ix in range(config.num_layers)
        },
        **{
            f"layers.{ix}.ln1.weight": hf_state[f"encoder.block.{ix}.layer.0.layer_norm.weight"]
            for ix in range(config.num_layers)
        },
        **{
            f"layers.{ix}.ln2.weight": hf_state[f"encoder.block.{ix}.layer.1.layer_norm.weight"]
            for ix in range(config.num_layers)
        },
        **ffn_state,
        "ln.weight": hf_state["encoder.final_layer_norm.weight"],
    }
    return based_state


def hf_to_based_t5_dec_state(hf_state: Dict[str, Tensor], config: T5Config) -> Dict[str, Tensor]:
    match config.ffn_type:
        case T5FFNType.GEGLU:
            ffn_state: Dict[str, Tensor] = {
                **{
                    f"layers.{ix}.ffn.ff_in.weight": cat(
                        [
                            hf_state[f"decoder.block.{ix}.layer.2.DenseReluDense.wi_0.weight"],
                            hf_state[f"decoder.block.{ix}.layer.2.DenseReluDense.wi_1.weight"],
                        ],
                        dim=-2,
                    )
                    for ix in range(config.num_layers)
                },
                **{
                    f"layers.{ix}.ffn.ff_out.weight": hf_state[f"decoder.block.{ix}.layer.2.DenseReluDense.wo.weight"]
                    for ix in range(config.num_layers)
                },
            }
        case T5FFNType.ReLU:
            ffn_state: Dict[str, Tensor] = {
                **{
                    f"layers.{ix}.ffn.ff_in.weight": hf_state[f"decoder.block.{ix}.layer.2.DenseReluDense.wi.weight"]
                    for ix in range(config.num_layers)
                },
                **{
                    f"layers.{ix}.ffn.ff_out.weight": hf_state[f"decoder.block.{ix}.layer.2.DenseReluDense.wo.weight"]
                    for ix in range(config.num_layers)
                },
            }
        case _:
            raise NotImplementedError(f"FFN type {config['ffn_type']} not implemented.")

    based_state: Dict[str, Tensor] = {
        # "vocab_embed.weight": hf_state["shared.weight"],
        **{
            f"layers.{ix}.self_attn.qkv_proj.weight": cat(
                [
                    hf_state[f"decoder.block.{ix}.layer.0.SelfAttention.q.weight"],
                    hf_state[f"decoder.block.{ix}.layer.0.SelfAttention.k.weight"],
                    hf_state[f"decoder.block.{ix}.layer.0.SelfAttention.v.weight"],
                ],
                dim=-2,
            )
            for ix in range(config.num_layers)
        },
        f"relative_attention_bias.bias_emb.weight": (
            cat(
                [
                    hf_state[f"decoder.block.{ix}.layer.0.SelfAttention.relative_attention_bias.weight"]
                    for ix in range(config.num_layers)
                ],
                dim=-1,
            ) if config.pos_emb_per_layer else hf_state[
                f"decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
            ]
        ),
        **{
            f"layers.{ix}.self_attn.o_proj.weight": hf_state[f"decoder.block.{ix}.layer.0.SelfAttention.o.weight"]
            for ix in range(config.num_layers)
        },
        **{
            f"layers.{ix}.ln1.weight": hf_state[f"decoder.block.{ix}.layer.0.layer_norm.weight"]
            for ix in range(config.num_layers)
        },
        **{
            f"layers.{ix}.ln2.weight": hf_state[f"decoder.block.{ix}.layer.1.layer_norm.weight"]
            for ix in range(config.num_layers)
        },
        **{
            f"layers.{ix}.cross_attn.q_proj.weight": hf_state[f"decoder.block.{ix}.layer.1.EncDecAttention.q.weight"]
            for ix in range(config.num_layers)
        },
        **{
            f"layers.{ix}.cross_attn.kv_proj.weight": cat(
                [
                    hf_state[f"decoder.block.{ix}.layer.1.EncDecAttention.k.weight"],
                    hf_state[f"decoder.block.{ix}.layer.1.EncDecAttention.v.weight"],
                ],
                dim=-2,
            )
            for ix in range(config.num_layers)
        },
        **{
            f"layers.{ix}.cross_attn.o_proj.weight": hf_state[f"decoder.block.{ix}.layer.1.EncDecAttention.o.weight"]
            for ix in range(config.num_layers)
        },
        **{
            f"layers.{ix}.ln3.weight": hf_state[f"decoder.block.{ix}.layer.2.layer_norm.weight"]
            for ix in range(config.num_layers)
        },
        **ffn_state,
        "ln.weight": hf_state["decoder.final_layer_norm.weight"],
    }
    return based_state


def hf_to_based_t5_state(hf_state: Dict[str, Tensor], config: T5Config) -> Dict[str, Tensor]:
    based_state: Dict[str, Tensor] = {
        **{f"encoder.{k}": v for k, v in hf_to_based_t5_enc_state(hf_state, config).items()},
        **{f"decoder.{k}": v for k, v in hf_to_based_t5_dec_state(hf_state, config).items()},
        "lm_head.weight": hf_state["lm_head.weight"],
    }
    return based_state

def replace_norms(mod: Module) -> None:
    """
    Make an HF T5 or UMT5 model use the same RMSNorm as us, to remove it as
    a confounding factor in parity-testing
    """
    from transformers.models.t5.modeling_t5 import T5LayerNorm
    from transformers.models.umt5.modeling_umt5 import UMT5LayerNorm
    from torch.nn import RMSNorm
    for child_name, child_mod in mod.named_children():
        # print(child_name, child_mod.__class__.__name__)
        match child_mod:
            case T5LayerNorm() | UMT5LayerNorm():
                if child_mod.__class__.__module__.startswith('apex'):
                    normalized_shape = child_mod.normalized_shape
                    eps = child_mod.eps
                    elementwise_affine = child_mod.elementwise_affine
                else:
                    normalized_shape = child_mod.weight.size(-1)
                    eps = child_mod.variance_epsilon
                    elementwise_affine = True
                # norm = RMSNormCast(
                #     normalized_shape,
                #     eps=eps,
                #     elementwise_affine=elementwise_affine,
                #     device=child_mod.weight.device,
                #     dtype=child_mod.weight.dtype,
                # )
                # actually let's replace it with the superclass we delegate to
                norm = RMSNorm(
                    normalized_shape,
                    eps=eps,
                    elementwise_affine=elementwise_affine,
                    device=child_mod.weight.device,
                    dtype=child_mod.weight.dtype,
                )
                with inference_mode():
                    norm.weight.copy_(child_mod.weight)
                setattr(mod, child_name, norm)
            case _:
                replace_norms(child_mod)

def replace_gates(mod: Module) -> None:
    """
    Make an HF T5 model use the same GELU as us, to remove it as
    a confounding factor in parity-testing
    """
    from transformers.activations import NewGELUActivation
    for child_name, child_mod in mod.named_children():
        # print(child_name, child_mod.__class__.__name__)
        if isinstance(child_mod, NewGELUActivation):
            gelu = GELU(approximate='tanh')
            setattr(mod, child_name, gelu)
        else:
            replace_gates(child_mod)