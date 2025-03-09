from enum import Enum
from typing import Optional

class Checkpoint(str, Enum):
    T5v1_1Small = 't5-v1.1-small'
    T5v1_1XL = 't5-v1.1-xl'
    T5v1_1XXL = 't5-v1.1-xxl'
    T5v1Large = 't5-v1-large'
    PileT5Large = 'pile-t5-large'

enc_ffn_out_scale_dict: dict[Checkpoint, Optional[list[float]]] = {
    # 8 layers
    Checkpoint.T5v1_1Small: [*[1]*2, 1/2, *[1]*4, 1/2],
    # 24 layers
    Checkpoint.T5v1_1XL: [1, 1/2, *[1]*3, 1/4, *[1]*18],
    # 24 layers
    Checkpoint.T5v1_1XXL: [*[1]*7, 1/4, *[1]*16],
}

enc_attn_out_scale_dict: dict[Checkpoint, Optional[list[float]]] = {
    Checkpoint.T5v1_1Small: None,
    Checkpoint.T5v1_1XL: None,
    Checkpoint.T5v1_1XXL: None,
}

dec_self_attn_out_scale_dict: dict[Checkpoint, Optional[list[float]]] = {
    # 8 layers
    Checkpoint.T5v1_1Small: None,
    # 24 layers
    Checkpoint.T5v1_1XL: None,
    # 24 layers
    Checkpoint.T5v1_1XXL: None,
}

dec_cross_attn_out_scale_dict: dict[Checkpoint, Optional[list[float]]] = {
    # 8 layers
    Checkpoint.T5v1_1Small: None,
    # 24 layers
    Checkpoint.T5v1_1XL: None,
    # 24 layers
    Checkpoint.T5v1_1XXL: None,
}

dec_ffn_out_scale_dict: dict[Checkpoint, Optional[list[float]]] = {
    # 8 layers
    Checkpoint.T5v1_1Small: [*[1]*2, 1/2, *[1]*5],
    # 24 layers
    Checkpoint.T5v1_1XL: None,
    # 24 layers
    Checkpoint.T5v1_1XXL: None,
}

ckpt_to_hf_model_name: dict[Checkpoint, str] = {
    Checkpoint.T5v1_1Small: 'google/t5-v1_1-small',
    Checkpoint.T5v1_1XL: 'google/t5-v1_1-xl',
    Checkpoint.T5v1_1XXL: 'google/t5-v1_1-xxl',
    Checkpoint.T5v1Large: 'google-t5/t5-large',
    Checkpoint.PileT5Large: 'EleutherAI/pile-t5-large',
}

ckpt_is_umt5: dict[Checkpoint, str] = {
    Checkpoint.T5v1_1Small: False,
    Checkpoint.T5v1_1XL: False,
    Checkpoint.T5v1_1XXL: False,
    Checkpoint.T5v1Large: False,
    Checkpoint.PileT5Large: True,
}