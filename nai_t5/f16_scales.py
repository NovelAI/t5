from typing import Optional
from .checkpoint_names import Checkpoint

enc_ffn_out_scale_dict: dict[Checkpoint, Optional[list[float]]] = {
    # 8 layers
    Checkpoint.T5v1_1Small: [*[1]*6, 1/2, 1/2],
    # 24 layers
    Checkpoint.T5v1_1XL: [*[1]*5, 1/8, *[1]*18],
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