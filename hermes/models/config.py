from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    n_embed: int
    n_heads: int
    n_kv_heads: int
    n_layer: int
    n_mlp: int

    n_vocab: int
    tie_word_embeddings: bool

    rope_theta: float
    rms_norm_eps: float

    d_head: Optional[int] = None
    n_experts: Optional[int] = None
    n_experts_per_token: Optional[int] = None
    n_moe_mlp: Optional[int] = None


@dataclass
class VisionConfig:
    n_embed: int
    n_layer: int
    n_heads: int
    n_output_embed: int
    n_mlp: int
    deepstack_visual_indexes: list[int]
    num_position_embeddings: int

    in_channels: int = 3
    temporal_patch_size: int = 2
    patch_size: int = 16
    spatial_merge_size: int = 2
