from __future__ import annotations

import sys
from pathlib import Path

import torch

from src.encoders.base_encoder import BaseEncoder
from src.encoders.hooks import canonical_layer_map
from src.utils.config import load_yaml


class MedSAM2Encoder(BaseEncoder):
    def __init__(self, config_path: str):
        self.cfg = load_yaml(config_path)
        self.model = None
        self.device = torch.device(
            self.cfg.get("device", "cuda")
            if torch.cuda.is_available()
            else "cpu"
        )
        self.layer_aliases = self.cfg.get("layers", {"l1": 0, "l2": 1, "l3": 2})

    def load_weights(self) -> None:
        medsam_root = Path(self.cfg["medsam_root"])
        if str(medsam_root) not in sys.path:
            sys.path.insert(0, str(medsam_root))
        from sam2.build_sam import build_sam2

        self.model = build_sam2(
            config_file=self.cfg["config_name"],
            ckpt_path=self.cfg["checkpoint_path"],
            device=str(self.device),
            mode="eval",
        )
        self.model.eval()

    def encode_image(self, image_tensor: torch.Tensor):
        if self.model is None:
            self.load_weights()
        with torch.no_grad():
            return self.model.forward_image(image_tensor.to(self.device))

    def extract_intermediate(self, image_tensor: torch.Tensor, layers: list[str] | None = None):
        if self.model is None:
            self.load_weights()
        with torch.no_grad():
            backbone_out = self.model.forward_image(image_tensor.to(self.device))
        feature_maps = backbone_out["backbone_fpn"]
        alias_map = canonical_layer_map(feature_maps)
        alias_map.update(self.layer_aliases)
        target_layers = layers or list(alias_map.keys())
        out = {}
        for layer_name in target_layers:
            idx = alias_map[layer_name]
            if idx >= len(feature_maps):
                continue
            out[layer_name] = feature_maps[idx]
        return out

    def get_feature_spec(self) -> dict:
        return {
            "model_type": self.cfg["model_type"],
            "checkpoint_path": self.cfg["checkpoint_path"],
            "layers": self.layer_aliases,
            "input_size": int(self.cfg.get("input_size", 512)),
            "device": str(self.device),
        }
