from __future__ import annotations


def canonical_layer_map(backbone_features: list) -> dict[str, int]:
    if len(backbone_features) >= 3:
        return {"l1": 0, "l2": 1, "l3": 2}
    if len(backbone_features) == 2:
        return {"l1": 0, "l2": 1}
    return {"l1": 0}
