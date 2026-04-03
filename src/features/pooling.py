from __future__ import annotations

import numpy as np


def avg_pool_masked(feature_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = feature_map[:, mask]
    if masked.size == 0:
        return np.zeros(feature_map.shape[0], dtype=np.float32)
    return masked.mean(axis=1).astype(np.float32)


def max_pool_masked(feature_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = feature_map[:, mask]
    if masked.size == 0:
        return np.zeros(feature_map.shape[0], dtype=np.float32)
    return masked.max(axis=1).astype(np.float32)


def bbox_pool(feature_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros(feature_map.shape[0], dtype=np.float32)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = feature_map[:, y0:y1, x0:x1]
    return cropped.reshape(feature_map.shape[0], -1).mean(axis=1).astype(np.float32)


POOLERS = {
    "avg_pool_masked": avg_pool_masked,
    "max_pool_masked": max_pool_masked,
    "bbox_pool": bbox_pool,
}
