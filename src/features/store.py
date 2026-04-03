from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir


class FeatureStore:
    def __init__(self, feature_root: str, index_path: str, extract_version: str):
        self.feature_root = Path(feature_root)
        self.index_path = Path(index_path)
        self.extract_version = extract_version
        ensure_dir(self.feature_root)
        ensure_dir(self.index_path.parent)

    def write_vector(
        self,
        case_id: str,
        sequence_id: str,
        layer: str,
        roi_type: str,
        pool_type: str,
        vector: np.ndarray,
    ) -> dict:
        case_dir = ensure_dir(self.feature_root / case_id / sequence_id)
        out_path = case_dir / f"{layer}_{roi_type}_{pool_type}.npy"
        np.save(out_path, vector.astype(np.float32))
        return {
            "case_id": case_id,
            "sequence_id": sequence_id,
            "layer": layer,
            "roi_type": roi_type,
            "pool_type": pool_type,
            "feature_path": str(out_path),
            "feature_dim": int(vector.shape[0]),
            "extract_version": self.extract_version,
        }

    def write_index(self, rows: list[dict]) -> None:
        df = pd.DataFrame(rows)
        if self.index_path.exists():
            existing = pd.read_csv(self.index_path)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(
                subset=["case_id", "sequence_id", "layer", "roi_type", "pool_type"],
                keep="last",
            )
        df.to_csv(self.index_path, index=False)
