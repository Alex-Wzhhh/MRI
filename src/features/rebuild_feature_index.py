from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml
from src.utils.io import ensure_dir
from src.utils.logger import get_logger

LOGGER = get_logger("rebuild_feature_index")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-config",
        default="/home/alex/Project/MRI/configs/feature/feature_store.yaml",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.feature_config)
    feature_root = Path(cfg["feature_root"])
    index_path = Path(cfg["index_path"])
    pool_type = str(cfg["pool_type"])
    suffix = f"_{pool_type}.npy"

    rows: list[dict] = []
    for path in sorted(feature_root.glob("case*/seq_*/*.npy")):
        if not path.name.endswith(suffix):
            continue
        stem = path.name[: -len(suffix)]
        if "_" not in stem:
            continue
        layer, roi_type = stem.split("_", 1)
        vector = np.load(path)
        rows.append(
            {
                "case_id": path.parent.parent.name,
                "sequence_id": path.parent.name,
                "layer": layer,
                "roi_type": roi_type,
                "pool_type": pool_type,
                "feature_path": str(path),
                "feature_dim": int(vector.shape[0]),
                "extract_version": cfg["extract_version"],
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(
            subset=["case_id", "sequence_id", "layer", "roi_type", "pool_type"],
            keep="last",
        ).sort_values(["case_id", "sequence_id", "layer", "roi_type"])
    ensure_dir(index_path.parent)
    df.to_csv(index_path, index=False)
    LOGGER.info("feature rows: %d", len(df))
    LOGGER.info("feature index: %s", index_path)


if __name__ == "__main__":
    main()
