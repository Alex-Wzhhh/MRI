from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml
from src.utils.io import save_json
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

LOGGER = get_logger("make_splits")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-config",
        default="/home/alex/Project/MRI/configs/data/center1.yaml",
    )
    parser.add_argument(
        "--split-config",
        default="/home/alex/Project/MRI/configs/data/splits.yaml",
    )
    args = parser.parse_args()

    data_cfg = load_yaml(args.data_config)
    split_cfg = load_yaml(args.split_config)
    seed_everything(int(split_cfg["seed"]))

    df = pd.read_csv(data_cfg["binary_ready_path"])
    y = df[split_cfg["stratify_column"]].astype(int)
    case_ids = df["case_id"].tolist()

    train_val_idx, test_idx = train_test_split(
        df.index,
        test_size=float(split_cfg["test_size"]),
        random_state=int(split_cfg["seed"]),
        stratify=y,
    )
    train_val_df = df.loc[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_df.index,
        test_size=float(split_cfg["val_size"]),
        random_state=int(split_cfg["seed"]),
        stratify=train_val_df[split_cfg["stratify_column"]].astype(int),
    )

    split_v1 = {
        "seed": int(split_cfg["seed"]),
        "train": df.loc[train_idx, "case_id"].tolist(),
        "val": df.loc[val_idx, "case_id"].tolist(),
        "test": df.loc[test_idx, "case_id"].tolist(),
    }
    split_path = "/home/alex/Project/MRI/Data/splits/split_v1.json"
    save_json(split_v1, split_path)

    skf = StratifiedKFold(
        n_splits=int(split_cfg["n_splits"]),
        shuffle=True,
        random_state=int(split_cfg["seed"]),
    )
    folds = []
    for fold_id, (train_index, test_index) in enumerate(skf.split(case_ids, y), start=1):
        folds.append(
            {
                "fold": fold_id,
                "train": df.iloc[train_index]["case_id"].tolist(),
                "test": df.iloc[test_index]["case_id"].tolist(),
            }
        )
    cv_path = "/home/alex/Project/MRI/Data/splits/cv5_v1.json"
    save_json({"seed": int(split_cfg["seed"]), "folds": folds}, cv_path)

    LOGGER.info("split_v1: %s", split_path)
    LOGGER.info("cv5_v1: %s", cv_path)


if __name__ == "__main__":
    main()
