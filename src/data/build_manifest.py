from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml
from src.utils.io import ensure_dir
from src.utils.logger import get_logger

LOGGER = get_logger("build_manifest")

CHINESE_TO_ENGLISH = {
    "年龄": "age",
    "性别": "sex",
    "乙肝病毒携带": "hbv_status",
    "肝硬化": "cirrhosis",
    "甲胎蛋白": "afp",
    "谷丙转氨酶ALT": "alt",
    "谷草转氨酶AST": "ast",
    "血红蛋白": "hemoglobin",
    "血小板": "platelet",
    "分化程度": "differentiation",
    "组织学分型": "histology_type",
}


def _build_sequence_paths(image_dir: str, suffixes: list[str]) -> list[str]:
    return [str(Path(image_dir) / f"case_placeholder_{suffix}.nii.gz") for suffix in suffixes]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/alex/Project/MRI/configs/data/center1.yaml",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    manifest_df = pd.read_csv(cfg["manifest_path"])
    labels_df = pd.read_csv(cfg["labels_path"])

    labels_df = labels_df.rename(columns={"case": "case_id"})
    manifest_df = manifest_df.rename(columns={"case": "case_id"})

    labels_cols = ["case_id"] + [
        col
        for col in [
            "年龄",
            "性别",
            "乙肝病毒携带",
            "肝硬化",
            "甲胎蛋白",
            "谷丙转氨酶ALT",
            "谷草转氨酶AST",
            "血红蛋白",
            "血小板",
            "分化程度",
            "组织学分型",
            "MVI",
            "mvi_label_clean",
            "mvi_status",
            "is_binary_ready",
        ]
        if col in labels_df.columns
    ]
    manifest_cols = [
        "case_id",
        "has_image",
        "has_seg",
        "has_csv",
        "raw_mvi",
        "split_group",
        "missing_reasons",
        "image_dir",
        "seg_path",
    ]

    merged = labels_df[labels_cols].merge(
        manifest_df[manifest_cols],
        on="case_id",
        how="outer",
    )

    merged["center_id"] = cfg["center_id"]
    merged["mvi_binary"] = merged["mvi_label_clean"].where(
        merged["mvi_status"].isin(["binary_negative", "binary_positive"])
    )
    merged["mvi_trinary"] = merged["mvi_label_clean"].where(
        merged["mvi_status"].isin(
            ["binary_negative", "binary_positive", "m2_excluded_from_binary"]
        )
    )
    merged["is_binary_ready"] = (
        merged["has_image"].fillna(0).astype(int).eq(1)
        & merged["has_seg"].fillna(0).astype(int).eq(1)
        & merged["mvi_status"].isin(["binary_negative", "binary_positive"])
    )
    merged["is_seg_ready"] = (
        merged["has_image"].fillna(0).astype(int).eq(1)
        & merged["has_seg"].fillna(0).astype(int).eq(1)
    )

    for cn, en in CHINESE_TO_ENGLISH.items():
        if cn in merged.columns:
            merged[en] = merged[cn]

    suffixes = cfg["image_suffixes"]

    def seq_paths(row: pd.Series) -> str:
        image_dir = row.get("image_dir")
        case_id = row["case_id"]
        if not isinstance(image_dir, str) or not image_dir:
            return json.dumps([])
        return json.dumps(
            [str(Path(image_dir) / f"{case_id}_{suffix}.nii.gz") for suffix in suffixes],
            ensure_ascii=False,
        )

    merged["sequence_paths_json"] = merged.apply(seq_paths, axis=1)
    merged["sequence_names_json"] = json.dumps(cfg["sequence_names"], ensure_ascii=False)

    ordered_cols = [
        "case_id",
        "center_id",
        "image_dir",
        "seg_path",
        "sequence_paths_json",
        "sequence_names_json",
        "has_image",
        "has_seg",
        "has_csv",
        "raw_mvi",
        "mvi_binary",
        "mvi_trinary",
        "mvi_status",
        "is_binary_ready",
        "is_seg_ready",
        "missing_reasons",
        "age",
        "sex",
        "hbv_status",
        "cirrhosis",
        "afp",
        "alt",
        "ast",
        "hemoglobin",
        "platelet",
        "differentiation",
        "histology_type",
    ]
    for col in ordered_cols:
        if col not in merged.columns:
            merged[col] = ""
    canonical = merged[ordered_cols].sort_values("case_id").reset_index(drop=True)

    binary_ready = canonical[canonical["is_binary_ready"]].copy()
    excluded = canonical[~canonical["is_binary_ready"]].copy()

    if "missing_reasons" not in excluded.columns:
        excluded["missing_reasons"] = ""
    excluded["exclude_reason"] = excluded["missing_reasons"].replace("", "not_binary_ready")

    for out_path in [
        cfg["canonical_manifest_path"],
        cfg["binary_ready_path"],
        cfg["excluded_cases_path"],
    ]:
        ensure_dir(Path(out_path).parent)

    canonical.to_csv(cfg["canonical_manifest_path"], index=False)
    binary_ready.to_csv(cfg["binary_ready_path"], index=False)
    excluded.to_csv(cfg["excluded_cases_path"], index=False)

    LOGGER.info("canonical_manifest: %s", cfg["canonical_manifest_path"])
    LOGGER.info("binary_ready: %d cases", len(binary_ready))
    LOGGER.info("excluded_cases: %d cases", len(excluded))


if __name__ == "__main__":
    main()
