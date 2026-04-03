from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import classification_metrics
from src.eval.report import write_results
from src.probes.logistic_probe import LogisticProbe
from src.probes.mlp_probe import MLPProbe
from src.probes.svm_probe import LinearSVMProbe
from src.utils.config import load_yaml
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

LOGGER = get_logger("train_probe")


def build_probe(cfg: dict):
    probe_type = cfg["probe_type"]
    kwargs = {k: v for k, v in cfg.items() if k != "probe_type" and k != "dropout"}
    if probe_type == "logistic":
        return LogisticProbe(**kwargs)
    if probe_type == "svm":
        return LinearSVMProbe(**kwargs)
    if probe_type == "mlp":
        if "hidden_layer_sizes" in kwargs and isinstance(kwargs["hidden_layer_sizes"], list):
            kwargs["hidden_layer_sizes"] = tuple(kwargs["hidden_layer_sizes"])
        return MLPProbe(**kwargs)
    raise ValueError(f"Unsupported probe type: {probe_type}")


def load_case_feature_map(index_path: str, layer: str, roi_type: str, pool_type: str) -> dict[str, dict[str, np.ndarray]]:
    index_df = pd.read_csv(index_path)
    subset = index_df[
        (index_df["layer"] == layer)
        & (index_df["roi_type"] == roi_type)
        & (index_df["pool_type"] == pool_type)
    ]
    feature_map: dict[str, dict[str, np.ndarray]] = {}
    for row in subset.itertuples(index=False):
        feature_map.setdefault(row.case_id, {})[row.sequence_id] = np.load(row.feature_path)
    return feature_map


def aggregate_case_vector(sequence_vectors: dict[str, np.ndarray], mode: str, sequence_id: str | None) -> np.ndarray:
    if not sequence_vectors:
        raise ValueError("No sequence vectors provided.")
    ordered_keys = sorted(sequence_vectors.keys())
    if mode == "single_sequence":
        if sequence_id is None:
            sequence_id = ordered_keys[0]
        return sequence_vectors[sequence_id]
    vectors = [sequence_vectors[k] for k in ordered_keys]
    if mode == "mean_over_sequences":
        return np.stack(vectors, axis=0).mean(axis=0)
    if mode == "concat_sequences":
        return np.concatenate(vectors, axis=0)
    raise ValueError(f"Unsupported sequence_mode: {mode}")


def build_split_arrays(split_cases: list[str], labels: dict[str, int], feature_map: dict[str, dict[str, np.ndarray]], mode: str, sequence_id: str | None):
    X, y, used_cases = [], [], []
    for case_id in split_cases:
        if case_id not in labels or case_id not in feature_map:
            continue
        try:
            vector = aggregate_case_vector(feature_map[case_id], mode, sequence_id)
        except Exception:
            continue
        X.append(vector)
        y.append(labels[case_id])
        used_cases.append(case_id)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), used_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-config", required=True)
    parser.add_argument(
        "--feature-index",
        default="/home/alex/Project/MRI/Data/features/stage1/feature_index.csv",
    )
    parser.add_argument(
        "--manifest",
        default="/home/alex/Project/MRI/Data/manifests/binary_ready.csv",
    )
    parser.add_argument(
        "--split-file",
        default="/home/alex/Project/MRI/Data/splits/split_v1.json",
    )
    parser.add_argument("--layer", default="l2")
    parser.add_argument("--roi-type", default="tumor")
    parser.add_argument("--pool-type", default="avg_pool_masked")
    parser.add_argument("--sequence-mode", default="mean_over_sequences")
    parser.add_argument("--sequence-id", default="")
    args = parser.parse_args()

    probe_cfg = load_yaml(args.probe_config)
    seed_everything(int(probe_cfg.get("random_state", 42)))
    feature_map = load_case_feature_map(
        args.feature_index, args.layer, args.roi_type, args.pool_type
    )
    manifest_df = pd.read_csv(args.manifest)
    labels = {
        row.case_id: int(row.mvi_binary)
        for row in manifest_df.itertuples(index=False)
        if not pd.isna(row.mvi_binary)
    }
    with Path(args.split_file).open("r", encoding="utf-8") as f:
        split_data = json.load(f)

    X_train, y_train, train_cases = build_split_arrays(
        split_data["train"],
        labels,
        feature_map,
        args.sequence_mode,
        args.sequence_id or None,
    )
    X_val, y_val, val_cases = build_split_arrays(
        split_data["val"],
        labels,
        feature_map,
        args.sequence_mode,
        args.sequence_id or None,
    )
    X_test, y_test, test_cases = build_split_arrays(
        split_data["test"],
        labels,
        feature_map,
        args.sequence_mode,
        args.sequence_id or None,
    )

    probe = build_probe(probe_cfg).fit(X_train, y_train)
    rows = []
    for split_name, X, y, used_cases in [
        ("train", X_train, y_train, train_cases),
        ("val", X_val, y_val, val_cases),
        ("test", X_test, y_test, test_cases),
    ]:
        y_pred = probe.predict(X)
        y_score = probe.score_samples(X)
        metrics = classification_metrics(y, y_pred, y_score)
        row = {
            "encoder_version": "medsam2_stage1_v1",
            "layer": args.layer,
            "roi_type": args.roi_type,
            "pool_type": args.pool_type,
            "sequence_mode": args.sequence_mode,
            "probe_type": probe_cfg["probe_type"],
            "split_version": Path(args.split_file).stem,
            "seed": int(probe_cfg.get("random_state", 42)),
            "split_name": split_name,
            "n_cases": len(used_cases),
        }
        row.update(metrics)
        rows.append(row)
    csv_path, md_path = write_results(rows, "/home/alex/Project/MRI/outputs/reports")
    LOGGER.info("results.csv: %s", csv_path)
    LOGGER.info("results.md: %s", md_path)


if __name__ == "__main__":
    main()
