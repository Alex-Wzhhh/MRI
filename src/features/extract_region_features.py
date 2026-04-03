from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.encoders.medsam2_encoder import MedSAM2Encoder
from src.features.pooling import POOLERS
from src.features.store import FeatureStore
from src.utils.config import load_yaml
from src.utils.logger import get_logger

LOGGER = get_logger("extract_region_features")


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    arr = slice_2d.astype(np.float32)
    mean = float(arr.mean())
    std = float(arr.std())
    if std < 1e-6:
        std = 1.0
    arr = (arr - mean) / std
    arr = np.clip(arr, -3.0, 3.0)
    arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-6)
    return arr


def to_model_input(slice_2d: np.ndarray, input_size: int) -> torch.Tensor:
    x = torch.from_numpy(normalize_slice(slice_2d)).float()[None, None, :, :]
    x = F.interpolate(x, size=(input_size, input_size), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    return x


def resize_mask(mask_2d: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(mask_2d.astype(np.float32))[None, None, :, :]
    tensor = F.interpolate(tensor, size=shape_hw, mode="nearest")
    return tensor[0, 0].numpy() > 0.5


def load_roi_metadata(path: str) -> dict[str, dict]:
    data = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            data[record["case_id"]] = record
    return data


def extract_case_vectors(
    encoder: MedSAM2Encoder,
    image_items: list[tuple[str, str]],
    seg_path: str,
    layers: list[str],
    roi_types: list[str],
    peritumor_meta: dict,
    pool_type: str,
    input_size: int,
    max_slices: int,
) -> dict[tuple[str, str], list[np.ndarray]]:
    seg_img = nib.load(seg_path)
    seg = np.asarray(seg_img.get_fdata() > 0, dtype=bool)
    roi_npz = np.load(peritumor_meta["roi_mask_path"])
    pool_fn = POOLERS[pool_type]
    layer_vectors: dict[tuple[str, str], list[np.ndarray]] = {}

    tumor_slices = np.where(seg.any(axis=(0, 1)))[0]
    if max_slices > 0 and len(tumor_slices) > max_slices:
        take = np.linspace(0, len(tumor_slices) - 1, num=max_slices, dtype=int)
        tumor_slices = tumor_slices[take]
    valid_slices = [int(z) for z in tumor_slices]

    for sequence_id, image_path in image_items:
        image_img = nib.load(image_path)
        volume = np.asarray(image_img.get_fdata(), dtype=np.float32)

        for z in valid_slices:
            if z >= volume.shape[2]:
                continue
            slice_img = volume[:, :, z]
            slice_mask_tumor = seg[:, :, z]
            if not slice_mask_tumor.any():
                continue
            model_input = to_model_input(slice_img, input_size)
            feature_maps = encoder.extract_intermediate(model_input, layers)
            for layer_name, feature_tensor in feature_maps.items():
                feat = feature_tensor[0].detach().cpu().numpy()
                for roi_name in roi_types:
                    roi_mask = roi_npz[roi_name][:, :, z].astype(bool)
                    resized_mask = resize_mask(roi_mask, feat.shape[1:])
                    vector = pool_fn(feat, resized_mask)
                    key = (sequence_id, layer_name, roi_name)
                    layer_vectors.setdefault(key, []).append(vector)

    return layer_vectors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="/home/alex/Project/MRI/Data/manifests/binary_ready.csv",
    )
    parser.add_argument(
        "--roi-metadata",
        default="/home/alex/Project/MRI/Data/processed/roi_metadata.jsonl",
    )
    parser.add_argument(
        "--encoder-config",
        default="/home/alex/Project/MRI/configs/encoder/medsam2.yaml",
    )
    parser.add_argument(
        "--feature-config",
        default="/home/alex/Project/MRI/configs/feature/feature_store.yaml",
    )
    parser.add_argument(
        "--roi-config",
        default="/home/alex/Project/MRI/configs/data/roi.yaml",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-slices", type=int, default=8)
    parser.add_argument("--layers", default="")
    parser.add_argument("--roi-types", default="")
    parser.add_argument("--sequence-ids", default="")
    parser.add_argument("--case-ids-file", default="")
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if args.case_ids_file:
        case_ids = {
            line.strip()
            for line in Path(args.case_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        df = df[df["case_id"].isin(case_ids)].copy()
    if args.offset > 0:
        df = df.iloc[args.offset :].copy()
    if args.limit > 0:
        df = df.head(args.limit).copy()
    encoder_cfg = load_yaml(args.encoder_config)
    feature_cfg = load_yaml(args.feature_config)
    roi_cfg = load_yaml(args.roi_config)

    roi_metadata = load_roi_metadata(args.roi_metadata)
    encoder = MedSAM2Encoder(args.encoder_config)
    store = FeatureStore(
        feature_root=feature_cfg["feature_root"],
        index_path=feature_cfg["index_path"],
        extract_version=feature_cfg["extract_version"],
    )

    layers = (
        [item.strip() for item in args.layers.split(",") if item.strip()]
        if args.layers
        else list(encoder_cfg["layers"].keys())
    )
    roi_types = (
        [item.strip() for item in args.roi_types.split(",") if item.strip()]
        if args.roi_types
        else roi_cfg["roi_types"]
    )
    selected_sequence_ids = (
        {item.strip() for item in args.sequence_ids.split(",") if item.strip()}
        if args.sequence_ids
        else None
    )
    rows = []

    for row in df.itertuples(index=False):
        case_meta = roi_metadata.get(row.case_id)
        if case_meta is None:
            LOGGER.warning("Missing ROI metadata for %s", row.case_id)
            continue
        sequence_paths = json.loads(row.sequence_paths_json)
        image_items = []
        for idx, path in enumerate(sequence_paths):
            seq_id = f"seq_{idx:04d}"
            if selected_sequence_ids is not None and seq_id not in selected_sequence_ids:
                continue
            if Path(path).exists():
                image_items.append((seq_id, path))
        vectors = extract_case_vectors(
            encoder=encoder,
            image_items=image_items,
            seg_path=row.seg_path,
            layers=layers,
            roi_types=roi_types,
            peritumor_meta=case_meta,
            pool_type=feature_cfg["pool_type"],
            input_size=int(encoder_cfg["input_size"]),
            max_slices=int(args.max_slices),
        )
        for (sequence_id, layer_name, roi_name), seq_vectors in vectors.items():
            vector = np.stack(seq_vectors, axis=0).mean(axis=0)
            rows.append(
                store.write_vector(
                    case_id=row.case_id,
                    sequence_id=sequence_id,
                    layer=layer_name,
                    roi_type=roi_name,
                    pool_type=feature_cfg["pool_type"],
                    vector=vector,
                )
            )

    if rows and not args.skip_index:
        store.write_index(rows)
        LOGGER.info("feature index: %s", feature_cfg["index_path"])
    LOGGER.info("processed cases: %d", len(df))
    LOGGER.info("written feature rows: %d", len(rows))


if __name__ == "__main__":
    main()
