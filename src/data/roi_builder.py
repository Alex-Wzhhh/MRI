from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml
from src.utils.io import ensure_dir
from src.utils.logger import get_logger

LOGGER = get_logger("roi_builder")


def compute_bbox(mask: np.ndarray) -> list[int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return [-1, -1, -1, -1, -1, -1]
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return [
        int(mins[0]),
        int(maxs[0]),
        int(mins[1]),
        int(maxs[1]),
        int(mins[2]),
        int(maxs[2]),
    ]


def build_peritumor(mask: np.ndarray, spacing: tuple[float, float, float], mm: float) -> np.ndarray:
    dist = distance_transform_edt(~mask.astype(bool), sampling=spacing)
    dilated = dist <= mm
    return np.logical_and(dilated, ~mask.astype(bool))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="/home/alex/Project/MRI/Data/manifests/binary_ready.csv",
    )
    parser.add_argument(
        "--roi-config",
        default="/home/alex/Project/MRI/configs/data/roi.yaml",
    )
    parser.add_argument(
        "--output",
        default="/home/alex/Project/MRI/Data/processed/roi_metadata.jsonl",
    )
    parser.add_argument(
        "--mask-dir",
        default="/home/alex/Project/MRI/Data/processed/roi_masks",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.roi_config)
    df = pd.read_csv(args.manifest)
    out_path = Path(args.output)
    ensure_dir(out_path.parent)
    mask_dir = ensure_dir(args.mask_dir)

    with out_path.open("w", encoding="utf-8") as f:
        for row in df.itertuples(index=False):
            seg_path = Path(row.seg_path)
            if not seg_path.exists():
                LOGGER.warning("Missing seg path: %s", seg_path)
                continue
            seg_img = nib.load(str(seg_path))
            seg = np.asarray(seg_img.get_fdata() > 0, dtype=bool)
            spacing = tuple(float(x) for x in seg_img.header.get_zooms()[:3])
            if int(seg.sum()) < int(cfg["min_voxels"]):
                LOGGER.warning("Skip tiny mask: %s", row.case_id)
                continue

            tumor_bbox = compute_bbox(seg)
            record = {
                "case_id": row.case_id,
                "seg_path": str(seg_path),
                "mask_shape": list(seg.shape),
                "spacing": list(spacing),
                "roi_mask_path": str(mask_dir / f"{row.case_id}_roi_masks.npz"),
                "roi": {
                    "tumor": {
                        "bbox": tumor_bbox,
                        "voxel_count": int(seg.sum()),
                    }
                },
            }
            roi_masks = {"tumor": seg.astype(np.uint8)}
            for roi_name, mm in cfg["peritumor_mm"].items():
                ring = build_peritumor(seg, spacing, float(mm))
                roi_masks[roi_name] = ring.astype(np.uint8)
                record["roi"][roi_name] = {
                    "bbox": compute_bbox(ring),
                    "voxel_count": int(ring.sum()),
                }
            np.savez_compressed(record["roi_mask_path"], **roi_masks)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    LOGGER.info("roi metadata: %s", out_path)


if __name__ == "__main__":
    main()
