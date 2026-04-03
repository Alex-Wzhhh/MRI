from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import ensure_dir


def write_results(rows: list[dict], output_dir: str) -> tuple[str, str]:
    out_dir = ensure_dir(output_dir)
    csv_path = out_dir / "results.csv"
    md_path = out_dir / "results.md"
    df = pd.DataFrame(rows)
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(
            subset=[
                "encoder_version",
                "layer",
                "roi_type",
                "pool_type",
                "sequence_mode",
                "probe_type",
                "split_version",
                "seed",
                "split_name",
            ],
            keep="last",
        )
    df.to_csv(csv_path, index=False)
    with Path(md_path).open("w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))
        f.write("\n")
    return str(csv_path), str(md_path)
