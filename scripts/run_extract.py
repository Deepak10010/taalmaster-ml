"""
DVC STAGE 1 — Extract
=======================
Pulls raw data from Neon and writes each DataFrame to a STABLE path
under data/dvc/raw/. DVC hashes these files so the downstream stages
only re-run when something actually changed.

Why stable paths (no timestamps)?
  DVC tracks files by content hash. If the path is `data/raw/20260416_202324/...`,
  every run creates a new path and DVC can't compare to the previous run.
  Using `data/dvc/raw/` means:
    - Run A produces hash X → DVC caches it
    - Run B produces the same hash X → DVC skips downstream stages
    - Run B produces hash Y → downstream stages re-run
"""
import json
from pathlib import Path

import pandas as pd

from ingestion.data_extraction import extract_all


OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "dvc" / "raw"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[extract] pulling from Neon...")
    raw = extract_all()

    for key, df in raw.items():
        if isinstance(df, pd.DataFrame):
            out = OUT_DIR / f"{key}.parquet"
            df.to_parquet(out, index=False)
            print(f"  wrote {out}  ({len(df):,} rows)")

    counts_path = OUT_DIR / "content_counts.json"
    with open(counts_path, "w") as f:
        json.dump(raw["content_counts"], f, indent=2)
    print(f"  wrote {counts_path}")

    print(f"[extract] done")


if __name__ == "__main__":
    main()
