#!/usr/bin/env python3
"""
fetch_runs.py -- pull ALL runs from the wandb project via the API and write a
FLAT csv (one column per field) matching the export schema that
analyze_results.py / select_attack_configs.py consume.

The naive api.runs() loop that stores whole dicts per cell does NOT produce that
schema. This flattens config + summary into top-level columns and pages through
every run (the UI CSV export truncates around 1000; the API does not).

Shows a live progress bar with percent + ETA (via tqdm if available; otherwise a
lightweight built-in fallback so it still works without the dependency).

Usage:
    python fetch_runs.py                       # -> project.csv
    python fetch_runs.py --out mydata.csv --per-page 500
"""
import argparse
import sys
import time
import pandas as pd
import wandb
from pathlib import Path

PROJECT = "dpnerds/privacy_preserving_federated_learning"


def get_total(runs):
    """Total run count for the query, if the API exposes it (for a real %)."""
    try:
        n = getattr(runs, "length", None)
        if isinstance(n, int) and n > 0:
            return n
    except Exception:
        pass
    try:
        return len(runs)          # may work depending on wandb version
    except Exception:
        return None


def progress(iterable, total, desc):
    """tqdm if present; else a minimal percent/ETA bar on one line."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc, unit="run", dynamic_ncols=True)
    except Exception:
        def gen():
            start = time.time()
            for i, item in enumerate(iterable, 1):
                yield item
                if total:
                    frac = i / total
                    elapsed = time.time() - start
                    eta = elapsed / frac - elapsed if frac > 0 else 0
                    filled = int(30 * frac)
                    bar = "#" * filled + "-" * (30 - filled)
                    sys.stdout.write(
                        f"\r{desc}: [{bar}] {i}/{total} {frac*100:5.1f}%  ETA {eta:5.0f}s")
                else:
                    sys.stdout.write(f"\r{desc}: {i} runs")
                sys.stdout.flush()
            sys.stdout.write("\n")
        return gen()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--out", default="results/project.csv")
    ap.add_argument("--per-page", type=int, default=1000)
    args = ap.parse_args()

    # Create output directory if it doesn't exist
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=60)
    runs = api.runs(args.project, per_page=args.per_page)

    total = get_total(runs)
    print(f"Fetching runs from {args.project}"
          + (f"  (total: {total})" if total else "  (total unknown)"))

    rows = []
    for run in progress(runs, total, "fetch"):
        row = {
            "Name": run.name,
            "State": run.state,          # 'finished' / 'failed' / 'running' / 'crashed'
            "Sweep": run.sweep.id if run.sweep else None,
            "Runtime": run.summary._json_dict.get("_runtime"),
        }
        for k, v in run.config.items():
            if not k.startswith("_"):
                row[k] = v
        for k, v in run.summary._json_dict.items():
            if k.startswith("_"):
                continue
            if isinstance(v, (int, float, str, bool)) or v is None:
                row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}: {len(df)} runs, {df.shape[1]} columns")
    
    if "State" in df:
        print("State:", df["State"].value_counts(dropna=False).to_dict())
    if "DATASET" in df:
        print("DATASET:", df["DATASET"].value_counts(dropna=False).to_dict())
    for probe in ("server/global_eval/accuracy_binary", "server/global_eval/mse", "epsilon"):
        if probe in df.columns:
            print(f"  {probe:38} non-null={df[probe].notna().sum()}")


if __name__ == "__main__":
    main()
