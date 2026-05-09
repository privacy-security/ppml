import csv
import os
from typing import List, Tuple, Dict, Optional

def write_round_metrics_to_file(*args, **kwargs):
    """
    Universal metrics writer for Flower aggregated metrics.

    Supports two calling styles for backward compatibility:
      write_round_metrics_to_file(round_number, metrics, filepath="metrics.csv")
      write_round_metrics_to_file(metrics=metrics, round_number=None, filepath="metrics.csv")
      write_round_metrics_to_file(metrics=metrics)  # filepath optional

    Args:
        round_number (Optional[int]): FL round (may be None).
        metrics (List[Tuple[int, Dict]]): Flower metrics list.
        filepath (str): CSV file path.
    """

    # Normalize arguments
    # Accept either call signature:
    #   (round_number, metrics, filepath=...)
    # or keyword style: metrics=..., round_number=..., filepath=...
    if len(args) == 0:
        round_number = kwargs.get("round_number", None)
        metrics = kwargs.get("metrics", [])
        filepath = kwargs.get("filepath", "metrics.csv")
    elif len(args) == 1:
        # single positional arg -> assume it's metrics
        round_number = kwargs.get("round_number", None)
        metrics = args[0]
        filepath = kwargs.get("filepath", "metrics.csv")
    else:
        # (round_number, metrics, [filepath])
        round_number = args[0]
        metrics = args[1]
        filepath = args[2] if len(args) > 2 else kwargs.get("filepath", "metrics.csv")

    # Ensure metrics is iterable
    if metrics is None:
        metrics = []

    # ----------------------------------------------------------
    # 1. Extract metric names dynamically and compute weighted sums
    # ----------------------------------------------------------
    aggregated = {}

    for num_examples, m in metrics:
        for k, v in m.items():
            # initialize accumulators if missing
            if k not in aggregated:
                aggregated[k] = 0.0
                aggregated[k + "_weight"] = 0.0

            aggregated[k] += float(v) * float(num_examples)
            aggregated[k + "_weight"] += float(num_examples)

    # Final weighted averages:
    # For each metric_weight entry, compute metric = aggregated[metric] / aggregated[metric + "_weight"]
    final_metrics = {}
    for key in list(aggregated.keys()):
        if key.endswith("_weight"):
            metric_name = key[:-7]  # remove '_weight'
            weight = aggregated[key]
            if weight == 0:
                final_metrics[metric_name] = 0.0
            else:
                # aggregated[metric_name] holds the numerator
                final_metrics[metric_name] = aggregated.get(metric_name, 0.0) / weight

    # ----------------------------------------------------------
    # 2. Prepare row with dynamic columns
    # ----------------------------------------------------------
    row = {}
    row["round"] = round_number if round_number is not None else ""
    row.update(final_metrics)

    # ----------------------------------------------------------
    # 3. Create CSV with dynamic header if first time
    # ----------------------------------------------------------
    file_exists = os.path.isfile(filepath)

    # Use deterministic column order
    fieldnames = sorted(row.keys())

    # If file exists, ensure existing header is compatible; if not, we still append with our current columns.
    # (For more complex workflows you might want to rewrite the file when columns change.)
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        # Ensure we write only keys matching fieldnames
        writer.writerow({k: row.get(k, "") for k in fieldnames})
