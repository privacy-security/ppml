from flwr.common import Metrics
from typing import List, Tuple, Dict
from flwr.common import Metrics
import numpy as np
import wandb
import logging
from .utils import write_round_metrics_to_file

logger = logging.getLogger(__name__)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Universal metric aggregator.
    - Accepts list of tuples: (num_examples, metrics_dict)
    - Returns a Metrics dict with weighted averages for every metric reported by any client.
    - Ignores 'loss' (Flower handles loss separately) and noisy keys like 'compile_metrics'.
    """

    if not metrics:
        return {}

    print("\n====== DEBUG weighted_average INPUT ======")
    # Collect union of all keys reported by clients
    union_keys = set()
    for num_examples, m in metrics:
        union_keys.update(m.keys())
        print(f"num_examples={num_examples} metrics={m}")
    print("=========================================\n")


    # Exclude keys we don't want aggregated here
    excluded_keys = {"loss", "compile_metrics"}  # add more if you see other noise
    metric_keys = sorted(k for k in union_keys if k not in excluded_keys)

    aggregated: Dict[str, float] = {}

    # Total examples for each metric computed from clients that reported it
    for key in metric_keys:
        weighted_sum = 0.0
        total_weight = 0
        values = []
        for num_examples, m in metrics:
            if key in m:
                try:
                    val = float(m[key])
                    weighted_sum += num_examples * val
                    total_weight += num_examples
                    values.append(val)
                except Exception:
                    # skip non-convertible metrics
                    logger.warning(f"Could not convert metric {key} value={m.get(key)} to float; skipping client")
                    continue
        if total_weight > 0:
            aggregated[key] = weighted_sum / total_weight
        else:
            # fallback (shouldn't happen if at least one client reported it)
            aggregated[key] = 0.0

        # W&B logging (wrapped to avoid bubbling exceptions)
        try:
            wandb.log({
                f"aggr_{key}_mean": float(np.mean(values)) if values else 0.0,
                f"aggr_{key}_max": float(np.max(values)) if values else 0.0,
                f"aggr_{key}_min": float(np.min(values)) if values else 0.0,
                f"aggr_{key}_hist": wandb.Histogram(values) if values else None,
            })
        except Exception as e:
            logger.debug(f"W&B logging failed for aggregated key={key}: {e}")

    # Optional: write to file (keep backward compat)
    try:
        write_round_metrics_to_file(metrics=metrics)
    except TypeError:
        write_round_metrics_to_file(None, metrics)

    print("====== DEBUG weighted_average OUTPUT ======")
    print(aggregated)
    print("===========================================\n")
    return aggregated
