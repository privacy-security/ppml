"""
generate_configs.py -- emit wandb sweep YAMLs, ONE quartet per dataset
(<stem>_fl, <stem>_fl_dp_central, <stem>_fl_dp_local, <stem>_fl_dp_example).
Adding a dataset = add one entry to DATASETS below -> 4 new YAMLs, nothing else
changes.

Key corrections vs. the old configs:
  * BOTH client-level DP mechanisms are parameterized by a TARGET client-level
    epsilon. The Gaussian noise multiplier is computed at runtime in config.py
    from (epsilon, M, R, delta) via the accountant -- there is no
    `noise_multiplier` grid anymore.
  * A `seed` grid is added for the multi-seed protocol (mean +/- std).
  * NEW: `dp_level` is now an EXPLICIT key on every DP sweep.

dp_level -- the unit of privacy
-------------------------------
  "client"   silo-level. Protects a whole client's participation in a round.
             q = 1 (full participation), no amplification -> expensive at low M.
             Two placements: central (server noises the aggregate) and local
             (each client noises its own update). Both share the same z.
  "example"  record-level DP-SGD, run LOCALLY inside each client's train_step
             (per-example clip to C + Gaussian noise z*C). Amplified by
             minibatch subsampling, q = batch / n_examples_per_client, so z is
             far smaller than client-level z at the same epsilon.
             There is deliberately NO central example-level branch.

The key must be emitted, not merely exported: config.py reads DP_LEVEL from the
environment (via run.sh), but server_app.py reads `dp_level` from the W&B config
to decide whether to wrap the strategy with DifferentialPrivacy*FixedClipping.
Without the sweep key, an example-level run would default to "client" on the
server and get double DP.
"""
import os

OUT_DIR = "config"

# ---- sweep axes -------------------------------------------------------------
# Full privacy frontier (Option C). Reported epsilon is whatever the accountant
# logs for the computed z, so these are targets, not post-hoc labels.
# NOTE: the same numeric grid is reused at example level, but the epsilon there
# is a RECORD-level budget. Equal numbers do NOT mean equal protection across
# levels -- that unit difference is the paper's point, not a confound.
EPSILONS = [1, 3, 10, 30, 100, 300]
SEEDS = [0, 1, 2, 3, 4]                 # R2.4 multi-seed
N_CLIENTS = [3, 6, 10]
# ---- clip norms (explicit per-dataset) -------------------------------------
# The DP clip norm must match a model's per-round update norm, which scales with
# model size. CIFAR (CNN, ~1e6 params) has update norm ~68 (median; p90 ~148);
# clipping it to {1,5} kept <1% of the update and froze training at random. Small
# tabular/GRU models sit near {1,5}.
#
# FL (no-DP) clip is a NO-OP -- nothing reads it when dp=False -- but each dataset
# still carries a single scalar so every plain run has a clean, un-duplicated
# grouping key. The cifar value (70) is cosmetic parity only; it changes nothing.
DP_CLIP = {                             # CLIENT-level clip GRID: clips the per-round UPDATE
    "network_monitoring":     [1, 5],
    "body_signal_of_smoking": [1, 5],
    "cifar10":                [1, 5],
    "household_power":        [1, 5],
}
# EXAMPLE-level clip GRID: clips a PER-EXAMPLE GRADIENT, a different object on a
# different scale than the per-round update above. Kept numerically identical to
# DP_CLIP for now (so nothing changes silently), but separated so the two can be
# retuned independently once per-example gradient norms are measured.
EXAMPLE_CLIP = {
    "network_monitoring":     [1, 5],
    "body_signal_of_smoking": [1, 5],
    "cifar10":                [1, 5],
    "household_power":        [1, 5],
}
FL_CLIP = {                             # FL clip SCALAR (no-op; keeps groups clean)
    "network_monitoring":     5,
    "body_signal_of_smoking": 5,
    "cifar10":                1,
    "household_power":        5,
}
# tf.vectorized_map (pfor) is the fast path for per-example gradients in
# DPSGDModel, but dp_sgd.py's own docstring warns that some layers break
# vectorization. Recurrent stacks are the known risk, so the GRU datasets default
# to the sequential map_fn path; flip to True once a run confirms pfor converts.
EXAMPLE_VECTORIZED = {
    "network_monitoring":     False,    # GRU stack
    "body_signal_of_smoking": True,     # Dense MLP
    "cifar10":                True,     # CNN
    "household_power":        False,    # GRU stack
}

def dp_clip(ds):
    return DP_CLIP[ds]

def example_clip(ds):
    return EXAMPLE_CLIP[ds]

def fl_clip(ds):
    return FL_CLIP[ds]

def example_vectorized(ds):
    return EXAMPLE_VECTORIZED[ds]

PARTITION = "iid"

# ---- per-dataset training settings -----------------------------------------
# household_power mirrors network_monitoring (same GRU pipeline) so the dataset
# is the ONLY changed variable in the time-series comparison.
DATASETS = {
    "network_monitoring":     dict(stem="network",   lr=0.01,   bs=128, epochs=30),
    "body_signal_of_smoking": dict(stem="smoking",   lr=0.001,  bs=128, epochs=30),
    "cifar10":                dict(stem="cifar",     lr=0.0005, bs=16,  epochs=50),
    "household_power":        dict(stem="household", lr=0.01,   bs=128, epochs=30),
}

HEADER = "program: run.sh\nproject: privacy_preserving_federated_learning\nmethod: grid\n\nparameters:\n"
COMMAND = "\ncommand:\n  - ${env}\n  - bash\n  - ${program}\n  - ${args}\n"


def _scalar(x):
    if isinstance(x, bool):
        return "True" if x else "False"
    if isinstance(x, str):
        return f'"{x}"'
    return str(x)


def _value(key, x):
    return f"  {key}:\n    value: {_scalar(x)}\n"


def _values(key, xs):
    return f"  {key}:\n    values: [{', '.join(_scalar(x) for x in xs)}]\n"


def build_fl(ds, c):
    # No dp_level key: consistent with `local` and `clipping`, which the plain-FL
    # sweep also omits. Nothing reads dp_level when dp=False.
    return (
        HEADER
        + _value("dataset", ds)
        + _value("aggregation_type", "regular")
        + _value("dp", False)
        + _values("n_clients", N_CLIENTS)
        + _value("partition_type", PARTITION)
        + _value("learning_rate", c["lr"])
        + _value("batch_size", c["bs"])
        + _value("epochs", c["epochs"])
        + _value("l2_norm_clip", fl_clip(ds))
        + _values("seed", SEEDS)
        + COMMAND
    )


def build_dp_central(ds, c):
    # Client-level CDP: server clips + noises the aggregate. server_app.py wraps
    # the strategy only when dp_level == "client" and local == False.
    return (
        HEADER
        + _value("dataset", ds)
        + _value("aggregation_type", "regular")
        + _value("dp", True)
        + _value("dp_level", "client")
        + _value("local", False)
        + _value("clipping", "server")
        + _values("epsilon", EPSILONS)
        + _values("l2_norm_clip", dp_clip(ds))
        + _value("partition_type", PARTITION)
        + _values("n_clients", N_CLIENTS)
        + _value("learning_rate", c["lr"])
        + _value("batch_size", c["bs"])
        + _value("epochs", c["epochs"])
        + _values("seed", SEEDS)
        + COMMAND
    )


def build_dp_local(ds, c):
    # Client-level LDP: each client clips + noises its own update in client.fit().
    return (
        HEADER
        + _value("dataset", ds)
        + _value("aggregation_type", "regular")
        + _value("dp", True)
        + _value("dp_level", "client")
        + _value("local", True)
        + _values("epsilon", EPSILONS)
        + _values("l2_norm_clip", dp_clip(ds))
        + _value("partition_type", PARTITION)
        + _values("n_clients", N_CLIENTS)
        + _value("learning_rate", c["lr"])
        + _value("batch_size", c["bs"])
        + _value("epochs", c["epochs"])
        + _values("seed", SEEDS)
        + COMMAND
    )


def build_dp_example(ds, c):
    # Example-level (record) DP-SGD, local only. No `clipping` key: nothing is
    # clipped at the update level and no server-side strategy wrapper is used.
    # `local: True` is stated explicitly to match what config.py forces anyway
    # (dp_level == "example" => local_dp = True), so the W&B grouping key agrees
    # with the mechanism actually run.
    return (
        HEADER
        + _value("dataset", ds)
        + _value("aggregation_type", "regular")
        + _value("dp", True)
        + _value("dp_level", "example")
        + _value("local", True)
        + _values("epsilon", EPSILONS)
        + _values("l2_norm_clip", example_clip(ds))
        + _value("example_vectorized", example_vectorized(ds))
        + _value("partition_type", PARTITION)
        + _values("n_clients", N_CLIENTS)
        + _value("learning_rate", c["lr"])
        + _value("batch_size", c["bs"])
        + _value("epochs", c["epochs"])
        + _values("seed", SEEDS)
        + COMMAND
    )


def main(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for ds, c in DATASETS.items():
        stem = c["stem"]
        quartet = {
            f"{stem}_fl.yaml": build_fl(ds, c),
            f"{stem}_fl_dp_central.yaml": build_dp_central(ds, c),
            f"{stem}_fl_dp_local.yaml": build_dp_local(ds, c),
            f"{stem}_fl_dp_example.yaml": build_dp_example(ds, c),
        }
        for fname, content in quartet.items():
            with open(os.path.join(out_dir, fname), "w") as f:
                f.write(content)
            n += 1
            print(f"wrote {out_dir}/{fname}")
    print(f"\n{n} YAMLs ({len(DATASETS)} datasets x 4 mechanisms).")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else OUT_DIR)
