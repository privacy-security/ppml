"""
dp_utils.py -- one source of truth for client/contribution-level DP accounting.

It wraps the SAME accountant server already uses (Google's dp_accounting,
RDP over R self-composed Poisson-sampled Gaussian rounds), so:

  * client_level_epsilon(z, ...)          reproduces logged fl_dp_epsilon_approx
  * noise_multiplier_for_epsilon(eps, ...) inverts it to the z that hits a target eps

Both CDP and LDP call noise_multiplier_for_epsilon with the SAME arguments -- at full
participation (q=1) the per-client accounting is identical, so they get the same z and
differ only in where the noise is added (server aggregate vs. each client).

delta convention: delta = 1 / (10 * M).
"""
from __future__ import annotations
import math

try:
    import dp_accounting
    _HAVE = True
except Exception:
    dp_accounting = None
    _HAVE = False

_ORDERS = [1.0 + x / 10.0 for x in range(1, 1000)] + list(range(12, 400))


def delta_for_clients(num_clients_total: int) -> float:
    """Your delta rule: 1 / (10 * M)."""
    return 1.0 / (10.0 * float(num_clients_total))


def client_level_epsilon(
    noise_multiplier: float,
    num_rounds: int,
    num_clients_total: int,
    num_clients_sampled: int | None = None,
    delta: float | None = None,
) -> float | None:
    """Client-level epsilon for R rounds of a (Poisson-sampled) Gaussian mechanism.

    Mirrors server_app.compute_fl_user_level_epsilon exactly, so the epsilon returned
    here equals what server logs.
    """
    if not _HAVE:
        return None
    if noise_multiplier <= 0 or num_rounds <= 0 or num_clients_total <= 0:
        return None
    if num_clients_sampled is None:
        num_clients_sampled = num_clients_total
    if delta is None:
        delta = delta_for_clients(num_clients_total)

    q = min(1.0, float(num_clients_sampled) / float(num_clients_total))
    acc = dp_accounting.rdp.RdpAccountant(_ORDERS)
    event = dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(q, dp_accounting.GaussianDpEvent(noise_multiplier)),
        int(num_rounds),
    )
    acc.compose(event)
    return float(acc.get_epsilon(target_delta=delta))


def noise_multiplier_for_epsilon(
    target_epsilon: float,
    num_rounds: int,
    num_clients_total: int,
    num_clients_sampled: int | None = None,
    delta: float | None = None,
    tol: float = 1e-3,
    z_lo: float = 1e-3,
    z_hi: float = 1.0e4,
    max_iter: int = 100,
) -> float:
    """Smallest noise multiplier z whose client-level epsilon <= target_epsilon.

    Monotone: more noise -> smaller epsilon, so we binary-search z.
    """
    if not _HAVE:
        raise RuntimeError(
            "dp_accounting is not installed, but it is required to calibrate noise to a "
            "target epsilon. `pip install dp-accounting`."
        )
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be > 0")
    if num_clients_sampled is None:
        num_clients_sampled = num_clients_total
    if delta is None:
        delta = delta_for_clients(num_clients_total)

    def eps(z):
        return client_level_epsilon(z, num_rounds, num_clients_total, num_clients_sampled, delta)

    lo, hi = z_lo, z_hi
    for _ in range(max_iter):
        mid = math.sqrt(lo * hi)
        e = eps(mid)
        if e is None:
            raise RuntimeError("accountant returned None during inversion")
        if e > target_epsilon:   # too little noise -> increase z
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


if __name__ == "__main__":
    # Print the calibration table (sanity check against your logs).
    print("dp_accounting available:", _HAVE)
    for name, R in [("network", 30), ("smoking", 30), ("cifar", 50)]:
        for M in (3, 6, 10):
            d = delta_for_clients(M)
            zs = {e: round(noise_multiplier_for_epsilon(e, R, M, M, d), 3) for e in (1, 3, 5)}
            back = {e: round(client_level_epsilon(zs[e], R, M, M, d), 2) for e in (1, 3, 5)}
            print(f"{name:8} R={R} M={M:<3} delta={d:.4f}  "
                  f"z(1,3,5)={zs[1]},{zs[3]},{zs[5]}  check_eps={back[1]},{back[3]},{back[5]}")
        print()

"""
==================================================================
EXAMPLE-LEVEL (record-level) DP-SGD accounting
==================================================================
Protected unit = one training RECORD (not a whole client). Each client runs
DP-SGD locally: per-example gradient clipping to C + Gaussian noise. A record
lives in exactly one client, so its privacy is accounted over THAT client's
local training across all rounds, WITH subsampling amplification from the
minibatch sampling (q = batch / n_examples_per_client). This amplification is why example-
level z is far smaller than client-level z for the same epsilon.

steps  T = num_rounds * local_epochs * ceil(n_examples_per_client / batch)
rate   q = batch / n_examples_per_client
delta tied to the record-sampling universe (per-client dataset size).
"""

def example_delta(n_examples_per_client: int) -> float:
    """delta convention for example level: 1 / (10 * n_examples_per_client)."""
    return 1.0 / (10.0 * float(n_examples_per_client))


def example_level_epsilon(
    noise_multiplier: float,
    steps: int,
    sample_rate: float,
    delta: float,
) -> float | None:
    """Record-level epsilon for `steps` Poisson-subsampled Gaussian steps."""
    if not _HAVE:
        return None
    if noise_multiplier <= 0 or steps <= 0 or not (0 < sample_rate <= 1):
        return None
    # Subsampled RDP is expensive per order, and z is computed once per run at
    # startup, so use the same moderate grid as the server accountant (fast,
    # accurate enough) rather than the 1400-order client-level grid.
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)) + [128, 256, 512]
    acc = dp_accounting.rdp.RdpAccountant(orders)
    event = dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(sample_rate, dp_accounting.GaussianDpEvent(noise_multiplier)),
        int(steps),
    )
    acc.compose(event)
    return float(acc.get_epsilon(target_delta=delta))


def example_level_noise_for_epsilon(
    target_epsilon: float,
    steps: int,
    sample_rate: float,
    delta: float,
    z_lo: float = 1e-3,
    z_hi: float = 1.0e4,
    max_iter: int = 100,
) -> float:
    """Smallest z whose record-level epsilon <= target_epsilon (binary search)."""
    if not _HAVE:
        raise RuntimeError("dp_accounting required. `pip install dp-accounting`.")
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be > 0")

    def eps(z):
        return example_level_epsilon(z, steps, sample_rate, delta)

    lo, hi = z_lo, z_hi
    for _ in range(max_iter):
        mid = math.sqrt(lo * hi)
        e = eps(mid)
        if e is None:
            raise RuntimeError("accountant returned None during inversion")
        if e > target_epsilon:   # too little noise -> increase z
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def example_level_steps(n_examples_per_client: int, batch_size: int, num_rounds: int, local_epochs: int = 1) -> int:
    """T = rounds * local_epochs * ceil(n_examples_per_client / batch)."""
    spe = math.ceil(float(n_examples_per_client) / float(batch_size))
    return int(num_rounds) * int(local_epochs) * int(spe)
