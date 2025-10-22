# neat_eval_optimized.py
from __future__ import annotations
import os
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, List, Tuple, Any
import numpy as np

# --------------------------------------------------------------------
# Worker-global variables (each process will get its own copy via initializer)
# --------------------------------------------------------------------
_WORKER_PROFILES: List[List[float]] | None = None     # list of profiles (python lists)
_WORKER_M: int = 0
_WORKER_N: int = 0
_WORKER_THETA_MINUS: List[List[List[float]]] | None = None  # [profile_idx][i] -> list length n-1
_WORKER_S_THETA: List[float] | None = None
_WORKER_TOL: float = 1e-4
_WORKER_ALPHA: float = 0.0
_WORKER_ALPHA_DELTA: float = 0.0
_WORKER_CONFIG: Any = None   # neat config object for FeedForwardNetwork.create

# --------------------------------------------------------------------
# Initializer run once per worker process
# --------------------------------------------------------------------
def _init_worker(
    worst_case_profiles: Iterable[Iterable[float]],
    n: int,
    config: Any,
    tol: float = 1e-4,
    alpha: float = 0.0,
    alpha_delta: float = 0.0,
) -> None:
    """
    Run in each worker process once. Precompute:
      - profiles as python lists
      - s_theta (sum or 1.0 max)
      - theta_minus_i lists for fast net.activate calls
    This avoids repeated numpy conversion and concatenation per genome.
    """
    global _WORKER_PROFILES, _WORKER_M, _WORKER_N, _WORKER_THETA_MINUS, _WORKER_S_THETA
    global _WORKER_TOL, _WORKER_ALPHA, _WORKER_ALPHA_DELTA, _WORKER_CONFIG

    _WORKER_CONFIG = config
    _WORKER_TOL = tol
    _WORKER_ALPHA = alpha
    _WORKER_ALPHA_DELTA = alpha_delta
    _WORKER_N = n

    # Convert to numpy once for validation, then to python lists for activate()
    arr = np.asarray(list(worst_case_profiles), dtype=float)
    if arr.ndim == 1 and arr.shape[0] == 0:
        raise ValueError("worst_case_profiles is empty")
    if arr.size == 0:
        raise ValueError("worst_case_profiles is empty")

    m = arr.shape[0]
    # Basic validation of shape: each profile must have length n
    if arr.shape[1] != n:
        # provide a clear message
        raise ValueError(f"worst_case_profiles shape[1] = {arr.shape[1]} != expected n = {n}")

    _WORKER_M = m
    # make python lists-of-lists (neat net.activate prefers sequences)
    _WORKER_PROFILES = [row.tolist() for row in arr]

    # s_theta is max(sum(theta), 1.0) per profile
    sums = np.sum(arr, axis=1)
    _WORKER_S_THETA = np.maximum(sums, 1.0).tolist()

    # Precompute theta_minus_i for every profile and every i as python lists
    # Format: _WORKER_THETA_MINUS[profile_idx][i] -> list length n-1
    _WORKER_THETA_MINUS = []
    for theta_list in _WORKER_PROFILES:
        row = []
        # use list slicing (cheap in Python)
        for i in range(n):
            row.append(theta_list[:i] + theta_list[i + 1 :])
        _WORKER_THETA_MINUS.append(row)


# --------------------------------------------------------------------
# Core per-genome evaluator that uses worker globals (runs inside worker)
# --------------------------------------------------------------------
def _evaluate_single_genome_worker(pair: Tuple[int, Any]) -> Tuple[int, float, float, bool, float]:
    """
    Evaluate a single genome inside a worker process. This expects that
    _init_worker has already run in this process to set up globals.
    Returns: (genome_id, fitness, worst_alpha, infeasible_flag, max_violation)
    """
    global _WORKER_PROFILES, _WORKER_M, _WORKER_N, _WORKER_THETA_MINUS, _WORKER_S_THETA
    global _WORKER_TOL, _WORKER_ALPHA, _WORKER_ALPHA_DELTA, _WORKER_CONFIG

    gid, genome = pair

    # create network for genome using global config (unavoidable per-genome)
    # neat.nn.FeedForwardNetwork.create(genome, config)
    # Some neat wrappers store config on genome; use _WORKER_CONFIG for reliability
    net = __import__("neat").nn.FeedForwardNetwork.create(genome, _WORKER_CONFIG)

    infeasible = False
    worst_alpha = _WORKER_ALPHA
    max_violation = 0.0
    alpha_use = _WORKER_ALPHA - _WORKER_ALPHA_DELTA

    # Local copies for speed
    M = _WORKER_M
    N = _WORKER_N
    theta_minus = _WORKER_THETA_MINUS
    s_theta_list = _WORKER_S_THETA
    tol = _WORKER_TOL

    for idx in range(M):
        # fast locals
        s_theta = s_theta_list[idx]
        theta_minus_for_profile = theta_minus[idx]

        # Sum the network outputs S = sum_i net(theta_minus_i)
        S = 0.0
        # net.activate expects a sequence; precomputed lists avoid allocations
        for i in range(N):
            # cast to float in case net returns np.float32 etc
            S += float(net.activate(theta_minus_for_profile[i])[0])

        left_error = (N - 1) * s_theta - S
        right_error = S - (N - (alpha_use)) * s_theta
        total_error = (left_error if left_error > 0.0 else 0.0) + (right_error if right_error > 0.0 else 0.0)

        if total_error > max_violation:
            max_violation = total_error

        if total_error > tol:
            infeasible = True
            # continue to compute max_violation across profiles
            continue

        achieved_alpha = N - S / s_theta
        if achieved_alpha < worst_alpha:
            worst_alpha = achieved_alpha

    fitness = -max_violation if infeasible else max(0.0, worst_alpha)
    return (gid, fitness, worst_alpha, infeasible, max_violation)


# --------------------------------------------------------------------
# Batch worker wrapper: evaluate a list of genomes in one call (reduces pickling)
# --------------------------------------------------------------------
def _evaluate_genome_batch_worker(batch: List[Tuple[int, Any]]) -> List[Tuple[int, float, float, bool, float]]:
    """
    Evaluate a batch of (gid, genome) pairs inside a worker process.
    Returns a list of results.
    """
    results = []
    for pair in batch:
        results.append(_evaluate_single_genome_worker(pair))
    return results


# --------------------------------------------------------------------
# Utility: chunk an iterable into fixed-size batches
# --------------------------------------------------------------------
def _chunked(iterable: Iterable, size: int):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


# --------------------------------------------------------------------
# Public function to evaluate genomes in parallel (optimized)
# --------------------------------------------------------------------
def eval_genomes_in_parallel_optimized(
    genome_pairs: Iterable[Tuple[int, Any]],
    config: Any,
    worst_case_profiles: Iterable[Iterable[float]],
    alpha: float,
    alpha_delta: float,
    n: int,
    max_workers: int | None = None,
    batch_size: int = 8,
    tol: float = 1e-4,
) -> None:
    """
    Parallel evaluation using ProcessPoolExecutor with a per-worker initializer
    and batching to reduce overhead.

    - genome_pairs: iterable of (gid, genome) pairs
    - config: neat config object (stored in worker globals)
    - worst_case_profiles: iterable of profile iterables (must be length n)
    - alpha, alpha_delta, n: same meaning as your original code
    - max_workers: number of worker processes (default: min(os.cpu_count(), 16))
    - batch_size: number of genomes per job submitted to worker (tune)
    - tol: tolerance for infeasibility check
    """

    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, 16)

    # Convert genome_pairs to list because we'll need to map results back to genomes
    genome_pairs_list = list(genome_pairs)
    # mapping from id -> genome object (in parent process)
    id_to_genome = {gid: g for (gid, g) in genome_pairs_list}

    # Build batches (lists of (gid, genome))
    batches = list(_chunked(genome_pairs_list, batch_size))

    # Start executor with initializer to precompute constants inside each worker
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(worst_case_profiles, n, config, tol, alpha, alpha_delta),
    ) as exe:
        # Submit batch tasks
        futures = [exe.submit(_evaluate_genome_batch_worker, batch) for batch in batches]

        # Collect results as they complete
        for fut in as_completed(futures):
            batch_results = fut.result()
            for gid, fitness, worst_alpha, infeasible, max_violation in batch_results:
                genome_obj = id_to_genome[gid]
                # write back into genome objects in parent process
                genome_obj.fitness = fitness
                # store worst-case alpha (wca) attribute for downstream use
                setattr(genome_obj, "wca", worst_alpha)

    # function returns None; genomes updated in-place
    return


# --------------------------------------------------------------------
# Compatibility wrapper: replicate previous neat_evaluator signature
# --------------------------------------------------------------------
def neat_evaluator(
    genomes: Iterable[Tuple[int, Any]],
    config: Any,
    worst_case_profiles: Iterable[Iterable[float]],
    alpha: float,
    alpha_delta: float,
    n: int,
    *,
    max_workers: int | None = None,
    batch_size: int = 8,
    tol: float = 1e-4,
) -> None:
    """
    Top-level convenience wrapper; matches old API.
    """
    eval_genomes_in_parallel_optimized(
        genome_pairs=list(genomes),
        config=config,
        worst_case_profiles=worst_case_profiles,
        alpha=alpha,
        alpha_delta=alpha_delta,
        n=n,
        max_workers=max_workers,
        batch_size=batch_size,
        tol=tol,
    )
