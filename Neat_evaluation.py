# neat_eval_parallel_rewrite.py
from __future__ import annotations
import os
from typing import Iterable, Any, Tuple, List
import numpy as np
import neat
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
from torch import nn

# --------------------------------------------------------------------
# Module-level worker globals (populated in each worker by initializer)
# --------------------------------------------------------------------
_WORKER_PROFILES: np.ndarray | None = None
_WORKER_M: int = 0
_WORKER_N: int = 0
_WORKER_THETA_MINUS_BY_I: np.ndarray | None = None
_WORKER_S_THETA: np.ndarray | None = None
_WORKER_TOL: float = 1e-4
_WORKER_ALPHA: float = 0.0
_WORKER_ALPHA_DELTA: float = 0.0
_WORKER_CONFIG: Any = None


# --------------------------------------------------------------------
# Worker initializer: called once in each worker process
# --------------------------------------------------------------------
def _init_worker(
    worst_case_profiles: Iterable[Iterable[float]],
    n: int,
    config: Any,
    tol: float = 1e-4,
    alpha: float = 0.0,
    alpha_delta: float = 0.0,
) -> None:
    """Initialize module-level globals inside each worker process.

    This is passed as `initializer`/`initargs` to ProcessPoolExecutor so
    workers avoid re-parsing the profiles for every genome.
    """
    global _WORKER_PROFILES, _WORKER_M, _WORKER_N, _WORKER_THETA_MINUS_BY_I, _WORKER_S_THETA
    global _WORKER_TOL, _WORKER_ALPHA, _WORKER_ALPHA_DELTA, _WORKER_CONFIG

    _WORKER_CONFIG = config
    _WORKER_TOL = float(tol)
    _WORKER_ALPHA = float(alpha)
    _WORKER_ALPHA_DELTA = float(alpha_delta)
    _WORKER_N = int(n)

    arr = np.asarray(list(worst_case_profiles), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != _WORKER_N:
        raise ValueError(f"Profiles must be shape (M, N), got {arr.shape}")

    _WORKER_M = int(arr.shape[0])
    _WORKER_PROFILES = arr
    _WORKER_S_THETA = np.maximum(arr.sum(axis=1), 1.0)

    theta_minus_by_i = np.empty((_WORKER_N, _WORKER_M, _WORKER_N - 1), dtype=float)
    for i in range(_WORKER_N):
        mask = np.ones(_WORKER_N, bool)
        mask[i] = False
        theta_minus_by_i[i] = arr[:, mask]
    _WORKER_THETA_MINUS_BY_I = theta_minus_by_i


class GenomeNetTorch(nn.Module):
    def __init__(self, genome, config):
        super().__init__()
        self.config = config
        # gather nodes and connections
        try:
            node_ids = list(genome.nodes.keys())
        except Exception:
            node_ids = list(genome.nodes)
        self.node_ids = node_ids
        self.incoming = {}
        for (i, o), conn in genome.connections.items():
            if not conn.enabled:
                continue
            self.incoming.setdefault(o, []).append((i, conn.weight))

        num_inputs = config.genome_config.num_inputs
        num_outputs = config.genome_config.num_outputs
        self.input_ids = [i for i in range(-num_inputs, 0)]
        self.output_ids = list(range(num_outputs))

        self.topo_order = self._compute_topo_order()

    def _compute_topo_order(self):
        preds = {}
        nodes = set(self.node_ids)
        for (i, o), _ in getattr(self, 'incoming', {}).items():
            preds.setdefault(o, set()).add(i)

        ready = [n for n in nodes if (n not in preds) or all(p in self.input_ids for p in preds.get(n, []))]
        order = []
        visited = set()
        while ready:
            node = ready.pop(0)
            if node in visited:
                continue
            order.append(node)
            visited.add(node)
            for (i, o), _ in getattr(self, 'incoming', {}).items():
                if i == node:
                    pred_set = preds.get(o, set())
                    if all((p in visited) or (p in self.input_ids) for p in pred_set):
                        ready.append(o)
        for n in nodes:
            if n not in visited:
                order.append(n)
        return order

    def forward(self, x):
        device = x.device
        batch = x.shape[0]
        node_vals = {}
        for idx, nid in enumerate(self.input_ids):
            node_vals[nid] = x[:, idx]
        for node in self.topo_order:
            if node in self.input_ids:
                continue
            incoming = self.incoming.get(node, [])
            if not incoming:
                node_vals[node] = torch.zeros(batch, device=device)
                continue
            s = torch.zeros(batch, device=device)
            for i, w in incoming:
                v = node_vals.get(i)
                if v is None:
                    continue
                s = s + v * float(w)
            node_vals[node] = torch.tanh(s)
        outputs = [node_vals.get(o, torch.zeros(batch, device=device)) for o in self.output_ids]
        if outputs:
            return torch.stack(outputs, dim=1)
        return torch.zeros(batch, 1, device=device)


# --------------------------------------------------------------------
# Genome evaluation function that runs inside a worker process
# --------------------------------------------------------------------
def _eval_genome_parallel(pair: Tuple[int, Any]) -> Tuple[int, float, float, bool, float]:
    """Evaluate a single (gid, genome) pair using module-level worker globals.

    Returns (gid, fitness, worst_alpha, infeasible_flag, max_violation)
    """
    gid, genome = pair

    N = _WORKER_N
    M = _WORKER_M
    tol = _WORKER_TOL
    alpha_use = _WORKER_ALPHA - _WORKER_ALPHA_DELTA
    s_theta = _WORKER_S_THETA
    theta_minus_by_i = _WORKER_THETA_MINUS_BY_I

    try:
        net = GenomeNetTorch(genome, _WORKER_CONFIG)
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.to(dev).eval()
        stacked_inputs = torch.from_numpy(theta_minus_by_i.reshape(N * M, N - 1)).to(dev, dtype=torch.float32)
        with torch.no_grad():
            outputs = net(stacked_inputs).view(N, M, -1)
        S = outputs.sum(dim=2).sum(dim=0).cpu().numpy()  # shape (M,)
    except Exception:
        # fallback to CPU neat if something goes wrong in torch path
        net = neat.nn.FeedForwardNetwork.create(genome, _WORKER_CONFIG)
        S = np.zeros(M)
        for i in range(N):
            out_vals = [float(net.activate(inp.tolist())[0]) for inp in theta_minus_by_i[i]]
            S += np.array(out_vals)

    left_error = (N - 1) * s_theta - S
    left_error[left_error < 0.0] = 0.0
    right_error = S - (N - (alpha_use)) * s_theta
    right_error[right_error < 0.0] = 0.0
    total_error = left_error + right_error

    infeasible = bool((total_error > tol).any())
    max_violation = float(total_error.max()) if total_error.size > 0 else 0.0

    achieved_alpha = N - S / s_theta
    worst_alpha_val = float(np.min(achieved_alpha)) if achieved_alpha.size > 0 else float(_WORKER_ALPHA)
    worst_alpha = min(_WORKER_ALPHA, worst_alpha_val)

    fitness = worst_alpha - max_violation
    return (gid, float(fitness), float(worst_alpha), bool(infeasible), float(max_violation))


# --------------------------------------------------------------------
# Simple neat_evaluator class: create in main, call to evaluate a generation
# --------------------------------------------------------------------
class neat_evaluator:
    """Simple evaluator object.

    Usage:
        evaluator = neat_evaluator(n, neat_config, max_workers=8)
        # inside training loop:
        winner = evaluator.run_one_generation(pop, worst_case_profiles, alpha, alpha_delta)

    Methods:
      - evaluate_genomes(genomes, config, worst_case_profiles, alpha, alpha_delta)
        sets genome.fitness and attributes wca/vio for the provided genomes (no return)

      - run_one_generation(population, worst_case_profiles, alpha, alpha_delta)
        runs `pop.run(...)` for a single generation and returns the winning genome.
    """

    def __init__(self, n: int, config: Any, *, max_workers: int | None = None, tol: float = 1e-4):
        self.n = int(n)
        self.config = config
        self.tol = float(tol)
        self.max_workers = max_workers if max_workers is not None else max(1, (os.cpu_count() or 1))

    def evaluate_genomes(
        self,
        genomes: Iterable[Tuple[int, Any]],
        config: Any,
        worst_case_profiles: Iterable[Iterable[float]],
        alpha: float,
        alpha_delta: float,
    ) -> None:
        """Evaluate the provided genomes (an iterable of (gid, genome)) across the given profiles.

        This uses a ProcessPoolExecutor with initializer to pre-load profiles in workers.
        """
        genome_pairs = list(genomes)
        if not genome_pairs:
            return

        # ensure args types
        alpha_f = float(alpha)
        alpha_delta_f = float(alpha_delta)
        tol_f = float(self.tol)
        max_workers = min(int(self.max_workers), len(genome_pairs))

        # Prepare executor with initializer so each worker precomputes arrays once.
        results: List[Tuple[int, float, float, bool, float]] = []
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(worst_case_profiles, self.n, self.config, tol_f, alpha_f, alpha_delta_f),
        ) as exe:
            futures = [exe.submit(_eval_genome_parallel, pair) for pair in genome_pairs]
            for fut in as_completed(futures):
                results.append(fut.result())

        # assign results back to genome objects
        id_to_genome = {gid: g for gid, g in genome_pairs}
        for gid, fitness, worst_alpha, infeasible, max_violation in results:
            genome = id_to_genome[gid]
            genome.fitness = float(fitness)
            setattr(genome, "wca", float(worst_alpha))
            setattr(genome, "vio", float(max_violation))

    def run_one_generation(self, pop: "neat.Population", worst_case_profiles: Iterable[Iterable[float]], alpha: float, alpha_delta: float):
        """Run a single generation with `pop.run(...)` using this evaluator and return the winner genome."""
        # pop.run expects a function(genomes, config) -> None that sets genome.fitness
        def _eval_wrapper(genomes, config):
            self.evaluate_genomes(genomes, config, worst_case_profiles, alpha, alpha_delta)

        winner = pop.run(_eval_wrapper, 1)
        return winner

    def run(self, pop: "neat.Population", worst_case_profiles: Iterable[Iterable[float]], alpha: float, alpha_delta: float, generations: int = 1):
        """Run multiple generations (>=1). Returns the winner of the last generation.

        Example:
            winner = evaluator.run(pop, profiles, alpha, alpha_delta, generations=5)
        """
        if generations < 1:
            raise ValueError("generations must be >= 1")

        def _eval_wrapper(genomes, config):
            self.evaluate_genomes(genomes, config, worst_case_profiles, alpha, alpha_delta)

        winner = pop.run(_eval_wrapper, generations)
        return winner

    # Backwards-compatible callable API (so lambda genomes, config: evaluator(genomes, config, ... ) still works)
    def __call__(self, genomes: Iterable[Tuple[int, Any]], config: Any, worst_case_profiles: Iterable[Iterable[float]], alpha: float, alpha_delta: float):
        return self.evaluate_genomes(genomes, config, worst_case_profiles, alpha, alpha_delta)
