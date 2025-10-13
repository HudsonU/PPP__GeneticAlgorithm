# Pulp_utils_neat.py
import numpy as np
import neat
import time
import sys
from pulp import (
    LpVariable,
    LpMaximize,
    LpProblem,
    lpSum,
    value,
    LpStatus,
    PULP_CBC_CMD,
)
from utility import (alpha)
from math import isfinite

# NOTE: this assumes config.genome_config has:
#   - num_inputs (int)
#   - input_keys (ordered sequence of input node ids, e.g. [-4,-3,-2,-1])
#   - output_keys (sequence with a single output node id)
#
# Call worst_case_analysis_neat(genome, config, alpha_delta=0, input_bounds=(0,1))


def compute_big_m(genome, config, input_bounds=(0.0, 1.0), scale=1.0, max_iters=int(1e4),
                  debug=False):
    """
    Robust interval propagation for NEAT genomes (ReLU). Improvements vs. naive:
      - detect SCCs and apply damped updates to nodes in strongly-coupled components
      - clamp extremely large widths to a safe range to avoid numerical explosion
      - keep orphan nodes' pre-interval equal to bias (as before)
    Parameters:
      - debug: if True prints suspicious nodes / sccs
    """
    input_keys = list(getattr(config.genome_config, "input_keys", []))
    output_keys = list(getattr(config.genome_config, "output_keys", []))

    # collect all nodes mentioned (includes inputs, outputs, genome nodes and conn nodes)
    conn_nodes = set()
    for (in_n, out_n) in genome.connections.keys():
        conn_nodes.add(in_n)
        conn_nodes.add(out_n)
    all_nodes = set(genome.nodes.keys()) | set(input_keys) | set(output_keys) | conn_nodes

    # initialize bounds
    node_bounds = {}
    for k in all_nodes:
        if k in input_keys:
            node_bounds[k] = (float(input_bounds[0]), float(input_bounds[1]))
        else:
            node_bounds[k] = (-10.0, 10.0)

    # Precompute incoming enabled connections
    incoming_map = {n: [] for n in all_nodes}
    enabled_edges = []
    for (in_n, out_n), conn in genome.connections.items():
        if conn.enabled:
            incoming_map.setdefault(out_n, [])
            incoming_map[out_n].append((in_n, conn))
            enabled_edges.append((in_n, out_n, float(conn.weight)))

    # Build adjacency for SCC detection (only internal nodes and enabled edges)
    # We'll do a small Tarjan implementation to get SCCs.
    nodes_for_graph = [n for n in all_nodes if n not in input_keys]
    idx_map = {n:i for i,n in enumerate(nodes_for_graph)}
    graph = {n: [] for n in nodes_for_graph}
    for (in_n, out_n, w) in enabled_edges:
        if out_n in graph and in_n in graph:
            graph[in_n].append(out_n)

    # Tarjan to compute SCCs
    index = 0
    stack = []
    onstack = set()
    indices = {}
    lowlink = {}
    sccs = []

    def strongconnect(v):
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)
        for w in graph.get(v, ()):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in onstack:
                lowlink[v] = min(lowlink[v], indices[w])
        if lowlink[v] == indices[v]:
            # root of SCC
            comp = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in nodes_for_graph:
        if v not in indices:
            strongconnect(v)

    # map node -> scc_id
    node_to_scc = {}
    for sid, comp in enumerate(sccs):
        for n in comp:
            node_to_scc[n] = sid

    # choose damping factor per SCC: larger SCCs (or self-loop) -> stronger damping
    scc_damping = {}
    for sid, comp in enumerate(sccs):
        size = len(comp)
        if size == 1:
            # check self-loop
            n = comp[0]
            has_self = any((in_n == n and out_n == n and conn.enabled) for (in_n, out_n), conn in genome.connections.items())
            scc_damping[sid] = 0.9 if has_self else 1.0  # near 1 = no damping for simple nodes
        else:
            # larger SCC -> damp more
            scc_damping[sid] = 0.6  # update = old*beta + new*(1-beta) with beta ~0.6

    # iterative relax / propagation with damping inside SCCs
    for it in range(int(max_iters)):
        changed = False
        # compute tentative updates in temp dict so damping uses old values
        updates = {}
        for node in all_nodes:
            if node in input_keys:
                continue
            node_gene = genome.nodes.get(node)
            bias = float(getattr(node_gene, "bias", 0.0))
            incoming = incoming_map.get(node, [])
            if not incoming:
                lb, ub = bias, bias
            else:
                lb = bias
                ub = bias
                for in_n, conn in incoming:
                    w = float(conn.weight)
                    in_lb, in_ub = node_bounds.get(in_n, (float(input_bounds[0]), float(input_bounds[1])))
                    if w >= 0:
                        lb += w * in_lb
                        ub += w * in_ub
                    else:
                        lb += w * in_ub
                        ub += w * in_lb
            new_lb, new_ub = min(lb, ub), max(lb, ub)
            updates[node] = (new_lb, new_ub)

        # apply updates with damping for SCC nodes
        for node, (new_lb, new_ub) in updates.items():
            prev_lb, prev_ub = node_bounds[node]
            # if node belongs to an SCC that we damp, apply convex combination
            if node in node_to_scc:
                beta = scc_damping.get(node_to_scc[node], 1.0)
                # combine: result = beta * old + (1-beta) * new
                mixed_lb = beta * prev_lb + (1.0 - beta) * new_lb
                mixed_ub = beta * prev_ub + (1.0 - beta) * new_ub
                new_lb, new_ub = mixed_lb, mixed_ub
            # guard ordering
            if new_lb > new_ub:
                new_lb, new_ub = new_ub, new_lb
            # small tolerance compare
            if not (np.isclose(prev_lb, new_lb) and np.isclose(prev_ub, new_ub)):
                node_bounds[node] = (float(new_lb), float(new_ub))
                changed = True

        if not changed:
            break

    # --- AFTER propagation: orphan detection & final normalization ---
    orphan_nodes = set()
    for n in all_nodes:
        if n in input_keys:
            continue
        incoming = incoming_map.get(n, [])
        if len(incoming) == 0:
            orphan_nodes.add(n)

    # final normalization & clamping for safety
    # define a numeric clamp to keep values reasonable for the LP
    MAX_ABS_BOUND = 1e6  # clamp magnitude to 1e6 (tunable)
    MAX_WIDTH = 1e6
    for k, (lb, ub) in list(node_bounds.items()):
        if lb > ub:
            lb, ub = ub, lb
        if not isfinite(lb) or not isfinite(ub):
            lb, ub = -1.0, 1.0
        # ensure a tiny slack
        if abs(ub - lb) < 1e-12:
            lb -= 1e-9
            ub += 1e-9
        # clamp extremely wide ranges
        if ub - lb > MAX_WIDTH:
            mid = 0.5 * (ub + lb)
            lb = max(mid - MAX_WIDTH/2, -MAX_ABS_BOUND)
            ub = min(mid + MAX_WIDTH/2,  MAX_ABS_BOUND)
        lb = max(lb, -MAX_ABS_BOUND)
        ub = min(ub,  MAX_ABS_BOUND)
        node_bounds[k] = (float(lb), float(ub))

    # optional debug: print SCCs that are larger than 1 or self-looped and their nodes/bounds
    if debug:
        for sid, comp in enumerate(sccs):
            if len(comp) > 1:
                print(f"[DEBUG] SCC {sid} size {len(comp)} damping {scc_damping.get(sid)} nodes={comp}")
                for n in comp:
                    print(f"       bound {n}: {node_bounds.get(n)}")
            else:
                n = comp[0]
                # check self-loop
                if any((in_n == n and out_n == n and conn.enabled) for (in_n, out_n), conn in genome.connections.items()):
                    print(f"[DEBUG] self-loop node {n} bound {node_bounds.get(n)}")

    return node_bounds

def get_worst_case_profile_via_mip_neat(
    genome,
    config,
    side,
    left_pad,
    alpha_delta,
    input_bounds=(0.0, 1.0),
    solver=None,
):
    """
    For a NEAT genome (single-output network), find worst-case GLOBAL profile (length n_agents)
    that maximizes the chosen 'violation' depending on side ("left" or "right").

    Returns: (violation_value (float), profile_list (list floats length n_agents), output_sum (float))
    """
    assert side in ("left", "right")

    # Basic sizes
    num_inputs = int(config.genome_config.num_inputs)  # n-1
    input_keys = list(config.genome_config.input_keys)  # e.g. [-4, -3, -2, -1]
    output_keys = list(config.genome_config.output_keys)
    assert len(output_keys) == 1, "This code assumes exactly one network output."
    output_node = output_keys[0]
    n_agents = num_inputs + 1

    # compute per-node M (bounds)
    node_bounds = compute_big_m(genome, config, input_bounds=input_bounds, scale=2.0, max_iters=int(1e4))

    # ----- compute safe M_global from node bounds -----
    if output_node in node_bounds:
        L_out, U_out = node_bounds[output_node]
    else:
        raise KeyError(f"Output node {output_node} not found in genome nodes.")

    # Post-activation output bounds (ReLU) -- compute from pre-activation results
    out_lower = max(0.0, float(L_out))
    out_upper = max(0.0, float(U_out))

    sum_min = n_agents * out_lower
    sum_max = n_agents * out_upper

    # required M to make both branches feasible:
    M_req = max(sum_max - 1.0, 1.0 - sum_min)

    # add a small slack and a sensible floor
    slack = 1e-3

    # --- additional safety: make sure global M dominates any internal node magnitudes ---
    max_abs_node = 0.0
    for (L, U) in node_bounds.values():
        if not (isfinite(L) and isfinite(U)):
            continue
        max_abs_node = max(max_abs_node, abs(L), abs(U))

    safety_floor = max_abs_node * max(1.0, n_agents) * 2.0

    # final global M: at least 1.0, at least M_req + slack, and at least the safety_floor
    M_global = float(max(1.0, M_req + slack, safety_floor))

    # ---------------------------------------------------

    # Build problem
    prob = LpProblem(f"{side}-error", LpMaximize)

    variables = {}  # key -> LpVariable

    # 1) global ordered bids: g_0 .. g_{n-1}
    for i in range(n_agents):
        v = LpVariable(f"g_{i}", lowBound=float(input_bounds[0]), upBound=float(input_bounds[1]))
        variables[("g", i)] = v
        # legacy alias
        variables[("input", i)] = v

    # ordering & simple bounds (redundant w/ bounds but explicit)
    prob += variables[("g", 0)] >= float(input_bounds[0])
    prob += variables[("g", n_agents - 1)] <= float(input_bounds[1])
    for i in range(n_agents - 1):
        prob += variables[("g", i)] <= variables[("g", i + 1)]

    # s variable and binary
    s = LpVariable("s", lowBound=None)
    s_bin = LpVariable("s_bin", cat="Binary")
    prob += s >= lpSum([variables[("g", i)] for i in range(n_agents)])
    prob += s >= 1.0
    prob += s <= lpSum([variables[("g", i)] for i in range(n_agents)]) + M_global * s_bin
    prob += s <= 1.0 + M_global * (1 - s_bin)

    # 2) For each agent, create node variables for each non-input node in genome
    compute_nodes = [k for k in genome.nodes.keys() if k not in input_keys]

    # determine orphan nodes (no enabled incoming connections)
    incoming_count = {n: 0 for n in compute_nodes}
    for (i, o), conn in genome.connections.items():
        if conn.enabled and o in incoming_count:
            incoming_count[o] += 1
    orphan_nodes = {n for n, cnt in incoming_count.items() if cnt == 0}

    # create per-agent node vars
    for a in range(n_agents):
        for node in compute_nodes:
            # raise if node not in bounds dict
            if node not in node_bounds:
                raise KeyError(f"Node {node} not found in node_bounds")
            L, U = node_bounds[node]
            # raise if bounds are not finite
            if not isfinite(L) or not isfinite(U):
                raise KeyError(f"Node {node} bounds not finite: L={L}, U={U}")
            if L > U:
                L, U = U, L

            # get bias from genome node
            node_gene = genome.nodes.get(node)
            bias = float(getattr(node_gene, "bias", 0.0))

            if node in orphan_nodes:
                # Orphan: do NOT mutate global node_bounds; declare fixed pre/out vars for this agent
                pre_name = f"pre_{a}_{node}"
                out_name = f"out_{a}_{node}"
                # pre fixed to bias
                variables[("pre", a, node)] = LpVariable(pre_name, lowBound=bias, upBound=bias)
                # out fixed to relu(bias)
                relu_out = float(max(0.0, bias))
                variables[("out", a, node)] = LpVariable(out_name, lowBound=relu_out, upBound=relu_out)
                # no z variable for orphan
                continue

            # CASE: pre-interval entirely <= 0 -> ReLU output ALWAYS 0 (no z)
            if U <= 0.0:
                variables[("pre", a, node)] = LpVariable(f"pre_{a}_{node}", lowBound=L, upBound=U)
                variables[("out", a, node)] = LpVariable(f"out_{a}_{node}", lowBound=0.0, upBound=0.0)
                # no z
                continue

            # CASE: pre-interval entirely >= 0 -> ReLU is identity (no z)
            if L >= 0.0:
                variables[("pre", a, node)] = LpVariable(f"pre_{a}_{node}", lowBound=L, upBound=U)
                # out's feasible range equals pre's range (ReLU identity)
                variables[("out", a, node)] = LpVariable(f"out_{a}_{node}", lowBound=L, upBound=U)
                # no z
                continue

            # General case L < 0 < U: need z binary and full big-M encoding
            variables[("pre", a, node)] = LpVariable(f"pre_{a}_{node}", lowBound=L, upBound=U)
            variables[("out", a, node)] = LpVariable(f"out_{a}_{node}", lowBound=0.0)
            variables[("z", a, node)] = LpVariable(f"z_{a}_{node}", cat="Binary")

    # Helper: map an in_node (NEAT node id) to the variable for this agent
    def in_node_var_for_agent(a, in_node):
        """
        If in_node is an input key (negative id), map to the corresponding global g_j for this agent:
          - find index idx among input_keys (0..num_inputs-1)
          - global index = idx if idx < agent else idx+1
        Else the in_node is an internal node id -> return variables[('out', a, in_node)]
        """
        if in_node in input_keys:
            idx = input_keys.index(in_node)
            global_idx = idx if idx < a else idx + 1
            return variables[("g", global_idx)]
        else:
            # internal node -> must exist
            return variables[("out", a, in_node)]

    # 3) Add linear pre-activation equalities per agent/node using genome.connections
    for a in range(n_agents):
        for node in compute_nodes:
            # gather incoming weighted terms
            terms = []
            for (in_n, out_n), conn in genome.connections.items():
                if not conn.enabled or out_n != node:
                    continue
                w = float(conn.weight)
                if abs(w) <= 1e-12:
                    continue
                term_var = in_node_var_for_agent(a, in_n)
                terms.append(w * term_var)
            # bias
            bias = float(getattr(genome.nodes.get(node), "bias", 0.0))
            pre_var = variables[("pre", a, node)]
            out_var = variables[("out", a, node)]
            # add equality pre == bias + sum(terms) or pre == bias (orphan)
            if len(terms) > 0:
                prob += pre_var == bias + lpSum(terms)
            else:
                # Orphan node: fix pre to bias and out to ReLU(bias)
                prob += pre_var == bias
                prob += out_var == max(0.0, bias)

    # 4) Add ReLU constraints depending on interval cases
    for a in range(n_agents):
        for node in compute_nodes:
            if node in orphan_nodes:
                # orphan node already handled (pre fixed to bias, out fixed to ReLU(bias))
                continue

            L, U = node_bounds.get(node)
            if not isfinite(L) or not isfinite(U):
                raise KeyError(f"Node {node} bounds not finite: L={L}, U={U}")
            if L > U:
                L, U = U, L

            pre_v = variables[("pre", a, node)]
            out_v = variables[("out", a, node)]

            # CASE: U <= 0 => out == 0 (already created as 0..0 bound), but add equality for clarity
            if U <= 0.0:
                prob += out_v == 0.0
                continue

            # CASE: L >= 0 => out == pre (ReLU is identity)
            if L >= 0.0:
                prob += out_v == pre_v
                continue

            # General case L < 0 < U: canonical big-M with z
            z_v = variables[("z", a, node)]

            # canonical ReLU encoding
            prob += out_v >= pre_v                 # out >= pre
            prob += out_v >= 0.0                   # out >= 0

            # big-M bounds using node-specific L,U
            # out <= pre - L*(1-z)
            prob += out_v <= pre_v - float(L) * (1 - z_v)
            # out <= U*z
            prob += out_v <= float(U) * z_v

            # LINK z and pre so z == 1 -> pre >= 0; z == 0 -> pre <= 0
            prob += pre_v <= float(U) * z_v
            prob += pre_v >= float(L) * (1 - z_v)

    # 5) Build sum of final outputs (sum over agents of the single network output node)
    if output_node not in compute_nodes:
        raise RuntimeError(f"Output node {output_node} not available among genome nodes.")

    sum_outputs = lpSum([variables[("out", a, output_node)] for a in range(n_agents)])

    # 6) Violation variable and objective expression
    violation = LpVariable("violation", lowBound=None)
    if side == "right":
        # use adjusted alpha (alpha - alpha_delta) to compute the RHS
        adj_alpha = alpha - alpha_delta
        prob += violation == sum_outputs - (n_agents - adj_alpha) * s
    else:  # side == "left"
        prob += violation == (n_agents - 1) * s - sum_outputs

    # Objective
    prob += violation
    prob.setObjective(violation)

    # Solve
    if solver is None:
        solver = PULP_CBC_CMD(msg=True, timeLimit=None)
    prob.solve(solver)

    if LpStatus[prob.status] not in ("Optimal", "Optimal Solution Found", "Integer Feasible"):
        # print a warning but continue to extract what we can
        prob.writeLP("debug_model.lp")
        print("Node Bounds:", node_bounds)
        input_keys_out = getattr(config.genome_config, "input_keys", None)
        print("Input nodes:", input_keys_out)
        print(genome)
        raise KeyError(f"[WARNING] solver status = {LpStatus[prob.status]} for side={side}")

    # Extract the global profile (length n_agents)
    profile = np.array([float(value(variables[("g", i)])) for i in range(n_agents)], dtype=float)

    # Extract objective (violation)
    err = float(value(violation))

    # Optionally extract sum_outputs value
    out_sum_val = float(value(sum_outputs))

    return err, profile, out_sum_val


def worst_case_analysis_neat(genome, config, alpha_delta, input_bounds=(0.0, 1.0)):
    """
    Run left & right worst-case MIP analyses and return:
      (wcp_left, wcp_right, error_left, error_right, total_error)
    where wcp_* are lists (length n_agents) and total_error = max(0,left)+max(0,right)
    """

    err_right, wcp_right, _ = get_worst_case_profile_via_mip_neat(
        genome, config, side="right", left_pad=0.0, alpha_delta=alpha_delta, input_bounds=input_bounds
    )

    err_left, wcp_left, _ = get_worst_case_profile_via_mip_neat(
        genome, config, side="left", left_pad=0.0, alpha_delta=alpha_delta, input_bounds=input_bounds
    )
    total_error = max(0.0, err_left) + max(0.0, err_right)
    return wcp_left, wcp_right, err_left, err_right, total_error
