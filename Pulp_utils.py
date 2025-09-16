# Pulp_utils_neat.py
import numpy as np
import neat
import time
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

def compute_big_m(genome, config, input_bounds, scale=1.0, max_iters=10000):
    """
    Conservative per-node Big-M estimation for a NEAT genome (ReLU nets).
    Iteratively propagate interval bounds from inputs through connections
    until convergence or max_iters. Returns dict node_id -> (L,U).
    """
    input_keys = list(getattr(config.genome_config, "input_keys", []))
    output_keys = list(getattr(config.genome_config, "output_keys", []))

    # collect all nodes mentioned
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
            node_bounds[k] = (-10.0, 10.0)  # wide initial range

    # iterative relax / propagation
    for it in range(max_iters):
        changed = False
        for node in all_nodes:
            if node in input_keys:
                continue
            node_gene = genome.nodes.get(node, None)
            bias = float(getattr(node_gene, "bias", 0.0)) if node_gene is not None else 0.0
            lb = bias
            ub = bias

            for (in_node, out_node), conn in genome.connections.items():
                if not conn.enabled or out_node != node:
                    continue
                w = float(conn.weight)
                in_lb, in_ub = node_bounds.get(in_node, (float(input_bounds[0]), float(input_bounds[1])))
                if w >= 0:
                    lb += w * in_lb
                    ub += w * in_ub
                else:
                    lb += w * in_ub
                    ub += w * in_lb

            prev_lb, prev_ub = node_bounds[node]
            new_lb, new_ub = min(lb, ub), max(lb, ub)
            if not np.isclose(prev_lb, new_lb) or not np.isclose(prev_ub, new_ub):
                node_bounds[node] = (new_lb, new_ub)
                changed = True

        if not changed:
            break

        # ensure bounds are consistent and slightly slack for LP feasibility
        for k, (lb, ub) in node_bounds.items():
            if lb > ub:
                lb, ub = ub, lb
            if not isfinite(lb) or not isfinite(ub):
                lb, ub = -10.0, 10.0
            # give slight slack if too tight
            if abs(ub - lb) < 1e-6:
                lb -= 1e-5
                ub += 1e-5
            # make sure 0 is within bounds for ReLU
            if k not in input_keys:
                lb = min(lb, 0.0)
                ub = max(ub, 0.0)
            node_bounds[k] = (float(lb), float(ub))

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
    num_inputs = int(config.genome_config.num_inputs)         # n-1
    input_keys = list(config.genome_config.input_keys)        # e.g. [-4, -3, -2, -1]
    output_keys = list(config.genome_config.output_keys)
    assert len(output_keys) == 1, "This code assumes exactly one network output."
    output_node = output_keys[0]
    n_agents = num_inputs+1

    # compute per-node M
    node_bounds  = compute_big_m(genome, config, input_bounds=input_bounds, scale=1.0, max_iters=int(1e4))
    
    # ----- compute safe M_global from node bounds -----

    # get signed bounds for the network output
    if output_node in node_bounds:
        L_out, U_out = node_bounds[output_node]
    else:
        # fallback if output node was missing — be conservative
        L_out, U_out = -1.0, 1.0

    # ReLU output bounds are post-activation:
    out_lower = max(0.0, float(L_out))
    out_upper = max(0.0, float(U_out))

    # sum over agents (network is the same per-agent so scale by n_agents)
    sum_min = n_agents * out_lower
    sum_max = n_agents * out_upper

    # required M to make both branches feasible:
    M_req = max(sum_max - 1.0, 1.0 - sum_min)

    # add a small slack and a sensible floor
    slack = 1e-3
    M_global = float(max(1.0, M_req + slack))
    # ---------------------------------------------------
    # a safe global big-M for sums / binary tricks
    # M_global = max(M_dict.values()) if len(M_dict) > 0 else 1000.0
    # M_global = max(M_global, 100.0)

    # Build problem
    prob = LpProblem(f"{side}-error", LpMaximize)

    variables = {}  # key -> LpVariable

    # 1) global ordered bids: g_0 .. g_{n-1}
    for i in range(n_agents):
        v = LpVariable(f"g_{i}", lowBound=float(input_bounds[0]), upBound=float(input_bounds[1]))
        variables[("g", i)] = v
        # create alias so older code using ('input', i) works
        variables[("input", i)] = v

    # ordering & simple bounds (redundant w/ bounds but explicit)
    prob += variables[("g", 0)] >= float(input_bounds[0])
    prob += variables[("g", n_agents - 1)] <= float(input_bounds[1])
    for i in range(n_agents - 1):
        prob += variables[("g", i)] <= variables[("g", i + 1)]

    # s variable and binary (copied logic from your earlier model)
    s = LpVariable("s", lowBound=None)
    s_bin = LpVariable("s_bin", cat="Binary")
    prob += s >= lpSum([variables[("g", i)] for i in range(n_agents)])
    prob += s >= 1.0
    prob += s <= lpSum([variables[("g", i)] for i in range(n_agents)]) + M_global * s_bin
    prob += s <= 1.0 + M_global * (1 - s_bin)

    # 2) For each agent, create node variables for each non-input node in genome
    #    We'll treat all genome.nodes that are *not* input_keys as compute nodes.
    compute_nodes = [k for k in genome.nodes.keys() if k not in input_keys]

    # create per-agent node vars
    for a in range(n_agents):
        for node in compute_nodes:
            L, U = node_bounds.get(node, (-1.0, 1.0))
            # defensive:
            if not isfinite(L) or not isfinite(U):
                L, U = -1.0, 1.0
            if L > U:
                L, U = U, L
            # pre-activation (bounded by -M_node..M_node)
            variables[("pre", a, node)] = LpVariable(f"pre_{a}_{node}", lowBound=L, upBound=U)
            # ReLU output (non-negative)
            variables[("out", a, node)] = LpVariable(f"out_{a}_{node}", lowBound=0.0)
            # binary switch for ReLU
            variables[("z", a, node)] = LpVariable(f"z_{a}_{node}", cat="Binary")

    eps = 1e-12

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
                if not conn.enabled:
                    continue
                if out_n != node:
                    continue
                w = float(conn.weight)
                if abs(w) <= eps:
                    continue
                term_var = in_node_var_for_agent(a, in_n)
                terms.append(w * term_var)
            # bias
            node_gene = genome.nodes.get(node, None)
            bias = float(getattr(node_gene, "bias", 0.0)) if node_gene is not None else 0.0
            # add equality pre == bias + sum(terms)
            if len(terms) > 0:
                prob += variables[("pre", a, node)] == bias + lpSum(terms)
            else:
                prob += variables[("pre", a, node)] == bias

    # 4) Add ReLU big-M encoding per agent/node
    # For each agent (position in the global profile) and each compute node,
    # add the 4 big-M linear constraints that encode out = ReLU(pre).
    for a in range(n_agents):
        for node in compute_nodes:
            # Per-node big-M value (estimated by compute_big_m). Use 1.0 as fallback.
            L, U = node_bounds.get(node, (-1.0, 1.0))
            # defensive:
            if not isfinite(L) or not isfinite(U):
                L, U = -1.0, 1.0
            if L > U:
                L, U = U, L

            # Shortcut references to the MILP variables for readability.
            pre_v = variables[("pre", a, node)]   # pre-activation variable (can be negative)
            out_v = variables[("out", a, node)]   # post-ReLU output (must be >= 0)
            z_v = variables[("z", a, node)]       # binary indicator: approx. 1 when pre<=0, 0 when pre>0 (or vice versa depending on inequalities)

            # 1) out >= pre
            prob += out_v >= pre_v

            # 2) out >= 0
            prob += out_v >= 0

            # 3) out <= pre - L * (1 - z)
            prob += out_v <= pre_v - float(L) * (1 - z_v)

            # 4) out <= U * z
            prob += out_v <= float(U) * z_v


    # 5) Build sum of final outputs (sum over agents of the single network output node)
    #    Note: assume the genome's single output node is 'output_node' and that node is included in compute_nodes
    if output_node not in compute_nodes:
        raise RuntimeError(f"Output node {output_node} not available among genome nodes.")

    sum_outputs = lpSum([variables[("out", a, output_node)] for a in range(n_agents)])

    # 6) Violation variable and objective expression (explicit variable so you can value() it)
    violation = LpVariable("violation", lowBound=None)
    if side == "right":
        # violation == sum_outputs + left_pad - (n_agents - (alpha - alpha_delta)) * s
        prob += violation == sum_outputs - (n_agents - alpha) * s
    else:  # side == "left"
        prob += violation == (n_agents - 1) * s - sum_outputs

    prob += violation  # objective: maximize violation

    # Solve
    if solver is None:
        solver = PULP_CBC_CMD(msg=True, timeLimit=None)
    prob.solve(solver)

    if LpStatus[prob.status] not in ("Optimal", "Optimal Solution Found", "Integer Feasible"):
        # print a warning but continue to extract what we can
        print(f"[WARNING] solver status = {LpStatus[prob.status]} for side={side}")
        prob.writeLP("debug_model.lp")
        input_keys = config.genome_config.input_keys
        print("Input nodes:", input_keys)
        print(genome)
        os.quit()

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
    where wcp_* are lists (length n_agents) and total_error = max(0,left)+max(0,right)+alpha_delta
    """
    err_left, wcp_left, _ = get_worst_case_profile_via_mip_neat(
        genome, config, side="left", left_pad=0.0, alpha_delta=alpha_delta, input_bounds=input_bounds
    )
    err_right, wcp_right, _ = get_worst_case_profile_via_mip_neat(
        genome, config, side="right", left_pad=0.0, alpha_delta=alpha_delta, input_bounds=input_bounds
    )
    total_error = max(0.0, err_left) + max(0.0, err_right) #+ float(alpha_delta)
    return wcp_left, wcp_right, err_left, err_right, total_error
