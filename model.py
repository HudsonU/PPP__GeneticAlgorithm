import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
import torch.nn.functional as F
import numpy
from copy import deepcopy
from string import ascii_lowercase
from pulp import (
    LpVariable,
    LpMaximize,
    LpProblem,
    lpSum,
    GUROBI_CMD,
    value,
    LpStatus,
    LpMinimize,
)
from torch.nn.functional import softmax
from settings import n, device
from utility import (
    s,
    alpha,
    kick,
)

# from random import randrange

class R(nn.Module):
    def __init__(self, shapes, initial_state_dict=None):
        super().__init__()
        # Example shapes:
        # [5, 10, 1] = 5 input nodes, 10 hidden nodes, 1 output node
        # [5, 10] = 5 input nodes, 10 output nodes (no hidden layer)
        # [5, 20, 10, 1] = 5 input nodes, two hidden layers, 1 output node
        stacks = []
        for i in range(len(shapes) - 1):
            stacks.append(nn.Linear(shapes[i], shapes[i + 1]))
            stacks.append(nn.ReLU())
        # remove the last ReLu
        stacks = stacks[:-1]
        self.linear_relu_stack = nn.Sequential(*stacks)
        self.shapes = shapes
        if initial_state_dict is not None:
            self.load_state_dict(initial_state_dict)
        self.initial_state_dict = deepcopy(self.state_dict())  

    def restore_initial_state_dict(self):
        try:
            for i in range(0, len(self.linear_relu_stack), 2):
                prune.remove(self.linear_relu_stack[i], "weight")
        except ValueError:
            pass
        self.load_state_dict(self.initial_state_dict)

    def restore_initial_state_dict_copy(self):
        return R(self.shapes, self.initial_state_dict)

    def copy_state_dict(self, r):
        try:
            for i in range(0, len(self.linear_relu_stack), 2):
                prune.remove(self.linear_relu_stack[i], "weight")
        except ValueError:
            pass
        self.load_state_dict(r.state_dict())

    def forward(self, x):
        return self.linear_relu_stack(x)

    def generate_used(self):
        used = [None] * len(self.shapes)
        for i in range(len(self.shapes)):
            used[i] = [False] * self.shapes[i]
        used[-1] = [True]
        used_index = -2
        for i in reversed(range(0, len(self.linear_relu_stack), 2)):
            current_mask = self.linear_relu_stack[i].weight_mask
            for j in range(len(current_mask)):
                for k in range(len(current_mask[j])):
                    if current_mask[j][k] == 1 and used[used_index + 1][j]:
                        used[used_index][k] = True
            used_index -= 1
        return used


class FunctionalR(nn.Module):
    """
    A functional version of the R class that can use either nn.Linear modules
    or F.linear operations with external parameters.
    """
    def __init__(self, shapes, initial_state_dict=None):
        super().__init__()
        self.shapes = shapes
        
        # Create standard nn.Linear layers for default behavior
        self.layers = nn.ModuleList()
        for i in range(len(shapes) - 1):
            self.layers.append(nn.Linear(shapes[i], shapes[i + 1]))
        
        # Flag to determine whether to use external parameters
        self.using_external_params = False
        self.external_params = None
        self.flat_params = None
        
        # Pre-compute parameter indices for efficient slicing
        self._compute_param_indices()
        
        # Load initial state if provided
        if initial_state_dict is not None:
            self.load_state_dict(initial_state_dict)
        
        # Store initial state for reset capability
        self.initial_state_dict = deepcopy(self.state_dict())
    
    def _compute_param_indices(self):
        """Pre-compute indices for slicing flat parameter tensor"""
        self.param_indices = []
        start_idx = 0
        
        for i in range(len(self.shapes) - 1):
            # Weight tensor shape and size
            weight_shape = (self.shapes[i+1], self.shapes[i])
            weight_size = self.shapes[i+1] * self.shapes[i]
            weight_end = start_idx + weight_size
            
            # Bias tensor shape and size  
            bias_shape = (self.shapes[i+1],)
            bias_size = self.shapes[i+1]
            bias_end = weight_end + bias_size
            
            # Store indices and shapes for this layer
            self.param_indices.append({
                'weight_start': start_idx,
                'weight_end': weight_end,
                'weight_shape': weight_shape,
                'bias_start': weight_end,
                'bias_end': bias_end,
                'bias_shape': bias_shape
            })
            
            start_idx = bias_end
    
    def get_parameter_shapes(self):
        """
        Returns the shapes of parameters needed for this network.
        
        Returns:
            list: List of tuples (weight_shape, bias_shape) for each layer
        """
        param_shapes = []
        for i in range(len(self.shapes) - 1):
            weight_shape = (self.shapes[i+1], self.shapes[i])
            bias_shape = (self.shapes[i+1],)
            param_shapes.append((weight_shape, bias_shape))
        return param_shapes
    
    def restore_initial_state_dict(self):
        """Reset parameters to initial values."""
        self.load_state_dict(self.initial_state_dict)
        self.using_external_params = False
    
    def update_params_flat(self, flat_params):
        """Update parameters from flat parameter tensor with pre-computed indices"""
        if self.param_indices is None:
            self._compute_param_indices()
            
        # Get device from first layer parameter
        target_device = next(self.parameters()).device
        
        # Move flat tensor to device once
        if flat_params.device != target_device:
            flat_params = flat_params.to(target_device)
        
        # Initialize storage for weights and biases if not exists
        if not hasattr(self, 'weights'):
            self.weights = [None] * len(self.param_indices)
        if not hasattr(self, 'biases'):
            self.biases = [None] * len(self.param_indices)
        
        for i, indices in enumerate(self.param_indices):
            # Extract weight and bias using pre-computed indices
            weight = flat_params[indices['weight_start']:indices['weight_end']].view(indices['weight_shape'])
            bias = flat_params[indices['bias_start']:indices['bias_end']].view(indices['bias_shape'])
            
            # Store parameters
            self.weights[i] = weight
            self.biases[i] = bias
            
        self.params_updated = True
        self.using_external_params = True
    
    def forward(self, x):
        """
        Forward pass using either built-in parameters or externally provided ones.
        
        Args:
            x: Input tensor
        """
        if not self.using_external_params:
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:
                    x = F.relu(x)
        else:
            if hasattr(self, 'weights') and hasattr(self, 'biases'):
                # Ensure input dtype matches weights dtype
                x = x.to(self.weights[0].dtype)
                for i in range(len(self.weights)):
                    x = F.linear(x, self.weights[i], self.biases[i])
                    if i < len(self.weights) - 1:
                        x = F.relu(x)
            elif self.external_params is not None:
                # Ensure input dtype matches weights dtype
                x = x.to(self.external_params[0][0].dtype)
                for i, (weight, bias) in enumerate(self.external_params):
                    x = F.linear(x, weight, bias)
                    if i < len(self.external_params) - 1:
                        x = F.relu(x)
        return x
    
    def copy_state_dict(self, r):
        """Copy parameters from another R instance."""
        self.load_state_dict(r.state_dict())
        self.using_external_params = False
    
    def get_worst_case_profile_via_mip(self, side, left_pad=0, alpha_delta=0):
        M = 1000  # this is the "big constant" for MIP -- larger M causes bug
        assert side in ["left", "right"]
        self.eval()

        weights = []
        biases = []
        
        # Extract weights and biases from the current model
        if not self.using_external_params:
            # Use the standard layers
            for layer in self.layers:
                weights.append(layer.weight.data.cpu().numpy())
                biases.append(layer.bias.data.cpu().numpy())
        else:
            # Use the external parameters
            if hasattr(self, 'weights') and hasattr(self, 'biases'):
                # New flat parameter format
                for i in range(len(self.weights)):
                    weights.append(self.weights[i].data.cpu().numpy())
                    biases.append(self.biases[i].data.cpu().numpy())
            elif self.external_params is not None:
                # Legacy format
                for weight, bias in self.external_params:
                    weights.append(weight.data.cpu().numpy())
                    biases.append(bias.data.cpu().numpy())
        
        for i, weight in enumerate(weights):
            assert weight.shape[0] == self.shapes[i + 1]
            assert weight.shape[1] == self.shapes[i]
        for i, bias in enumerate(biases):
            assert bias.shape[0] == self.shapes[i + 1]

        prob = LpProblem(f"{side}-error", LpMaximize)
        variables = {}

        for agent in range(n):
            variables[agent] = LpVariable(str(agent))
        prob += variables[0] >= 0
        prob += variables[n - 1] <= 1
        for i in range(n - 1):
            prob += variables[i] <= variables[i + 1]

        variables["s"] = LpVariable("s")
        variables["s-binary"] = LpVariable("s-binary", cat="Binary")
        prob += variables["s"] >= lpSum([variables[i] for i in range(n)])
        prob += variables["s"] >= 1
        prob += (
            variables["s"]
            <= lpSum([variables[i] for i in range(n)]) + M * variables["s-binary"]
        )
        prob += variables["s"] <= 1 + M * (1 - variables["s-binary"])

        for agent in range(n):
            for i, size in enumerate(self.shapes):
                for j in range(size):
                    if i < len(self.shapes) - 1:
                        variables[(agent, i, j, "nonneg")] = LpVariable(
                            f"{agent},{i},{j},nonneg"
                        )
                    if 0 < i < len(self.shapes) - 1:
                        variables[(agent, i, j, "binary")] = LpVariable(
                            f"{agent},{i},{j},binary", cat="Binary"
                        )
                    if i > 0:
                        variables[(agent, i, j)] = LpVariable(f"{agent},{i},{j}")

        for agent in range(n):
            for i in range(n - 1):
                if i < agent:
                    prob += variables[(agent, 0, i, "nonneg")] == variables[i]
                if i >= agent:
                    prob += variables[(agent, 0, i, "nonneg")] == variables[i + 1]

        for agent in range(n):
            for i in range(1, len(self.shapes) - 1):
                for j in range(self.shapes[i]):
                    prob += (
                        variables[(agent, i, j, "nonneg")] >= variables[(agent, i, j)]
                    )
                    prob += variables[(agent, i, j, "nonneg")] >= 0
                    prob += (
                        variables[(agent, i, j, "nonneg")]
                        <= variables[(agent, i, j)]
                        + M * variables[(agent, i, j, "binary")]
                    )
                    prob += variables[(agent, i, j, "nonneg")] <= M * (
                        1 - variables[(agent, i, j, "binary")]
                    )

        for agent in range(n):
            for i in range(1, len(self.shapes)):
                for j in range(self.shapes[i]):
                    prob += (
                        variables[(agent, i, j)]
                        == lpSum(
                            [
                                variables[(agent, i - 1, k, "nonneg")]
                                * weights[i - 1][j][k]
                                for k in range(self.shapes[i - 1])
                                if abs(weights[i - 1][j][k]) > 0.000001
                            ]
                        )
                        + biases[i - 1][j]
                    )

        if side == "right":
            prob += (
                lpSum(
                    [variables[(agent, len(self.shapes) - 1, 0)] for agent in range(n)]
                )
                + left_pad
                - (n - (alpha - alpha_delta)) * variables["s"]
            )
        if side == "left":
            prob += (n - 1) * variables["s"] - lpSum(
                [variables[(agent, len(self.shapes) - 1, 0)] for agent in range(n)]
            )

        prob.solve(
            GUROBI_CMD(
                msg=False,
                options=[
                    ("Method", 2),
                    ("Heuristics", 0.001),
                    ("MIPFocus", 2),
                    ("VarBranch", 1),
                    ("PreDual", 1),
                    ("Presolve", 2),
                ],
            )
        )
        if LpStatus[prob.status] != "Optimal":
            print(f"ERROR: {LpStatus[prob.status]}")
        error = value(prob.objective)
        wcp = [variables[agent].varValue for agent in range(n)]
        mip_received = [
            value(variables[(agent, len(self.shapes) - 1, 0)]) for agent in range(n)
        ]
        for agent in range(n):
            nn_received = self(torch.tensor(kick(agent, wcp), device=device)).item()
            if abs(nn_received - mip_received[agent]) / abs(nn_received) > 0.001:
                print(
                    f"ERROR: for {wcp}, nn says {agent} received {nn_received}, but mip says {mip_received[agent]}"
                )
        # print(f"side={side}, wcp={wcp}, error={error}")
        # print(f"{(n-1)*s(wcp)} <= {sum(mip_received)} <= {(n-alpha)*s(wcp)}")
        return error, wcp

    def worst_case_analysis(self, alpha_delta=0):
        error_left, wcp_left = self.get_worst_case_profile_via_mip(
            "left", alpha_delta=alpha_delta
        )
        error_right, wcp_right = self.get_worst_case_profile_via_mip(
            "right", alpha_delta=alpha_delta
        )
        return (
            wcp_left,
            wcp_right,
            error_left,
            error_right,
            max(0, error_left) + max(0, error_right) + alpha_delta,
        )

# Example usage
if __name__ == "__main__":
    # Test the FunctionalR class
    shapes = [5, 10, 1]  # 5 input, 10 hidden, 1 output