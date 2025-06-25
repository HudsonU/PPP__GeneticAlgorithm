#!/usr/bin/env python3

import numpy as np
from model import FunctionalR

def calculate_target_params(architecture):
    """Calculate total parameters for a given architecture."""
    total_params = 0
    for i in range(len(architecture) - 1):
        # Weight parameters: output_size x input_size
        weight_params = architecture[i+1] * architecture[i] 
        # Bias parameters: output_size
        bias_params = architecture[i+1]
        layer_params = weight_params + bias_params
        total_params += layer_params
        print(f"Layer {i+1}: {architecture[i]} -> {architecture[i+1]}")
        print(f"  Weight: {architecture[i+1]}x{architecture[i]} = {weight_params}")
        print(f"  Bias: {architecture[i+1]} = {bias_params}")
        print(f"  Layer total: {layer_params}")
    
    return total_params

def calculate_hypernetwork_params(hidden_layers, target_params):
    """Calculate total parameters in hypernetwork."""
    total_params = 0
    
    # Input layer (1 -> first hidden)
    input_size = 1  # Dummy input
    for i, hidden_size in enumerate(hidden_layers):
        weight_params = hidden_size * input_size
        bias_params = hidden_size
        layer_params = weight_params + bias_params
        total_params += layer_params
        print(f"  HN Layer {i+1}: {input_size} -> {hidden_size}")
        print(f"    Weight: {hidden_size}x{input_size} = {weight_params}")
        print(f"    Bias: {hidden_size} = {bias_params}")
        print(f"    Layer total: {layer_params}")
        input_size = hidden_size
    
    # Output layer (last hidden -> target_params)
    weight_params = target_params * input_size
    bias_params = target_params
    layer_params = weight_params + bias_params
    total_params += layer_params
    print(f"  HN Output Layer: {input_size} -> {target_params}")
    print(f"    Weight: {target_params}x{input_size} = {weight_params}")
    print(f"    Bias: {target_params} = {bias_params}")
    print(f"    Layer total: {layer_params}")
    
    return total_params

def find_architecture_for_121_params():
    """Find target architectures that produce 121 parameters."""
    print("Finding architectures that produce 121 parameters:")
    print("="*50)
    
    # Check various architectures
    architectures_to_test = [
        [4, 5, 1],      # Small
        [4, 10, 1],     # Medium  
        [4, 20, 1],     # Current
        [4, 25, 1],     # Larger
        [4, 30, 1],     # Even larger
        [5, 4, 1],      # Different input size
        [3, 8, 1],      # Different input size
        [4, 4, 4, 1],   # Two hidden layers
        [4, 6, 2, 1],   # Two hidden layers
        [2, 10, 1],     # Minimal input
    ]
    
    target_121_architectures = []
    
    for arch in architectures_to_test:
        print(f"\nArchitecture: {arch}")
        params = calculate_target_params(arch)
        print(f"Total parameters: {params}")
        
        if params == 121:
            target_121_architectures.append(arch)
            print("*** MATCHES 121 PARAMETERS! ***")
        
        print("-" * 30)
    
    return target_121_architectures

def suggest_hypernetwork_sizes(target_params=121):
    """Suggest hypernetwork configurations for 121 target parameters."""
    print(f"\nSuggested hypernetwork sizes for {target_params} target parameters:")
    print("="*60)
    
    # Test various hypernetwork configurations
    hn_configs = [
        [64],           # Single layer
        [128],          # Single layer, larger
        [32, 16],       # Two layers, small
        [64, 32],       # Two layers, medium
        [128, 64],      # Two layers, large
        [256, 128],     # Two layers, very large (current)
        [512, 256],     # Two layers, huge
        [64, 32, 16],   # Three layers
        [128, 64, 32],  # Three layers, larger
        [256, 128, 64], # Three layers, very large
    ]
    
    for hn_config in hn_configs:
        print(f"\nHypernetwork config: {hn_config}")
        hn_params = calculate_hypernetwork_params(hn_config, target_params)
        ratio = hn_params / target_params
        print(f"Total HN parameters: {hn_params}")
        print(f"Parameter ratio (HN/Target): {ratio:.2f}")
        
        # Efficiency assessment
        if ratio < 2:
            efficiency = "Very efficient"
        elif ratio < 5:
            efficiency = "Efficient"
        elif ratio < 10:
            efficiency = "Moderate"
        elif ratio < 20:
            efficiency = "Less efficient"
        else:
            efficiency = "Inefficient"
            
        print(f"Efficiency: {efficiency}")
        print("-" * 40)

if __name__ == "__main__":
    # Find architectures that produce exactly 121 parameters
    target_121_archs = find_architecture_for_121_params()
    
    if target_121_archs:
        print(f"\n\nArchitectures with exactly 121 parameters:")
        for arch in target_121_archs:
            print(f"  {arch}")
    else:
        print(f"\n\nNo tested architectures produce exactly 121 parameters.")
        print("You may need to try other combinations or use a close approximation.")
    
    # Suggest hypernetwork sizes for 121 parameters
    suggest_hypernetwork_sizes(121)
    
    # Also show current architecture parameter count
    current_arch = [4, 20, 1]
    print(f"\n\nCurrent architecture analysis:")
    print(f"Architecture: {current_arch}")
    current_params = calculate_target_params(current_arch)
    print(f"Current total parameters: {current_params}")
    
    if current_params != 121:
        print(f"Difference from 121: {current_params - 121}")
