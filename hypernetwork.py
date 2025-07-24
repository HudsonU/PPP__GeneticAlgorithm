import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this import
import numpy as np

class Hypernetwork(nn.Module):
    """
    A Hypernetwork that generates parameters for a target network without requiring input.
    
    Args:
        output_size (int): Number of parameters the hypernetwork should produce.
        hidden_layers (list): List of integers specifying the number of nodes in each hidden layer.
        activation (nn.Module or str, optional): Activation function to use. Default: nn.ReLU().
        output_activation (nn.Module or str, optional): Activation for output layer. Default: None.
        target_network (nn.Module, optional): Target network to extract layer shapes from.
    """
    def __init__(self, output_size=None, hidden_layers=[16, 8], activation=nn.ReLU(), 
                 output_activation=None, target_network=None):
        super(Hypernetwork, self).__init__()
        
        # If target_network is provided, extract output_size from it
        if target_network is not None and hasattr(target_network, 'get_parameter_shapes'):
            self.target_shapes = target_network.get_parameter_shapes()
            # Calculate total parameters needed
            total_params = 0
            for layer in self.target_shapes:
                weight_shape, bias_shape = layer
                total_params += np.prod(weight_shape) + np.prod(bias_shape)
            output_size = total_params
        else:
            assert output_size is not None, "Either output_size or target_network must be provided"
            self.target_shapes = None
        
        # Build layers
        layers = []
        input_size = 1  # Dummy input size since no input is required
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(activation if isinstance(activation, nn.Module) else self._get_activation(activation))
            input_size = hidden_size
        
        # Add output layer
        layers.append(nn.Linear(input_size, output_size))
        
        # Add output activation if specified
        if output_activation is not None:
            layers.append(output_activation if isinstance(output_activation, nn.Module) 
                         else self._get_activation(output_activation))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)  # Ensure proper initialization
    
    def _get_activation(self, activation_name):
        """Convert activation function name to instance."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
        }
        return activations.get(activation_name.lower(), nn.ReLU())
    
    def _init_weights(self, m):
        """Initialize weights using Xavier initialization for better stability."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    def forward(self):
        """Forward pass through the network."""
        dummy_input = torch.ones(1, 1, device=next(self.parameters()).device)  # Dummy input to trigger the forward pass on same device as model
        return self.model(dummy_input).squeeze(0)  # Remove batch dimension
    
    @staticmethod
    def split_params_for_layers(flat_params, layer_shapes):
        """
        Splits a flat parameter tensor into weights and biases for multiple layers.
        
        Args:
            flat_params (Tensor): Flat tensor of parameters
            layer_shapes (list): List of tuples [(weight_shape, bias_shape), ...] for each layer
            
        Returns:
            list: List of tuples (weight, bias) for each layer
        """
        params_list = []
        start_idx = 0
        
        for layer_shape in layer_shapes:
            weight_shape, bias_shape = layer_shape
            
            # Calculate number of parameters for weight and bias
            weight_size = np.prod(weight_shape)
            bias_size = np.prod(bias_shape)
            
            # Extract and reshape weight
            weight_end = start_idx + weight_size
            weight = flat_params[start_idx:weight_end].reshape(weight_shape)
            
            # Extract and reshape bias
            bias_end = weight_end + bias_size
            bias = flat_params[weight_end:bias_end].reshape(bias_shape)
            
            # Update start index for next layer
            start_idx = bias_end
            
            # Add to output list
            params_list.append((weight, bias))
            
        return params_list

    def get_parameters_for_target(self):
        """
        Generate parameters and split them for the target network.
        
        Returns:
            list: List of tuples (weight, bias) for each layer
        """
        flat_params = self.forward()
        if self.target_shapes is not None:
            return self.split_params_for_layers(flat_params, self.target_shapes)
        return flat_params
    
    def get_flat_parameters(self):
        """
        Generate flat parameters directly without splitting.
        
        Returns:
            Tensor: Flat parameter tensor
        """
        return self.forward()