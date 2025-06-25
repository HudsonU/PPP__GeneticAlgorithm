import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from hypernetwork import Hypernetwork
from model import FunctionalR  # Import the FunctionalR class

# Define a more complex target network using FunctionalR
target_network = FunctionalR([1, 10, 1])  # Input size 1, hidden layer size 10, output size 1

# Verify the parameter shapes
print("Target network parameter shapes:", target_network.get_parameter_shapes())
total_params = 0
for layer in target_network.get_parameter_shapes():
    weight_shape, bias_shape = layer
    layer_params = np.prod(weight_shape) + np.prod(bias_shape)
    print(f"Layer: weight shape {weight_shape}, bias shape {bias_shape}, total params: {layer_params}")
    total_params += layer_params
print(f"Total parameters needed: {total_params}")

# Create the hypernetwork with target network integration
hypernetwork = Hypernetwork(
    hidden_layers=[32, 16],  # Increased size for more complex target network
    activation='relu',
    target_network=target_network
)

# Loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(hypernetwork.parameters(), lr=1e-3)

def train_step(x, y):
    optimizer.zero_grad()
    
    # Generate weights from the hypernetwork and get parameters structured for the target network
    params = hypernetwork.get_parameters_for_target()
    
    # Update target network to use generated parameters
    target_network.update_params(params)
    
    # Forward pass through target network
    pred = target_network(x)
    
    # Calculate loss
    loss = loss_fn(pred, y)
    
    # Backpropagate
    loss.backward()
    
    # Debugging: Check if gradients are flowing to hypernetwork
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in hypernetwork.parameters())
    if not has_grad:
        print("WARNING: No gradients flowing to hypernetwork!")
    
    optimizer.step()
    
    return loss.item()

def main():
    # Example data for "predict next number" task
    batch_size = 32
    
    # Training loop
    epochs = 15000
    losses = []
    
    for epoch in range(epochs):
        # Generate random integers between 1 and 1000
        x = torch.randint(1, 1001, (batch_size, 1), dtype=torch.float32)
        # Target is input + 1
        y = x + 1.0
        
        loss = train_step(x, y)
        losses.append(loss)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.8f}")
    
    # Plot the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss')
    plt.yscale('log')  # Log scale often helps visualize convergence
    plt.grid(True)
    plt.savefig('training_loss.png')
    
    # Test the model with some examples
    print("\nTesting the model:")
    
    # Generate test data
    test_inputs = torch.tensor([[1.0], [5.0], [23.0], [42.0], [100.0], [500.0]], dtype=torch.float32)
    test_targets = test_inputs + 1.0
    
    # Generate parameters using hypernetwork
    with torch.no_grad():
        # Get parameters from hypernetwork
        params = hypernetwork.get_parameters_for_target()
        
        # Update target network to use these parameters
        target_network.update_params(params)
        
        # Test each input using the target network
        predictions = target_network(test_inputs)
        for i in range(len(test_inputs)):
            print(f"Input: {test_inputs[i][0]:.1f}, Predicted: {predictions[i][0]:.4f}, Expected: {test_targets[i][0]:.1f}")
        
        # Print the learned parameters (first layer only as an example)
        print(f"\nLearned parameters (first layer):")
        w1, b1 = params[0]
        print(f"First layer weight shape: {w1.shape}")
        print(f"First layer bias shape: {b1.shape}")

        # Print the learned parameters (second layer)
        w2, b2 = params[1]
        print(f"\nLearned parameters (second layer):")
        print(f"Second layer weight shape: {w2.shape}")
        print(f"Second layer bias shape: {b2.shape}")
        
        # Optional: Save the trained hypernetwork and target network
        torch.save(hypernetwork.state_dict(), 'hypernetwork.pt')
        print("Model saved to 'hypernetwork.pt'")

if __name__ == "__main__":
    main()