#!/usr/bin/env python3
import matplotlib.pyplot as plt

# Read the file and calculate allocative ratios
ratios = []

with open('filenames_processed.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            number = float(line)
            allocative_ratio = 0.71428571428 - number
            ratios.append(allocative_ratio)

# Plot the values over time
plt.figure(figsize=(12, 6))
plt.plot(range(len(ratios)), ratios, 'b-', linewidth=1, label='Current Ratio')

# Add max achieved line
max_ratio = max(ratios)
plt.axhline(y=max_ratio, color='r', linestyle='--', linewidth=2, label=f'Max: {max_ratio:.6f}')

plt.xlabel('Model Index')
plt.ylabel('Allocative Efficiency Ratio')
plt.title('Allocative Efficiency Ratios Over Time')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('allocative_ratios_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Plotted {len(ratios)} ratios")
print(f"Max ratio: {max(ratios):.8f}")
print(f"Final ratio: {ratios[-1]:.8f}")
