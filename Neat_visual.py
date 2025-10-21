import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import networkx as nx
import threading
import neat
import time

class NetVisualiser:
    def __init__(self, config):
        self.root = tk.Tk()
        self.root.title("NEAT Network Viewer")

        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.config = config

        # Run Tkinter in a separate thread
        threading.Thread(target=self.root.mainloop, daemon=True).start()

    def update(self, genome):
        self.ax.clear()
        G, pos = feed_forward_layout(genome, self.config)

        node_colors = []
        input_nodes = [i for i in range(-self.config.genome_config.num_inputs, 0)]
        output_nodes = list(range(self.config.genome_config.num_outputs))

        for n in G.nodes:
            if n in input_nodes:
                node_colors.append("green")      # Inputs
            elif n in output_nodes:
                node_colors.append("blue")       # Outputs
            else:
                node_colors.append("lightgrey")  # Hidden/internal

        nx.draw(
            G, pos, ax=self.ax, node_color=node_colors,
            with_labels=True, node_size=600, arrowsize=15,
            font_size=8, font_color="white"
        )
        self.canvas.draw()


def feed_forward_layout(genome, config):
    G = nx.DiGraph()

    # Input and output nodes
    input_nodes = [i for i in range(-config.genome_config.num_inputs, 0)]
    output_nodes = list(range(config.genome_config.num_outputs))

    # Add nodes and edges
    for n in genome.nodes:
        G.add_node(n)
    for (i, o), conn in genome.connections.items():
        if conn.enabled:
            G.add_edge(i, o)

    # Compute depth of each node
    depths = compute_node_depths(genome, input_nodes, output_nodes)

    # Group nodes by depth
    layers = {}
    for n, d in depths.items():
        layers.setdefault(d, []).append(n)

    # Assign positions: staggered horizontally, centered vertically
    pos = {}
    for depth in sorted(layers.keys()):
        nodes = layers[depth]
        num_nodes = len(nodes)
        horizontal_spread = 0.3 * num_nodes  # tweak this for spacing

        # Compute vertical positions centered around 0
        y_start = (num_nodes - 1) / 2
        for i, n in enumerate(nodes):
            y = y_start - i               # center vertically
            x_offset = ((i - num_nodes / 2) / num_nodes) * horizontal_spread
            x = depth
            pos[n] = (x, y)

    # Handle unreachable nodes
    free_nodes = [n for n in G.nodes if n not in pos]
    if free_nodes:
        min_depth = min(depths.values(), default=0)
        free_y = -len(pos) - 1
        for i, n in enumerate(free_nodes):
            pos[n] = (min_depth - 1, free_y - i)

    return G, pos


def compute_node_depths(genome, input_nodes, output_nodes):
    """
    Assign depth to each node:
    - Inputs at depth 0
    - Depth = max distance from any input
    - Unreachable nodes will get a default depth of max_depth + 1
    """
    depths = {n: 0 for n in input_nodes}
    changed = True

    while changed:
        changed = False
        for (i, o), conn in genome.connections.items():
            if not conn.enabled:
                continue
            if i in depths:
                new_depth = depths[i] + 1
                if o not in depths or depths[o] < new_depth:
                    depths[o] = new_depth
                    changed = True

    # Ensure outputs exist
    max_depth = max(depths.values(), default=0)
    for n in output_nodes:
        if n not in depths:
            depths[n] = max_depth + 1

    # Assign depth for any remaining nodes
    all_nodes = set(genome.nodes)
    for n in all_nodes:
        if n not in depths:
            depths[n] = max_depth + 1

    return depths
