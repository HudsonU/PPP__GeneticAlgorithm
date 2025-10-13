import random
import neat 
import networkx as nx
from neat.genes import DefaultConnectionGene, DefaultNodeGene  # import gene classes
from neat.graphs import creates_cycle

class FullyConnectedToOutputReproduction(neat.DefaultReproduction):
    def __init__(self, reproduction_config, reporters, stagnation):
        print("✅ Using FullyConnectedToOutputReproduction")
        super().__init__(reproduction_config, reporters, stagnation)
    
    def reproduce(self, config, species, pop_size, generation):
        new_population = super().reproduce(config, species, pop_size, generation)
        genome_config = config.genome_config
        for gid, genome in new_population.items():
            self._ensure_all_nodes_connect_to_output(genome, genome_config)
        return new_population

    def _ensure_all_nodes_connect_to_output(self, genome, genome_config):
        """
        Ensures every node (including inputs and hidden nodes) has a path to at least one output.
        If no path exists, connect the node to a random output.
        """
        # Create a directed graph of all enabled connections
        G = nx.DiGraph()
        all_keys = list(genome.nodes.keys()) + list(genome_config.input_keys)
        G.add_nodes_from(all_keys)
        for (src, dst), conn in genome.connections.items():
            conn.enabled = True  # Ensure all connections are enabled
            G.add_edge(src, dst)
        
        output = genome_config.output_keys[0]
        input_keys = list(genome_config.input_keys)
        hidden_keys = list(genome.nodes.keys())

        all_nodes = input_keys + hidden_keys

        for node in all_nodes:
            # Check if node has a path to any output
            #print(node,output)
            if nx.has_path(G, node, output):
                continue
            
            # No path exists → connect to a random output
            target = output
            if (node, target) not in genome.connections:
                conn = DefaultConnectionGene(key=(node, target))
                conn.weight = random.uniform(-1.0, 1.0)
                conn.enabled = True
                genome.connections[(node, target)] = conn

                # Add the new edge to the graph so subsequent checks see it
                G.add_edge(node, target)
