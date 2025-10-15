import heapq
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np


#----------------------------------------------------------------------------#
#                                GRAPH                                       #
#----------------------------------------------------------------------------#


class GraphManager:

    def __init__(self):
        self.nodes = set()
        self.edges = {} # NEW: (start, end) -> {'weight': int, 'path': List[Tuple]} 
        
    def add_node(self, node):
        self.nodes.add(node)
    
    def add_edge(self, node1, node2, weight, path: List[Tuple[int, int]]):
        self.edges[(node1, node2)] = {'weight': weight, 'path': path}
    
    # NEW: A helper to get the stored path for an edge
    def get_edge_path(self, start_node, end_node) -> Optional[List[Tuple[int, int]]]:
        edge_data = self.edges.get((start_node, end_node))
        return edge_data.get('path') if edge_data else None

    
    def get_neighbors(self, node):
        """Get all neighbors of a node."""
        neigh = set()
        for (n1, n2) in self.edges.keys():
            if n1 == node:
                neigh.add(n2)
        return neigh
    
    # MODIFIED: shortest_path needs to access the weight from the dictionary
    def shortest_path(self, start, end):
        if start not in self.nodes or end not in self.nodes:
            return None, float('inf')
    
        distances = {node: float('inf') for node in self.nodes}
        distances[start] = 0
        pq = [(0, start)]
        previous = {}
    
        while pq:
            current_dist, current = heapq.heappop(pq)
    
            if current_dist > distances[current]:
                continue
    
            if current == end:
                path = []
                while current in previous:
                    path.append(current)
                    current = previous[current]
                path.append(start)
                return path[::-1], distances[end]
    
            for neighbor in self.get_neighbors(current):
                # *** MODIFICATION HERE ***
                edge_data = self.edges.get((current, neighbor))
                if not edge_data:
                    continue
                
                weight = edge_data['weight']
                # *** END MODIFICATION ***
    
                alt = current_dist + weight
                if alt < distances[neighbor]:
                    distances[neighbor] = alt
                    previous[neighbor] = current
                    heapq.heappush(pq, (alt, neighbor))
    
        return None, float('inf')
    


class GraphVisualizer:
    """
    Visualizes GraphManager objects with straight-line edges between nodes.
    """

    def __init__(self, graph_manager: 'GraphManager', figsize=(12, 10)):
        """
        Args:
            graph_manager: An instance of the GraphManager class.
            figsize: A tuple specifying the figure size for the plot.
        """
        self.graph = graph_manager
        self.figsize = figsize

    def visualize(self,
                  show_weights=True,
                  show_labels=True,
                  node_size=300,
                  node_color='lightblue',
                  edge_color='gray',
                  edge_width=1.5,
                  highlight_nodes=None,
                  highlight_color='red',
                  title="World Graph Visualization"):
        """
        Creates a graph visualization with straight-line edges.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        nodes = list(self.graph.nodes)
        if not nodes:
            print("No nodes to visualize")
            return fig, ax

        pos = {node: node for node in nodes}

        # --- MODIFIED SECTION for straight-line edges ---
        for (start, end), edge_data in self.graph.edges.items():
            # Get the start and end coordinates for a straight line
            x_coords = [pos[start][0], pos[end][0]]
            y_coords = [pos[start][1], pos[end][1]]

            # Draw the straight line
            ax.plot(x_coords, y_coords,
                    color=edge_color,
                    linewidth=edge_width,
                    alpha=0.7,
                    zorder=1)
            
            # Place weight labels at the midpoint of the straight line
            if show_weights:
                weight = edge_data.get('weight', '')
                mid_point_x = (pos[start][0] + pos[end][0]) / 2
                mid_point_y = (pos[start][1] + pos[end][1]) / 2
                ax.text(mid_point_x, mid_point_y, str(weight),
                        fontsize=8,
                        ha='center',
                        va='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                        zorder=3)

        # --- UNCHANGED SECTION: Draw Nodes and Labels ---
        for node in nodes:
            x, y = pos[node]
            color = highlight_color if (highlight_nodes and node in highlight_nodes) else node_color
            ax.scatter(x, y, s=node_size, c=color, edgecolors='black', linewidth=1, zorder=2)
            if show_labels:
                ax.annotate(f'{node}', (x, y), xytext=(5, 5), textcoords='offset points',
                            fontsize=9, ha='left', zorder=4)

        # Formatting (unchanged)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

        if nodes:
            x_coords = [node[0] for node in nodes]
            y_coords = [node[1] for node in nodes]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            x_pad = (x_max - x_min) * 0.1 or 1
            y_pad = (y_max - y_min) * 0.1 or 1
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_max + y_pad, y_min - y_pad)

        plt.tight_layout()
        return fig, ax

    def show_path(self, pivotal_path: List[Tuple], path_color='red', path_width=3):
        """
        Highlights a specific path of pivotal states by drawing straight lines between them.
        """
        if not pivotal_path or len(pivotal_path) < 2:
            print("Path is too short to visualize.")
            return

        # Start with the base visualization, highlighting the pivotal nodes
        fig, ax = self.visualize(highlight_nodes=pivotal_path, highlight_color='orange')

        # --- MODIFIED SECTION for straight-line path segments ---
        for i in range(len(pivotal_path) - 1):
            start_node, end_node = pivotal_path[i], pivotal_path[i + 1]

            # Draw a straight line for this segment of the path
            ax.plot([start_node[0], end_node[0]],
                    [start_node[1], end_node[1]],
                    color=path_color,
                    linewidth=path_width,
                    alpha=0.8,
                    zorder=5,
                    label='Shortest Path' if i == 0 else "")

        ax.set_title(f"Shortest Path: {pivotal_path[0]} → {pivotal_path[-1]}",
                     fontsize=14, fontweight='bold')
        if any(ax.get_legend_handles_labels()):
            ax.legend()

        return fig, ax

    def show_statistics(self):
        """Display graph statistics."""
        nodes = list(self.graph.nodes)
        edges = self.graph.edges

        print("Graph Statistics:")
        print(f"  Nodes: {len(nodes)}")
        print(f"  Edges: {len(edges)}")

        if edges:
            weights = [data['weight'] for data in edges.values()]
            print(f"  Edge weights: min={min(weights)}, max={max(weights)}, avg={np.mean(weights):.1f}")

        connectivity = {node: len(self.graph.get_neighbors(node)) for node in nodes}

        if connectivity:
            print(f"  Node connectivity: min={min(connectivity.values())}, max={max(connectivity.values())}")
            sorted_nodes = sorted(connectivity.items(), key=lambda x: x[1], reverse=True)
            print(f"  Most connected nodes: {sorted_nodes[:5]}")