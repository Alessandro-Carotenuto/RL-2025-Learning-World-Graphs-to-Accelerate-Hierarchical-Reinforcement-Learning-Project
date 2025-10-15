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

#----------------------------------------------------------------------------


class GraphVisualizer:
    """
    Visualizes GraphManager objects, accurately rendering the stored coordinate
    paths for each edge.
    """
    
    def __init__(self, graph_manager: 'GraphManager', figsize=(12, 10)):
        """
        Args:
            graph_manager: GraphManager instance with updated edge structure
            figsize: Figure size tuple
        """
        self.graph = graph_manager
        self.figsize = figsize
    
    def visualize(self, 
                  show_weights=True, 
                  show_labels=True,
                  node_size=300,
                  node_color='lightblue',
                  edge_color='gray',
                  edge_width=1.5, # Increased width for better visibility
                  highlight_nodes=None,
                  highlight_color='red',
                  title="World Graph Visualization"):
        """
        Create graph visualization, drawing the actual paths for edges.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        nodes = list(self.graph.nodes)
        if not nodes:
            print("No nodes to visualize")
            return fig, ax
        
        pos = {node: node for node in nodes}
        
        # --- MODIFIED SECTION: Draw Edges Using Stored Paths ---
        for (start, end), edge_data in self.graph.edges.items():
            path = edge_data.get('path')
            if not path:
                # Fallback to straight line if no path is stored
                x_coords = [pos[start][0], pos[end][0]]
                y_coords = [pos[start][1], pos[end][1]]
            else:
                # Use the stored path coordinates
                x_coords = [p[0] for p in path]
                y_coords = [p[1] for p in path]
            
            # Draw the actual path line
            ax.plot(x_coords, y_coords, 
                    color=edge_color, 
                    linewidth=edge_width, 
                    alpha=0.7,
                    zorder=1)
            
            # --- MODIFIED SECTION: Place Weight Labels on Path Midpoint ---
            if show_weights and path:
                weight = edge_data['weight']
                # Place label at the midpoint of the path
                mid_point = path[len(path) // 2]
                ax.text(mid_point[0], mid_point[1], str(weight), 
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
        ax.invert_yaxis() # Often useful for grid worlds where (0,0) is top-left
        
        if nodes:
            x_coords = [node[0] for node in nodes]
            y_coords = [node[1] for node in nodes]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            x_pad = (x_max - x_min) * 0.1 or 1
            y_pad = (y_max - y_min) * 0.1 or 1
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_max + y_pad, y_min - y_pad) # Inverted Y
        
        plt.tight_layout()
        return fig, ax
    
    def show_path(self, pivotal_path: List[Tuple], path_color='red', path_width=3):
        """
        Highlight a specific path of pivotal states by drawing their detailed sub-paths.
        
        Args:
            pivotal_path: List of pivotal state nodes representing the high-level path
            path_color: Color for the highlighted path
            path_width: Width of the highlighted path
        """
        if not pivotal_path or len(pivotal_path) < 2:
            print("Path is too short to visualize.")
            return

        # Start with the base visualization, highlighting the pivotal nodes
        fig, ax = self.visualize(highlight_nodes=pivotal_path, highlight_color='orange')
        
        # --- MODIFIED SECTION: Draw the detailed path for each segment ---
        for i in range(len(pivotal_path) - 1):
            start_node, end_node = pivotal_path[i], pivotal_path[i + 1]
            
            # Retrieve the detailed coordinate path for this edge from the graph
            edge_path = self.graph.get_edge_path(start_node, end_node)
            
            if edge_path:
                x_coords = [p[0] for p in edge_path]
                y_coords = [p[1] for p in edge_path]
                
                ax.plot(x_coords, y_coords,
                       color=path_color,
                       linewidth=path_width,
                       alpha=0.8,
                       zorder=5,
                       label='Shortest Path' if i == 0 else "") # Label only once
        
        ax.set_title(f"Shortest Path: {pivotal_path[0]} → {pivotal_path[-1]}", 
                    fontsize=14, fontweight='bold')
        ax.legend()
        
        return fig, ax
    
    def show_statistics(self):
        """Display graph statistics."""
        nodes = list(self.graph.nodes)
        edges = self.graph.edges
        
        print(f"Graph Statistics:")
        print(f"  Nodes: {len(nodes)}")
        print(f"  Edges: {len(edges)}")
        
        if edges:
            # --- MODIFIED SECTION: Correctly extract weights ---
            weights = [data['weight'] for data in edges.values()]
            print(f"  Edge weights: min={min(weights)}, max={max(weights)}, avg={np.mean(weights):.1f}")
        
        connectivity = {node: len(self.graph.get_neighbors(node)) for node in nodes}
        
        if connectivity:
            print(f"  Node connectivity: min={min(connectivity.values())}, max={max(connectivity.values())}")
            sorted_nodes = sorted(connectivity.items(), key=lambda x: x[1], reverse=True)
            print(f"  Most connected nodes: {sorted_nodes[:5]}")
        """Display graph statistics."""
        nodes = list(self.graph.nodes)
        edges = self.graph.edges
        
        print(f"Graph Statistics:")
        print(f"  Nodes: {len(nodes)}")
        print(f"  Edges: {len(edges)}")
        
        if edges:
            weights = list(edges.values())
            print(f"  Edge weights: min={min(weights)}, max={max(weights)}, avg={np.mean(weights):.1f}")
        
        # Node connectivity
        connectivity = {}
        for node in nodes:
            neighbors = self.graph.get_neighbors(node)
            connectivity[node] = len(neighbors)
        
        if connectivity:
            print(f"  Node connectivity: min={min(connectivity.values())}, max={max(connectivity.values())}")
            
            # Most connected nodes
            sorted_nodes = sorted(connectivity.items(), key=lambda x: x[1], reverse=True)
            print(f"  Most connected nodes: {sorted_nodes[:5]}")