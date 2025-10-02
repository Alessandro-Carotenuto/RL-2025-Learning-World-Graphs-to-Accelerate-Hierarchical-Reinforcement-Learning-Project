import heapq
from typing import List
import matplotlib.pyplot as plt
import numpy as np


#----------------------------------------------------------------------------#
#                                GRAPH                                       #
#----------------------------------------------------------------------------#


class GraphManager:
    def __init__(self):
        self.nodes = set()
        self.edges = {}
        
    def add_node(self, node):
        self.nodes.add(node)
    
    def add_edge(self, node1, node2, weight):
        self.edges[(node1, node2)] = weight
    
    def get_neighbors(self, node):
        """Get all neighbors of a node."""
        neigh = set()
        for (n1, n2) in self.edges.keys():
            if n1 == node:
                neigh.add(n2)
        return neigh
    
    def shortest_path(self, start, end):
        if start not in self.nodes or end not in self.nodes:
            return None, float('inf')
    
        # Distance from start to each node
        distances = {node: float('inf') for node in self.nodes}
        distances[start] = 0
    
        # Priority queue (min-heap): (distance, node)
        pq = [(0, start)]
    
        # To reconstruct the path
        previous = {}
    
        while pq:
            current_dist, current = heapq.heappop(pq)
    
            # Skip if we already found a shorter path
            if current_dist > distances[current]:
                continue
    
            # Stop early if we reached the end
            if current == end:
                path = []
                while current in previous:
                    path.append(current)
                    current = previous[current]
                path.append(start)
                return path[::-1], distances[end]
    
            # Explore neighbors
            for neighbor in self.get_neighbors(current):
                weight = self.edges.get((current, neighbor))
                if weight is None:
                    continue  # skip if no direct edge
    
                alt = current_dist + weight
                if alt < distances[neighbor]:
                    distances[neighbor] = alt
                    previous[neighbor] = current
                    heapq.heappush(pq, (alt, neighbor))
    
        return None, float('inf')

#----------------------------------------------------------------------------

class GraphVisualizer:
    """
    Visualizes GraphManager objects with nodes positioned by coordinates.
    """
    
    def __init__(self, graph_manager, figsize=(12, 10)):
        """
        Args:
            graph_manager: GraphManager instance
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
                  edge_width=1,
                  highlight_nodes=None,
                  highlight_color='red',
                  title="World Graph Visualization"):
        """
        Create graph visualization.
        
        Args:
            show_weights: Show edge weights
            show_labels: Show node coordinate labels
            node_size: Size of nodes
            node_color: Color of regular nodes
            edge_color: Color of edges
            edge_width: Width of edge lines
            highlight_nodes: List of nodes to highlight
            highlight_color: Color for highlighted nodes
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Extract node positions
        nodes = list(self.graph.nodes)
        if not nodes:
            print("No nodes to visualize")
            return fig, ax
        
        # Use actual coordinates as positions
        pos = {node: node for node in nodes}
        
        # Draw edges
        for (start, end), weight in self.graph.edges.items():
            if start in pos and end in pos:
                x_coords = [pos[start][0], pos[end][0]]
                y_coords = [pos[start][1], pos[end][1]]
                
                ax.plot(x_coords, y_coords, 
                       color=edge_color, 
                       linewidth=edge_width, 
                       alpha=0.6,
                       zorder=1)
                
                # Add weight labels
                if show_weights:
                    mid_x = (x_coords[0] + x_coords[1]) / 2
                    mid_y = (y_coords[0] + y_coords[1]) / 2
                    ax.text(mid_x, mid_y, str(weight), 
                           fontsize=8, 
                           ha='center', 
                           va='center',
                           bbox=dict(boxstyle='round,pad=0.2', 
                                   facecolor='white', 
                                   alpha=0.8),
                           zorder=3)
        
        # Draw nodes
        for node in nodes:
            x, y = pos[node]
            
            # Choose color
            color = highlight_color if (highlight_nodes and node in highlight_nodes) else node_color
            
            # Draw node
            ax.scatter(x, y, 
                      s=node_size, 
                      c=color, 
                      edgecolors='black',
                      linewidth=1,
                      zorder=2)
            
            # Add labels
            if show_labels:
                ax.annotate(f'{node}', 
                           (x, y), 
                           xytext=(5, 5), 
                           textcoords='offset points',
                           fontsize=9,
                           ha='left',
                           zorder=4)
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Set reasonable limits
        if nodes:
            x_coords = [node[0] for node in nodes]
            y_coords = [node[1] for node in nodes]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Add padding
            x_pad = (x_max - x_min) * 0.1 or 1
            y_pad = (y_max - y_min) * 0.1 or 1
            
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
        
        plt.tight_layout()
        return fig, ax
    
    def show_path(self, path, path_color='red', path_width=3):
        """
        Highlight a specific path through the graph.
        
        Args:
            path: List of nodes representing path
            path_color: Color for path edges
            path_width: Width of path edges
        """
        if len(path) < 2:
            return
        
        fig, ax = self.visualize(highlight_nodes=path)
        
        # Draw path edges
        for i in range(len(path) - 1):
            start, end = path[i], path[i + 1]
            
            x_coords = [start[0], end[0]]
            y_coords = [start[1], end[1]]
            
            ax.plot(x_coords, y_coords,
                   color=path_color,
                   linewidth=path_width,
                   alpha=0.8,
                   zorder=5)
        
        ax.set_title(f"Path: {path[0]} → {path[-1]}", 
                    fontsize=14, fontweight='bold')
        
        return fig, ax
    
    def show_statistics(self):
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