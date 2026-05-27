import heapq
from collections import deque
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np


#----------------------------------------------------------------------------#
#                            GRAPH MANAGER                                   #
#----------------------------------------------------------------------------#


class GraphManager:

    def __init__(self): 
        self.nodes = set()        # Set of nodes represented as (x, y) tuples,  O(1) lookup time with set
        self.edges = {}           # Dict with keys as (node1, node2) tuples and values as dicts with 'weight' and 'path'
        self.adjacency_list = {}  # Dict with keys as nodes and values as sets of neighboring nodes

        # STRUCTURE OF ADJACENCY DICTIONARY:
        #
        # self.adjacency = {
        #     node1: {'outgoing': set(), 'ingoing': set()},
        #     node2: {'outgoing': set(), 'ingoing': set()}
        # }
        
    def add_node(self, node):
        # Add a node represented as (x, y) tuple
        self.nodes.add(node)
    
    def add_edge(self, node1, node2, weight, path: List[Tuple[int, int]]):
        # Directed edge from node1 to node2 with associated weight and path
        # A dictionary is used to associate node pairs with their weights and paths
        # - A nested dictionary is used to store both weight and path for each edge, using a string as key for readability
        self.edges[(node1, node2)] = {'weight': weight, 'path': path} 

        # Update adjacency list for outgoing edges

        if node1 not in self.adjacency_list:
            self.adjacency_list[node1] = {'outgoing': set(), 'ingoing': set()}
        
        if node2 not in self.adjacency_list:
            self.adjacency_list[node2] = {'outgoing': set(), 'ingoing': set()} 
        
        self.adjacency_list[node1]['outgoing'].add(node2)
        self.adjacency_list[node2]['ingoing'].add(node1)
    
    
    def get_edge_path(self, start_node, end_node) -> Optional[List[Tuple[int, int]]]:
        # Retrieve the path associated with an edge
        edge_data = self.edges.get((start_node, end_node))  # Get the edge data dictionary
        return edge_data.get('path') if edge_data else None # Return the path if edge exists, else None
    
    def get_outgoing_neighbors(self, node):
        # Get all neighboring nodes connected by outgoing edges from the given node
        neigh = set()
        if node in self.adjacency_list:
            neigh = self.adjacency_list[node]['outgoing']
        return neigh
    
    def get_all_neighbors(self, node):
        # Get all neighboring nodes connected to and from the given node
        neighout = set()
        neighin = set()
        if node in self.adjacency_list:
            neighout = self.adjacency_list[node]['outgoing']
            neighin = self.adjacency_list[node]['ingoing']
        neigh = neighout | neighin # Union of outgoing and ingoing neighbors
        return neigh

    def shortest_path(self, start, end):
        if start not in self.nodes or end not in self.nodes:    #if start or end node not in graph
            return None, float('inf')

        distances = {node: float('inf') for node in self.nodes} # Initialize distances to infinity, from start to all nodes
        distances[start] = 0                                    # Distance to start node is 0   
        pq = [(0, start)]                                       # Priority queue for Dijkstra's algorithm
                                                                # - Priority queue is a min-heap based on distance, i initialize as a 
                                                                # - list with a tuple (distance, node), but i treat it as a heap using heapq functions
                                                                # - cause works on lists directly but provides heap functionalities
                                                                
        previous = {}                                           # To reconstruct the shortest path

        while pq:                                               # While there are nodes to process
            current_dist, current = heapq.heappop(pq)           # Get the node with the smallest distance
            #i can pop beacuse pq is a min-heap based on distance, from heapq documentation
            #heapq.heappop pops and returns the smallest item from the heap, maintaining the heap invariant
            #in this case the smallest item is the tuple with the smallest distance
            #
            #-- In the first iteration, it will pop (0, start) since start has distance 0 and it's the only item in the heap


            if current_dist > distances[current]:                # If a shorter path to current has been found,  skip processing
                continue

            if current == end:                                    # If we reached the end node, reconstruct the path
                path = []                             
                while current in previous:                        
                    path.append(current)
                    current = previous[current]
                path.append(start)
                return path[::-1], distances[end]

           
            for neighbor in self.get_outgoing_neighbors(current):  # Lookup outgoing neighbors O(1) with adjacency list
                edge_data = self.edges.get((current, neighbor))    # Get edge data for current to neighbor

                if not edge_data:
                    continue
                
                weight = edge_data['weight']
                neighbor_to_curr = current_dist + weight                         #neighbor_to_curr is the distance to neighbor through current
                if neighbor_to_curr < distances[neighbor]:                       
                    distances[neighbor] = neighbor_to_curr
                    previous[neighbor] = current
                    heapq.heappush(pq, (neighbor_to_curr, neighbor))             # Push the updated distance and neighbor into the priority queue

        return None, float('inf')

    def get_reachable_nodes(self, start):
        """Return set of nodes reachable from start via directed edges (BFS)."""
        if start not in self.nodes:
            return set()
        visited = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in self.get_outgoing_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited





#----------------------------------------------------------------------------#
#                           GRAPH VISUALIZER                                 #
#----------------------------------------------------------------------------#

class GraphVisualizer:
    """
    Visualizes GraphManager objects with straight-line edges between nodes.
    """

    # Color map for ASCII grid cells — MiniGrid-style palette
    _CELL_COLORS = {
        '#': [40,  40,  40],   # wall  - near black
        '-': [190, 190, 190],  # empty - light gray
        'G': [0,   180, 0],    # goal  - green
        'K': [255, 215, 0],    # key   - gold
        'D': [140, 70,  20],   # door  - brown
        'B': [30,  120, 255],  # ball  - blue
    }

    def __init__(self, graph_manager: 'GraphManager', figsize=(12, 10)):
        self.graph = graph_manager
        self.figsize = figsize

    def _grid_to_image(self, grid_state) -> np.ndarray:
        """Convert ASCII grid (grid_state[y][x]) to an RGB uint8 image."""
        h = len(grid_state)
        w = len(grid_state[0]) if h > 0 else 0
        img = np.full((h, w, 3), 200, dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                char = grid_state[y][x] if x < len(grid_state[y]) else '-'
                if char == 'B':
                    char = '-'  # balls are randomized per episode, show as empty
                img[y, x] = self._CELL_COLORS.get(char, [200, 200, 200])
        return img

    def visualize(self,
                  show_weights=True,
                  show_labels=True,
                  node_size=300,
                  node_color='#C8A830',
                  edge_color='#555555',
                  edge_width=1.5,
                  highlight_nodes=None,
                  highlight_color='red',
                  title="World Graph Visualization",
                  grid_state=None):
        """
        Creates a graph visualization with straight-line edges.
        If grid_state is provided, the MiniGrid map is rendered as background
        and the graph is overlaid using the same (x, y) coordinate system,
        with y=0 at the top (MiniGrid convention).
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        nodes = list(self.graph.nodes)
        if not nodes:
            print("No nodes to visualize")
            return fig, ax

        pos = {node: node for node in nodes}

        # Background grid (optional)
        if grid_state is not None:
            img = self._grid_to_image(grid_state)
            h, w = img.shape[:2]
            # extent=[left, right, bottom, top]; bottom > top inverts y so row 0 is at top
            ax.imshow(img, extent=[-0.5, w - 0.5, h - 0.5, -0.5], zorder=0)
            ax.set_xlim(-0.5, w - 0.5)
            ax.set_ylim(h - 0.5, -0.5)  # y=0 at top, y=h at bottom
        else:
            x_vals = [n[0] for n in nodes]
            y_vals = [n[1] for n in nodes]
            x_min, x_max = min(x_vals), max(x_vals)
            y_min, y_max = min(y_vals), max(y_vals)
            x_pad = (x_max - x_min) * 0.1 or 1
            y_pad = (y_max - y_min) * 0.1 or 1
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_max + y_pad, y_min - y_pad)  # y=0 at top

        # Draw edges
        for (start, end), edge_data in self.graph.edges.items():
            ax.plot([pos[start][0], pos[end][0]],
                    [pos[start][1], pos[end][1]],
                    color=edge_color, linewidth=edge_width, alpha=0.7, zorder=1)

            if show_weights:
                weight = edge_data.get('weight', '')
                mid_x = (pos[start][0] + pos[end][0]) / 2
                mid_y = (pos[start][1] + pos[end][1]) / 2
                ax.text(mid_x, mid_y, str(weight), fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                        zorder=3)

        # Draw nodes
        for node in nodes:
            x, y = pos[node]
            color = highlight_color if (highlight_nodes and node in highlight_nodes) else node_color
            ax.scatter(x, y, s=node_size, c=color, edgecolors='black', linewidth=1, zorder=2)
            if show_labels:
                ax.annotate(f'{node}', (x, y), xytext=(5, 5), textcoords='offset points',
                            fontsize=9, ha='left', zorder=4)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, ax


        connectivity = {node: len(self.graph.get_neighbors(node)) for node in nodes}

        if connectivity:
            print(f"  Node connectivity: min={min(connectivity.values())}, max={max(connectivity.values())}")
            sorted_nodes = sorted(connectivity.items(), key=lambda x: x[1], reverse=True)
            print(f"  Most connected nodes: {sorted_nodes[:5]}")