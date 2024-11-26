import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_nodes(n, spacing=1.0, h=1.0):
   top_nodes = []
    base_nodes = []
    for i in range(n):
        for j in range(n):
            x = j * spacing
            y = i * spacing
            top_nodes.append([x, y, 0.0])    # Top grid at z=0
            base_nodes.append([x, y, -h])    # Base grid at z=-h
    node_coords = np.array(top_nodes + base_nodes)
    return node_coords

def get_truss_connections(n):
    connections = []
    total_nodes = 2 * n * n
    # Top grid connections (nodes 0 to n*n -1)
    for i in range(n):
        for j in range(n):
            current = i * n + j
            # Horizontal truss (to the right)
            if j < n - 1:
                right = current + 1
                connections.append((current, right))
            # Vertical truss (upwards)
            if i < n - 1:
                up = current + n
                connections.append((current, up))
    # Base grid connections (nodes n*n to 2*n*n -1)
    base_offset = n * n
    for i in range(n):
        for j in range(n):
            current = base_offset + i * n + j
            # Horizontal truss (to the right)
            if j < n - 1:
                right = current + 1
                connections.append((current, right))
            # Vertical truss (upwards)
            if i < n - 1:
                up = current + n
                connections.append((current, up))
    # Vertical trusses connecting top and base grids
    for i in range(n * n):
        top = i
        base = base_offset + i
        connections.append((top, base))
    return connections

def visualize_truss_connections(node_coords, connections, title="Truss Connections Visualization"):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Z", fontsize=12)
    
    # Plot truss connections
    for (n1, n2) in connections:
        x = [node_coords[n1, 0], node_coords[n2, 0]]
        y = [node_coords[n1, 1], node_coords[n2, 1]]
        z = [node_coords[n1, 2], node_coords[n2, 2]]
        ax.plot(x, y, z, color='blue', linewidth=0.5)
    
    # Plot nodes
    ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2],
               color='red', s=20, label='Nodes')
    
    ax.legend()
    ax.grid(True)
    plt.show()

def main_visualization_example():
    # Parameters for a smaller grid for demonstration
    n = 4  # Using a smaller grid (4x4) for clarity
    spacing = 1.0
    h = 1.0
    
    # Generate node coordinates
    node_coords = generate_nodes(n, spacing, h)
    
    # Define truss connections
    connections = get_truss_connections(n)
    
    # Visualize truss connections
    visualize_truss_connections(node_coords, connections, title=f"{n}x{n} Truss Fabric Connections")

if __name__ == "__main__":
    main_visualization_example()

