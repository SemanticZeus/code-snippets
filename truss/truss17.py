import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_nodes(n, spacing=1.0):
    node_coords = np.zeros((n * n, 3))
    for i in range(n):
        for j in range(n):
            node = i * n + j
            node_coords[node, 0] = j * spacing
            node_coords[node, 1] = i * spacing
            node_coords[node, 2] = 0.0
    return node_coords

def get_truss_connections(n):
    connections = []
    total_nodes = n * n
    for i in range(n):
        for j in range(n):
            current = i * n + j
            if j < n - 1:
                right = current + 1
                connections.append((current, right))
            if i < n - 1:
                up = current + n
                connections.append((current, up))
    return connections

def visualize_truss(node_coords, connections, title="Truss Fabric Visualization"):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Z", fontsize=12)
    for (n1, n2) in connections:
        x = [node_coords[n1, 0], node_coords[n2, 0]]
        y = [node_coords[n1, 1], node_coords[n2, 1]]
        z = [node_coords[n1, 2], node_coords[n2, 2]]
        ax.plot(x, y, z, color='blue', linewidth=1)
    ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2],
               color='red', s=20, label='Nodes')
    ax.legend()
    ax.grid(True)
    plt.show()

def main():
    n = 10
    spacing = 1.0
    node_coords = generate_nodes(n, spacing)
    connections = get_truss_connections(n)
    visualize_truss(node_coords, connections, title="10x10 Truss Fabric - Initial State")

if __name__ == "__main__":
    main()

