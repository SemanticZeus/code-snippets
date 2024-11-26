import matplotlib.pyplot as plt
import numpy as np

def visualize_fabric_3d(grid_size, displacement=None, scale=1):
    """
    Visualizes the fabric grid with vertical and horizontal trusses in 3D.

    Parameters:
    - grid_size: Tuple (rows, cols) defining the number of nodes in the fabric grid.
    - displacement: Optional numpy array of node displacements, shape (rows * cols, 3).
                    If provided, the displaced fabric will also be shown.
    - scale: Scaling factor for visualizing displacements.
    """
    rows, cols = grid_size
    node_positions = np.array([[j, i, 0] for i in range(rows) for j in range(cols)])
    
    # Initialize 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Fabric Visualization")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")

    # Draw trusses
    for i in range(rows):
        for j in range(cols):
            current_node = i * cols + j
            x1, y1, z1 = node_positions[current_node]

            # Horizontal truss (to the right)
            if j < cols - 1:
                next_node = current_node + 1
                x2, y2, z2 = node_positions[next_node]
                ax.plot([x1, x2], [y1, y2], [z1, z2], 'b-', lw=1)

            # Vertical truss (upwards)
            if i < rows - 1:
                next_node = current_node + cols
                x2, y2, z2 = node_positions[next_node]
                ax.plot([x1, x2], [y1, y2], [z1, z2], 'b-', lw=1)

    # Plot original nodes
    ax.scatter(node_positions[:, 0], node_positions[:, 1], node_positions[:, 2], color='r', label="Original Nodes")

    # Plot displaced fabric if displacement is provided
    if displacement is not None:
        displaced_positions = node_positions + scale * displacement
        for i in range(rows):
            for j in range(cols):
                current_node = i * cols + j
                x1, y1, z1 = displaced_positions[current_node]

                # Horizontal truss (to the right)
                if j < cols - 1:
                    next_node = current_node + 1
                    x2, y2, z2 = displaced_positions[next_node]
                    ax.plot([x1, x2], [y1, y2], [z1, z2], 'r--', lw=1)

                # Vertical truss (upwards)
                if i < rows - 1:
                    next_node = current_node + cols
                    x2, y2, z2 = displaced_positions[next_node]
                    ax.plot([x1, x2], [y1, y2], [z1, z2], 'r--', lw=1)

        # Plot displaced nodes
        ax.scatter(displaced_positions[:, 0], displaced_positions[:, 1], displaced_positions[:, 2], color='g', label="Displaced Nodes")

    ax.legend(loc="best")
    plt.show()


# Example usage:
grid_size = (4, 4)  # 4x4 fabric grid
displacement = np.random.uniform(-0.1, 0.1, size=(grid_size[0] * grid_size[1], 3))  # Example displacements in 3D
visualize_fabric_3d(grid_size, displacement, scale=10)

