import numpy as np

def generate_stiffness_matrix(grid_size, poisson_ratio, young_modulus):
    """
    Generates the stiffness matrix for a fabric model with vertical and horizontal trusses.

    Parameters:
    - grid_size: Tuple (rows, cols) defining the number of nodes in the fabric grid.
    - poisson_ratio: Poisson's ratio of the material.
    - young_modulus: Young's modulus of the trusses.

    Returns:
    - stiffness_matrix: The global stiffness matrix of the fabric.
    """
    rows, cols = grid_size
    num_nodes = rows * cols
    num_dofs = 2 * num_nodes  # Each node has 2 degrees of freedom (x, y)

    # Initialize the global stiffness matrix
    stiffness_matrix = np.zeros((num_dofs, num_dofs))

    # Function to compute local stiffness matrix for a truss element
    def compute_truss_stiffness(length, angle, E, A):
        c = np.cos(angle)
        s = np.sin(angle)
        k = E * A / length
        k_local = k * np.array([
            [c*c, c*s, -c*c, -c*s],
            [c*s, s*s, -c*s, -s*s],
            [-c*c, -c*s, c*c, c*s],
            [-c*s, -s*s, c*s, s*s]
        ])
        return k_local

    # Area of truss cross-section (assumed uniform for simplicity)
    cross_sectional_area = 1.0  # Arbitrary unit area

    # Loop through all grid connections
    for i in range(rows):
        for j in range(cols):
            current_node = i * cols + j

            # Coordinates of the current node
            x1, y1 = j, i

            # Horizontal truss (to the right)
            if j < cols - 1:
                next_node = current_node + 1
                x2, y2 = x1 + 1, y1
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = 0  # Horizontal
                k_local = compute_truss_stiffness(length, angle, young_modulus, cross_sectional_area)

                # Map local stiffness matrix to global stiffness matrix
                indices = np.array([2*current_node, 2*current_node+1, 2*next_node, 2*next_node+1])
                for ii in range(4):
                    for jj in range(4):
                        stiffness_matrix[indices[ii], indices[jj]] += k_local[ii, jj]

            # Vertical truss (upwards)
            if i < rows - 1:
                next_node = current_node + cols
                x2, y2 = x1, y1 + 1
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = np.pi / 2  # Vertical
                k_local = compute_truss_stiffness(length, angle, young_modulus, cross_sectional_area)

                # Map local stiffness matrix to global stiffness matrix
                indices = np.array([2*current_node, 2*current_node+1, 2*next_node, 2*next_node+1])
                for ii in range(4):
                    for jj in range(4):
                        stiffness_matrix[indices[ii], indices[jj]] += k_local[ii, jj]

    # Apply Poisson's ratio adjustment
    stiffness_matrix *= 1 / (1 - poisson_ratio**2)

    return stiffness_matrix


# Example usage
grid_size = (4, 4)  # 4x4 fabric grid
poisson_ratio = 0.3  # Poisson's ratio
young_modulus = 200e9  # Young's modulus in Pascals (e.g., steel)

stiffness_matrix = generate_stiffness_matrix(grid_size, poisson_ratio, young_modulus)

print("Stiffness Matrix:")
print(stiffness_matrix)




import matplotlib.pyplot as plt
import numpy as np

def visualize_fabric(grid_size, displacement=None, scale=1):
    """
    Visualizes the fabric grid with vertical and horizontal trusses.

    Parameters:
    - grid_size: Tuple (rows, cols) defining the number of nodes in the fabric grid.
    - displacement: Optional numpy array of node displacements, shape (rows * cols, 2).
                    If provided, the displaced fabric will also be shown.
    - scale: Scaling factor for visualizing displacements.
    """
    rows, cols = grid_size
    node_positions = np.array([[j, i] for i in range(rows) for j in range(cols)])

    # Plot original fabric grid
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Fabric Visualization")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    # Draw trusses
    for i in range(rows):
        for j in range(cols):
            current_node = i * cols + j
            x1, y1 = node_positions[current_node]

            # Horizontal truss (to the right)
            if j < cols - 1:
                next_node = current_node + 1
                x2, y2 = node_positions[next_node]
                ax.plot([x1, x2], [y1, y2], 'b-', lw=1)

            # Vertical truss (upwards)
            if i < rows - 1:
                next_node = current_node + cols
                x2, y2 = node_positions[next_node]
                ax.plot([x1, x2], [y1, y2], 'b-', lw=1)

    # Plot nodes
    ax.plot(node_positions[:, 0], node_positions[:, 1], 'ro', label="Original Nodes")

    # Plot displaced fabric if displacement is provided
    if displacement is not None:
        displaced_positions = node_positions + scale * displacement
        for i in range(rows):
            for j in range(cols):
                current_node = i * cols + j
                x1, y1 = displaced_positions[current_node]

                # Horizontal truss (to the right)
                if j < cols - 1:
                    next_node = current_node + 1
                    x2, y2 = displaced_positions[next_node]
                    ax.plot([x1, x2], [y1, y2], 'r--', lw=1)

                # Vertical truss (upwards)
                if i < rows - 1:
                    next_node = current_node + cols
                    x2, y2 = displaced_positions[next_node]
                    ax.plot([x1, x2], [y1, y2], 'r--', lw=1)

        # Plot displaced nodes
        ax.plot(displaced_positions[:, 0], displaced_positions[:, 1], 'go', label="Displaced Nodes")
        ax.legend(loc="best")

    plt.grid(True)
    plt.show()


# Example usage:
grid_size = (4, 4)  # 4x4 fabric grid
displacement = np.random.uniform(-0.1, 0.1, size=(grid_size[0] * grid_size[1], 2))  # Example displacements
visualize_fabric(grid_size, displacement, scale=10)

