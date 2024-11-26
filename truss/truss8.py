import numpy as np

def generate_stiffness_matrix(grid_size, young_modulus, poisson_ratio):
    """
    Generates the stiffness matrix for a fabric model with vertical and horizontal trusses.

    Parameters:
    - grid_size: Tuple (rows, cols) defining the number of nodes in the fabric grid.
    - young_modulus: Young's modulus of the trusses.
    - poisson_ratio: Poisson's ratio of the material.

    Returns:
    - stiffness_matrix: The global stiffness matrix of the fabric.
    """
    rows, cols = grid_size
    num_nodes = rows * cols
    num_dofs = 3 * num_nodes  # Each node has 3 degrees of freedom (x, y, z)

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

            # Horizontal truss (to the right)
            if j < cols - 1:
                next_node = current_node + 1
                length = 1  # Distance between adjacent nodes
                angle = 0  # Horizontal
                k_local = compute_truss_stiffness(length, angle, young_modulus, cross_sectional_area)

                # Map local stiffness matrix to global stiffness matrix
                indices = np.array([3*current_node, 3*current_node+1, 3*next_node, 3*next_node+1])
                for ii in range(4):
                    for jj in range(4):
                        stiffness_matrix[indices[ii], indices[jj]] += k_local[ii, jj]

            # Vertical truss (upwards)
            if i < rows - 1:
                next_node = current_node + cols
                length = 1  # Distance between adjacent nodes
                angle = np.pi / 2  # Vertical
                k_local = compute_truss_stiffness(length, angle, young_modulus, cross_sectional_area)

                # Map local stiffness matrix to global stiffness matrix
                indices = np.array([3*current_node, 3*current_node+1, 3*next_node, 3*next_node+1])
                for ii in range(4):
                    for jj in range(4):
                        stiffness_matrix[indices[ii], indices[jj]] += k_local[ii, jj]

    # Adjust stiffness matrix by Poisson's ratio
    stiffness_matrix *= 1 / (1 - poisson_ratio**2)

    return stiffness_matrix


# Example usage:
grid_size = (4, 4)  # 4x4 fabric grid
young_modulus = 200e9  # Young's modulus in Pascals (e.g., steel)
poisson_ratio = 0.3  # Poisson's ratio

# Generate stiffness matrix
stiffness_matrix = generate_stiffness_matrix(grid_size, young_modulus, poisson_ratio)

# Print stiffness matrix
np.set_printoptions(precision=2, suppress=True)  # Optional formatting for better readability
print("Stiffness Matrix:")
for row in stiffness_matrix:
    for x in row:
        print(f"{x:.2f}, ", end="")
    print()
print(stiffness_matrix)

