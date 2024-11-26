import numpy as np
import matplotlib.pyplot as plt

def generate_stiffness_matrix(grid_size, young_modulus, poisson_ratio):
    rows, cols = grid_size
    num_nodes = rows * cols
    num_dofs = 3 * num_nodes  # Each node has 3 DOFs: x, y, z
    stiffness_matrix = np.zeros((num_dofs, num_dofs))

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

    cross_sectional_area = 1.0

    for i in range(rows):
        for j in range(cols):
            current_node = i * cols + j

            if j < cols - 1:  # Horizontal truss
                next_node = current_node + 1
                length = 1.0
                angle = 0.0
                k_local = compute_truss_stiffness(length, angle, young_modulus, cross_sectional_area)
                indices = [3*current_node, 3*current_node+1, 3*next_node, 3*next_node+1]
                for ii in range(4):
                    for jj in range(4):
                        stiffness_matrix[indices[ii], indices[jj]] += k_local[ii % 2, jj % 2]

            if i < rows - 1:  # Vertical truss
                next_node = current_node + cols
                length = 1.0
                angle = np.pi / 2
                k_local = compute_truss_stiffness(length, angle, young_modulus, cross_sectional_area)
                indices = [3*current_node, 3*current_node+1, 3*next_node, 3*next_node+1]
                for ii in range(4):
                    for jj in range(4):
                        stiffness_matrix[indices[ii], indices[jj]] += k_local[ii % 2, jj % 2]

    stiffness_matrix *= 1 / (1 - poisson_ratio**2)
    return stiffness_matrix

def solve_displacement(stiffness_matrix, force_vector, fixed_dofs):
    num_dofs = len(force_vector)
    free_dofs = np.setdiff1d(np.arange(num_dofs), fixed_dofs)
    reduced_stiffness = stiffness_matrix[np.ix_(free_dofs, free_dofs)]
    reduced_force = force_vector[free_dofs]
    reduced_displacement = np.linalg.solve(reduced_stiffness, reduced_force)
    displacement = np.zeros(num_dofs)
    displacement[free_dofs] = reduced_displacement
    return displacement

def visualize_fabric_3d(grid_size, displacement, scale=1):
    rows, cols = grid_size
    node_positions = np.array([[j, i, 0] for i in range(rows) for j in range(cols)])
    displaced_positions = node_positions + displacement.reshape(-1, 3) * scale

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Fabric Simulation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    for i in range(rows):
        for j in range(cols):
            current_node = i * cols + j
            x1, y1, z1 = displaced_positions[current_node]

            if j < cols - 1:  # Horizontal truss
                next_node = current_node + 1
                x2, y2, z2 = displaced_positions[next_node]
                ax.plot([x1, x2], [y1, y2], [z1, z2], 'b-')

            if i < rows - 1:  # Vertical truss
                next_node = current_node + cols
                x2, y2, z2 = displaced_positions[next_node]
                ax.plot([x1, x2], [y1, y2], [z1, z2], 'b-')

    ax.scatter(displaced_positions[:, 0], displaced_positions[:, 1], displaced_positions[:, 2], color='r')
    plt.show()

# Parameters
grid_size = (6, 6)  # 6x6 fabric grid
young_modulus = 200e9  # Pa (e.g., steel)
poisson_ratio = 0.3  # Poisson's ratio
force_vector = np.zeros(3 * grid_size[0] * grid_size[1])  # Forces
force_vector[2::3] = -100  # Apply gravity in the Z direction
fixed_dofs = [0, 3*(grid_size[0]*grid_size[1]-1)]  # Fix two diagonal corners

# Generate stiffness matrix and solve
stiffness_matrix = generate_stiffness_matrix(grid_size, young_modulus, poisson_ratio)
displacement = solve_displacement(stiffness_matrix, force_vector, fixed_dofs)

# Visualize the result
visualize_fabric_3d(grid_size, displacement, scale=10)

