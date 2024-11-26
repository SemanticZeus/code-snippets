import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_grid(grid_size, spacing=1.0):
    """
    Generates a grid of nodes in 3D space.

    Parameters:
    - grid_size: Tuple (rows, cols) defining the number of nodes in the grid.
    - spacing: Distance between adjacent nodes.

    Returns:
    - node_coords: Numpy array of shape (num_nodes, 3) with (x, y, z) coordinates.
    """
    rows, cols = grid_size
    node_coords = np.zeros((rows * cols, 3))
    for i in range(rows):
        for j in range(cols):
            node = i * cols + j
            node_coords[node, 0] = j * spacing  # x-coordinate
            node_coords[node, 1] = i * spacing  # y-coordinate
            node_coords[node, 2] = 0.0          # z-coordinate (initially flat)
    return node_coords

def get_truss_connections(grid_size):
    """
    Defines truss connections (horizontal and vertical) for the grid.

    Parameters:
    - grid_size: Tuple (rows, cols) defining the number of nodes in the grid.

    Returns:
    - connections: List of tuples, each containing two node indices connected by a truss.
    """
    rows, cols = grid_size
    connections = []
    for i in range(rows):
        for j in range(cols):
            current = i * cols + j
            # Horizontal truss (to the right)
            if j < cols - 1:
                right = current + 1
                connections.append((current, right))
            # Vertical truss (upwards)
            if i < rows - 1:
                up = current + cols
                connections.append((current, up))
    return connections

def compute_local_stiffness_matrix(node1, node2, E, A):
    """
    Computes the local stiffness matrix for a truss element in 3D.

    Parameters:
    - node1, node2: Numpy arrays of shape (3,) representing the coordinates of the two nodes.
    - E: Young's modulus.
    - A: Cross-sectional area.

    Returns:
    - k_local: Numpy array of shape (6,6) representing the local stiffness matrix.
    """
    L = node2 - node1
    length = np.linalg.norm(L)
    if length == 0:
        raise ValueError("Zero length truss element detected.")
    l, m, n = L / length  # Direction cosines
    k = (E * A) / length
    # Local stiffness matrix in global coordinates
    k_local = k * np.array([
        [ l*l,  l*m,  l*n, -l*l, -l*m, -l*n],
        [ l*m,  m*m,  m*n, -l*m, -m*m, -m*n],
        [ l*n,  m*n,  n*n, -l*n, -m*n, -n*n],
        [-l*l, -l*m, -l*n,  l*l,  l*m,  l*n],
        [-l*m, -m*m, -m*n,  l*m,  m*m,  m*n],
        [-l*n, -m*n, -n*n,  l*n,  m*n,  n*n]
    ])
    return k_local

def assemble_global_stiffness(node_coords, connections, E, A):
    """
    Assembles the global stiffness matrix for the entire truss structure.

    Parameters:
    - node_coords: Numpy array of shape (num_nodes, 3) with node coordinates.
    - connections: List of tuples defining truss connections.
    - E: Young's modulus.
    - A: Cross-sectional area.

    Returns:
    - K: Global stiffness matrix of shape (3*num_nodes, 3*num_nodes).
    """
    num_nodes = node_coords.shape[0]
    K = np.zeros((3*num_nodes, 3*num_nodes))
    for (n1, n2) in connections:
        node1 = node_coords[n1]
        node2 = node_coords[n2]
        k_local = compute_local_stiffness_matrix(node1, node2, E, A)
        dof_indices = [
            3*n1, 3*n1+1, 3*n1+2,
            3*n2, 3*n2+1, 3*n2+2
        ]
        for i in range(6):
            for j in range(6):
                K[dof_indices[i], dof_indices[j]] += k_local[i, j]
    return K

def apply_boundary_conditions(K, F, fixed_dofs):
    """
    Applies boundary conditions by modifying the stiffness matrix and force vector.

    Parameters:
    - K: Global stiffness matrix.
    - F: Global force vector.
    - fixed_dofs: List or array of DOF indices to be fixed.

    Returns:
    - K_mod: Modified stiffness matrix.
    - F_mod: Modified force vector.
    """
    num_dofs = K.shape[0]
    free_dofs = np.setdiff1d(np.arange(num_dofs), fixed_dofs)

    # Partition the stiffness matrix
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    K_fc = K[np.ix_(free_dofs, fixed_dofs)]
    # Partition the force vector
    F_f = F[free_dofs]

    # Since fixed DOFs have zero displacement, F_ff = F_f - K_fc * U_c = F_f
    F_ff = F_f

    return K_ff, F_ff, free_dofs

def solve_displacements(K, F):
    """
    Solves for the displacements.

    Parameters:
    - K: Reduced stiffness matrix.
    - F: Reduced force vector.

    Returns:
    - U: Displacement vector for free DOFs.
    """
    return np.linalg.solve(K, F)

def visualize_truss(node_coords, connections, displacements=None, scale=1.0):
    """
    Visualizes the truss structure in 3D.

    Parameters:
    - node_coords: Numpy array of shape (num_nodes, 3) with original node coordinates.
    - connections: List of tuples defining truss connections.
    - displacements: Numpy array of shape (num_nodes, 3) with node displacements.
    - scale: Scaling factor for displacements in visualization.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Truss Fabric Simulation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Original Truss
    for (n1, n2) in connections:
        x = [node_coords[n1, 0], node_coords[n2, 0]]
        y = [node_coords[n1, 1], node_coords[n2, 1]]
        z = [node_coords[n1, 2], node_coords[n2, 2]]
        ax.plot(x, y, z, color='lightgray', linestyle='--', linewidth=1)

    # Deformed Truss
    if displacements is not None:
        deformed_coords = node_coords + displacements * scale
        for (n1, n2) in connections:
            x = [deformed_coords[n1, 0], deformed_coords[n2, 0]]
            y = [deformed_coords[n1, 1], deformed_coords[n2, 1]]
            z = [deformed_coords[n1, 2], deformed_coords[n2, 2]]
            ax.plot(x, y, z, color='blue', linewidth=2)

        # Plot nodes
        ax.scatter(deformed_coords[:, 0], deformed_coords[:, 1], deformed_coords[:, 2],
                   color='red', s=50, label='Deformed Nodes')

    # Plot original nodes
    ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2],
               color='black', s=20, label='Original Nodes')

    if displacements is not None:
        ax.legend()

    plt.show()

def main():
    # Parameters
    grid_size = (6, 6)          # 6 rows x 6 columns grid
    spacing = 1.0               # Distance between nodes
    E = 210e9                   # Young's Modulus in Pascals (e.g., steel)
    A = 0.01                    # Cross-sectional area in m^2
    poisson_ratio = 0.3         # Poisson's ratio (not directly used in truss)

    # Generate node coordinates
    node_coords = generate_grid(grid_size, spacing)

    # Define truss connections
    connections = get_truss_connections(grid_size)

    # Assemble global stiffness matrix
    print("Assembling global stiffness matrix...")
    K = assemble_global_stiffness(node_coords, connections, E, A)

    # Define force vector (gravity)
    num_nodes = node_coords.shape[0]
    num_dofs = 3 * num_nodes
    F = np.zeros(num_dofs)
    gravity = -9.81 * 1000  # Assuming mass of 1000 kg/m^3 for gravity force (N)
    F[2::3] = gravity       # Apply gravity in z-direction for all nodes

    # Define boundary conditions (fix two diagonal nodes)
    fixed_nodes = [0, num_nodes - 1]  # Top-left and bottom-right corners
    fixed_dofs = []
    for node in fixed_nodes:
        fixed_dofs.extend([3*node, 3*node+1, 3*node+2])  # Fix x, y, z DOFs

    # Apply boundary conditions
    print("Applying boundary conditions...")
    K_ff, F_ff, free_dofs = apply_boundary_conditions(K, F, fixed_dofs)

    # Check if K_ff is singular
    if np.linalg.matrix_rank(K_ff) < K_ff.shape[0]:
        print("Error: The reduced stiffness matrix is singular. Check boundary conditions and connectivity.")
        return

    # Solve for displacements
    print("Solving for displacements...")
    U_ff = solve_displacements(K_ff, F_ff)

    # Reconstruct full displacement vector
    U = np.zeros(num_dofs)
    U[free_dofs] = U_ff

    # Reshape displacements for visualization
    displacements = U.reshape(-1, 3)

    # Print displacement vector
    print("Displacement Vector (in meters):")
    for i in range(num_nodes):
        print(f"Node {i}: x={displacements[i,0]:.6e}, y={displacements[i,1]:.6e}, z={displacements[i,2]:.6e}")

    # Visualize the truss before and after deformation
    visualize_truss(node_coords, connections, displacements, scale=1.0)

if __name__ == "__main__":
    main()

