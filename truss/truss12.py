import numpy as np
import matplotlib.pyplot as plt

def generate_nodes(grid_size, spacing=1.0):
    """
    Generates node coordinates for a 2D grid.
    
    Parameters:
    - grid_size: Tuple (rows, cols) defining the grid dimensions.
    - spacing: Distance between adjacent nodes.
    
    Returns:
    - nodes: Numpy array of shape (num_nodes, 2) with (x, y) coordinates.
    """
    rows, cols = grid_size
    nodes = np.zeros((rows * cols, 2))
    for i in range(rows):
        for j in range(cols):
            node = i * cols + j
            nodes[node, 0] = j * spacing  # x-coordinate
            nodes[node, 1] = i * spacing  # y-coordinate
    return nodes

def get_truss_connections(grid_size):
    """
    Defines truss connections (horizontal and vertical) for the grid.
    
    Parameters:
    - grid_size: Tuple (rows, cols) defining the grid dimensions.
    
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

def compute_local_stiffness(node1, node2, E, A):
    """
    Computes the local stiffness matrix for a truss element in 2D.
    
    Parameters:
    - node1, node2: Numpy arrays of shape (2,) representing the coordinates of the two nodes.
    - E: Young's modulus.
    - A: Cross-sectional area.
    
    Returns:
    - k_local: Numpy array of shape (4,4) representing the local stiffness matrix.
    """
    L = node2 - node1
    length = np.linalg.norm(L)
    if length == 0:
        raise ValueError("Zero length truss element detected.")
    l, m = L / length  # Direction cosines
    k = (E * A) / length
    k_local = k * np.array([
        [ l*l, l*m, -l*l, -l*m],
        [ l*m, m*m, -l*m, -m*m],
        [-l*l, -l*m,  l*l,  l*m],
        [-l*m, -m*m,  l*m,  m*m]
    ])
    return k_local

def assemble_global_stiffness(nodes, connections, E, A):
    """
    Assembles the global stiffness matrix for the truss structure.
    
    Parameters:
    - nodes: Numpy array of shape (num_nodes, 2) with node coordinates.
    - connections: List of tuples defining truss connections.
    - E: Young's modulus.
    - A: Cross-sectional area.
    
    Returns:
    - K: Global stiffness matrix of shape (2*num_nodes, 2*num_nodes).
    """
    num_nodes = nodes.shape[0]
    K = np.zeros((2*num_nodes, 2*num_nodes))
    
    for (n1, n2) in connections:
        node1 = nodes[n1]
        node2 = nodes[n2]
        k_local = compute_local_stiffness(node1, node2, E, A)
        # DOF indices for node1 and node2
        dof_indices = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
        # Assemble into global K
        for i in range(4):
            for j in range(4):
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
    - K_reduced: Reduced stiffness matrix after applying boundary conditions.
    - F_reduced: Reduced force vector after applying boundary conditions.
    - free_dofs: Array of free DOF indices.
    """
    num_dofs = K.shape[0]
    free_dofs = np.setdiff1d(np.arange(num_dofs), fixed_dofs)
    K_reduced = K[np.ix_(free_dofs, free_dofs)]
    F_reduced = F[free_dofs]
    return K_reduced, F_reduced, free_dofs

def solve_displacements(K_reduced, F_reduced):
    """
    Solves for the displacements in the free DOFs.
    
    Parameters:
    - K_reduced: Reduced stiffness matrix.
    - F_reduced: Reduced force vector.
    
    Returns:
    - U_reduced: Displacements for the free DOFs.
    """
    return np.linalg.solve(K_reduced, F_reduced)

def visualize_truss_2d(nodes, connections, displacements=None, scale=1.0):
    """
    Visualizes the truss structure in 2D.
    
    Parameters:
    - nodes: Numpy array of shape (num_nodes, 2) with original node coordinates.
    - connections: List of tuples defining truss connections.
    - displacements: Numpy array of shape (num_nodes, 2) with node displacements.
    - scale: Scaling factor for displacements in visualization.
    """
    plt.figure(figsize=(8, 6))
    plt.title("Truss Fabric Simulation (2D)")
    plt.xlabel("X")
    plt.ylabel("Y")
    
    # Original Truss
    for (n1, n2) in connections:
        x = [nodes[n1, 0], nodes[n2, 0]]
        y = [nodes[n1, 1], nodes[n2, 1]]
        plt.plot(x, y, color='lightgray', linestyle='--', linewidth=1)
    
    # Deformed Truss
    if displacements is not None:
        deformed_nodes = nodes + displacements * scale
        for (n1, n2) in connections:
            x = [deformed_nodes[n1, 0], deformed_nodes[n2, 0]]
            y = [deformed_nodes[n1, 1], deformed_nodes[n2, 1]]
            plt.plot(x, y, color='blue', linewidth=2)
        
        # Plot nodes
        plt.scatter(deformed_nodes[:, 0], deformed_nodes[:, 1], color='red', label='Deformed Nodes')
    
    # Plot original nodes
    plt.scatter(nodes[:, 0], nodes[:, 1], color='black', label='Original Nodes')
    
    if displacements is not None:
        plt.legend()
    
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def main():
    # Parameters
    grid_size = (2, 2)          # 2 rows x 2 columns grid
    spacing = 1.0               # Distance between nodes
    E = 210e9                   # Young's Modulus in Pascals (e.g., steel)
    A = 0.01                    # Cross-sectional area in m^2
    
    # Generate node coordinates
    nodes = generate_nodes(grid_size, spacing)
    
    # Define truss connections
    connections = get_truss_connections(grid_size)
    
    # Assemble global stiffness matrix
    print("Assembling global stiffness matrix...")
    K = assemble_global_stiffness(nodes, connections, E, A)
    
    # Define force vector (gravity)
    num_nodes = nodes.shape[0]
    num_dofs = 2 * num_nodes
    F = np.zeros(num_dofs)
    gravity = -9.81 * 1000  # Assuming mass of 1000 kg/m^3 for gravity force (N)
    F[1::2] = gravity       # Apply gravity in y-direction for all nodes
    
    # Define boundary conditions (fix two diagonal nodes)
    fixed_nodes = [0, num_nodes - 1]  # Top-left and bottom-right corners
    fixed_dofs = []
    for node in fixed_nodes:
        fixed_dofs.extend([2*node, 2*node+1])  # Fix x and y DOFs
    
    # Apply boundary conditions
    print("Applying boundary conditions...")
    K_reduced, F_reduced, free_dofs = apply_boundary_conditions(K, F, fixed_dofs)
    
    # Check if K_reduced is singular
    if np.linalg.matrix_rank(K_reduced) < K_reduced.shape[0]:
        print("Error: The reduced stiffness matrix is singular. Check boundary conditions and connectivity.")
        return
    
    # Solve for displacements
    print("Solving for displacements...")
    U_reduced = solve_displacements(K_reduced, F_reduced)
    
    # Reconstruct full displacement vector
    U = np.zeros(num_dofs)
    U[free_dofs] = U_reduced
    displacements = U.reshape(-1, 2)
    
    # Print displacement vector
    print("\nDisplacement Vector (in meters):")
    for i in range(num_nodes):
        print(f"Node {i}: x={displacements[i,0]:.6e}, y={displacements[i,1]:.6e}")
    
    # Visualize the truss before and after deformation
    visualize_truss_2d(nodes, connections, displacements, scale=1.0)

if __name__ == "__main__":
    main()

