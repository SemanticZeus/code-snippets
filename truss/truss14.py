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

def compute_local_stiffness_3d(node1, node2, E, A):
    """
    Computes the local stiffness matrix for a 3D truss element.
    
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
    k_local = k * np.array([
        [ l*l, l*m, l*n, -l*l, -l*m, -l*n],
        [ l*m, m*m, m*n, -l*m, -m*m, -m*n],
        [ l*n, m*n, n*n, -l*n, -m*n, -n*n],
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
        k_local = compute_local_stiffness_3d(node1, node2, E, A)
        # DOF indices for node1 and node2
        dof_indices = [3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2]
        # Assemble into global K
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

def visualize_truss_3d(node_coords, connections, displacements, scale=1.0):
    """
    Visualizes the truss structure in 3D.
    
    Parameters:
    - node_coords: Numpy array of shape (num_nodes, 3) with original node coordinates.
    - connections: List of tuples defining truss connections.
    - displacements: Numpy array of shape (num_nodes, 3) with node displacements.
    - scale: Scaling factor for displacements in visualization.
    """
    deformed_coords = node_coords + displacements * scale
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Truss Fabric Simulation", fontsize=16)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Z", fontsize=12)
    
    # Original Truss in light gray dashed lines
    for (n1, n2) in connections:
        x = [node_coords[n1, 0], node_coords[n2, 0]]
        y = [node_coords[n1, 1], node_coords[n2, 1]]
        z = [node_coords[n1, 2], node_coords[n2, 2]]
        ax.plot(x, y, z, color='lightgray', linestyle='--', linewidth=1)
    
    # Deformed Truss in blue solid lines
    for (n1, n2) in connections:
        x = [deformed_coords[n1, 0], deformed_coords[n2, 0]]
        y = [deformed_coords[n1, 1], deformed_coords[n2, 1]]
        z = [deformed_coords[n1, 2], deformed_coords[n2, 2]]
        ax.plot(x, y, z, color='blue', linewidth=2)
    
    # Original Nodes in black
    ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2],
               color='black', s=20, label='Original Nodes')
    
    # Deformed Nodes in red
    ax.scatter(deformed_coords[:, 0], deformed_coords[:, 1], deformed_coords[:, 2],
               color='red', s=50, label='Deformed Nodes')
    
    ax.legend()
    plt.show()

def main_visualization_10x10():
    # Parameters for a 10x10 grid
    n = 10
    spacing = 1.0
    h = 1.0
    
    # Generate node coordinates
    node_coords = generate_nodes(n, spacing, h)
    
    # Define truss connections
    connections = get_truss_connections(n)
    
    # Visualize truss connections
    visualize_truss_connections(node_coords, connections, title=f"{n}x{n} Truss Fabric Connections")

if __name__ == "__main__":
    main_visualization_10x10()



def main():
    # Parameters
    n = 10                    # Grid size: 10x10
    spacing = 1.0             # Distance between nodes
    h = 1.0                   # Height difference between top and base grids
    E = 210e9                 # Young's Modulus in Pascals (e.g., steel)
    A = 0.01                  # Cross-sectional area in m^2
    gravity = 9.81            # Acceleration due to gravity in m/s^2
    mass_per_node = 1000      # Assumed mass per node in kg (for gravity force)
    
    # Generate node coordinates
    node_coords = generate_nodes(n, spacing, h)
    total_nodes = 2 * n * n
    print(f"Total nodes: {total_nodes}")
    
    # Define truss connections
    connections = get_truss_connections(n)
    print(f"Total truss elements: {len(connections)}")
    
    # Assemble global stiffness matrix
    print("Assembling global stiffness matrix...")
    K = assemble_global_stiffness(node_coords, connections, E, A)
    print("Global stiffness matrix assembled.")
    
    # Define force vector (gravity in -z for top grid nodes)
    F = np.zeros(3 * total_nodes)
    # Apply gravity to top grid nodes (first n*n nodes)
    for i in range(n * n):
        F[3*i + 2] = -mass_per_node * gravity  # F_z = -m * g
    print("Force vector defined.")
    
    # Define boundary conditions
    # Fix two diagonal top nodes: Node 0 (top-left) and Node n*n -1 (top-right)
    fixed_nodes = [0, n*n -1]  # Node indices: 0 and 99 for n=10
    fixed_dofs = []
    for node in fixed_nodes:
        fixed_dofs.extend([3*node, 3*node +1, 3*node +2])  # Fix x, y, z DOFs
    # Additionally, fix the z DOFs of all base grid nodes to prevent base movement
    base_offset = n * n
    for node in range(base_offset, 2 * n * n):
        fixed_dofs.append(3*node + 2)  # Fix z DOF
    print(f"Fixed nodes: {fixed_nodes}")
    print(f"Number of fixed DOFs: {len(fixed_dofs)}")
    
    # Apply boundary conditions
    print("Applying boundary conditions...")
    K_reduced, F_reduced, free_dofs = apply_boundary_conditions(K, F, fixed_dofs)
    print("Boundary conditions applied.")
    
    # Check if the reduced stiffness matrix is singular
    rank = np.linalg.matrix_rank(K_reduced)
    if rank < K_reduced.shape[0]:
        print("Error: The reduced stiffness matrix is singular. Check boundary conditions and connectivity.")
        return
    else:
        print("Reduced stiffness matrix is non-singular.")
    
    # Solve for displacements
    print("Solving for displacements...")
    U_reduced = solve_displacements(K_reduced, F_reduced)
    print("Displacements solved.")
    
    # Reconstruct full displacement vector
    U = np.zeros(3 * total_nodes)
    U[free_dofs] = U_reduced
    displacements = U.reshape(-1, 3)
    
    # Print displacement vector for fixed nodes (should be zero)
    print("\nDisplacement Vector (in meters) for Fixed Nodes:")
    for node in fixed_nodes:
        print(f"Node {node}: x={displacements[node,0]:.6e}, y={displacements[node,1]:.6e}, z={displacements[node,2]:.6e}")
    
    # Print displacement vector for a few free nodes
    print("\nDisplacement Vector (in meters) for Sample Free Nodes:")
    sample_nodes = [1, n, n*n -2, 2*n*n -1]  # Example nodes
    for node in sample_nodes:
        print(f"Node {node}: x={displacements[node,0]:.6e}, y={displacements[node,1]:.6e}, z={displacements[node,2]:.6e}")
    
    # Visualize the truss before and after deformation
    visualize_truss_3d(node_coords, connections, displacements, scale=1.0)

