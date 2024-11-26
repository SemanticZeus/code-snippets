import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_nodes(n, spacing=1.0, h=1.0):
    node_coords = np.zeros((2 * n * n, 3))
    for i in range(n):
        for j in range(n):
            x = j * spacing
            y = i * spacing
            node_coords[i * n + j, 0] = x
            node_coords[i * n + j, 1] = y
            node_coords[i * n + j, 2] = 0.0
            node_coords[n * n + i * n + j, 0] = x
            node_coords[n * n + i * n + j, 1] = y
            node_coords[n * n + i * n + j, 2] = -h
    return node_coords

def get_truss_connections(n):
    connections = []
    total_nodes = 2 * n * n
    for i in range(n):
        for j in range(n):
            current = i * n + j
            if j < n - 1:
                right = current + 1
                connections.append((current, right))
            if i < n - 1:
                up = current + n
                connections.append((current, up))
    base_offset = n * n
    for i in range(n):
        for j in range(n):
            current = base_offset + i * n + j
            if j < n - 1:
                right = current + 1
                connections.append((current, right))
            if i < n - 1:
                up = current + n
                connections.append((current, up))
    for i in range(n * n):
        top = i
        base = base_offset + i
        connections.append((top, base))
    return connections

def compute_local_stiffness_3d(node1, node2, E, A):
    L = node2 - node1
    length = np.linalg.norm(L)
    l, m, n = L / length
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
    num_nodes = node_coords.shape[0]
    K = np.zeros((3 * num_nodes, 3 * num_nodes))
    for (n1, n2) in connections:
        node1 = node_coords[n1]
        node2 = node_coords[n2]
        k_local = compute_local_stiffness_3d(node1, node2, E, A)
        dof_indices = [3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2]
        for i in range(6):
            for j in range(6):
                K[dof_indices[i], dof_indices[j]] += k_local[i, j]
    return K

def apply_boundary_conditions(K, F, fixed_dofs):
    num_dofs = K.shape[0]
    free_dofs = np.setdiff1d(np.arange(num_dofs), fixed_dofs)
    K_reduced = K[np.ix_(free_dofs, free_dofs)]
    F_reduced = F[free_dofs]
    return K_reduced, F_reduced, free_dofs

def solve_displacements(K_reduced, F_reduced):
    return np.linalg.solve(K_reduced, F_reduced)

def visualize_truss_3d(node_coords, connections, displacements, scale=1.0, title="Truss Fabric Visualization"):
    deformed_coords = node_coords + displacements * scale
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Z", fontsize=12)
    for (n1, n2) in connections:
        x = [node_coords[n1, 0], node_coords[n2, 0]]
        y = [node_coords[n1, 1], node_coords[n2, 1]]
        z = [node_coords[n1, 2], node_coords[n2, 2]]
        ax.plot(x, y, z, color='lightgray', linestyle='--', linewidth=1)
    for (n1, n2) in connections:
        x = [deformed_coords[n1, 0], deformed_coords[n2, 0]]
        y = [deformed_coords[n1, 1], deformed_coords[n2, 1]]
        z = [deformed_coords[n1, 2], deformed_coords[n2, 2]]
        ax.plot(x, y, z, color='blue', linewidth=2)
    ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2],
               color='black', s=20, label='Original Nodes')
    ax.scatter(deformed_coords[:, 0], deformed_coords[:, 1], deformed_coords[:, 2],
               color='red', s=50, label='Deformed Nodes')
    ax.legend()
    plt.show()

def main():
    n = 10
    spacing = 1.0
    E = 210e9
    A = 0.01
    gravity = 9.81
    mass_per_node = 1000
    node_coords = generate_nodes(n, spacing)
    total_nodes = n * n
    connections = get_truss_connections(n)
    K = assemble_global_stiffness(node_coords, connections, E, A)
    F = np.zeros(3 * total_nodes)
    for i in range(total_nodes):
        F[3*i + 2] = -mass_per_node * gravity
    fixed_nodes = [0, total_nodes -1]
    fixed_dofs = []
    for node in fixed_nodes:
        fixed_dofs.extend([3*node, 3*node +1, 3*node +2])
    K_reduced, F_reduced, free_dofs = apply_boundary_conditions(K, F, fixed_dofs)
    rank = np.linalg.matrix_rank(K_reduced)
    if rank < K_reduced.shape[0]:
        print("Error: The reduced stiffness matrix is singular. Check boundary conditions and connectivity.")
        return
    U_reduced = solve_displacements(K_reduced, F_reduced)
    U = np.zeros(3 * total_nodes)
    U[free_dofs] = U_reduced
    displacements = U.reshape(-1, 3)
    print("\nDisplacement Vector (in meters) for Fixed Nodes:")
    for node in fixed_nodes:
        print(f"Node {node}: x={displacements[node,0]:.6e}, y={displacements[node,1]:.6e}, z={displacements[node,2]:.6e}")
    print("\nDisplacement Vector (in meters) for Sample Free Nodes:")
    sample_nodes = [1, n, n*n -2]
    for node in sample_nodes:
        print(f"Node {node}: x={displacements[node,0]:.6e}, y={displacements[node,1]:.6e}, z={displacements[node,2]:.6e}")
    visualize_truss_3d(node_coords, connections, displacements, scale=1.0, title="10x10 Truss Fabric Deformation")

if __name__ == "__main__":
    main()

