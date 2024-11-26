import numpy as np
import matplotlib.pyplot as plt

def generate_nodes(n, spacing=1.0):
    node_coords = np.zeros((n * n, 2))
    for i in range(n):
        for j in range(n):
            node = i * n + j
            node_coords[node, 0] = j * spacing
            node_coords[node, 1] = i * spacing
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

def compute_local_stiffness_2d(node1, node2, E, A):
    L = node2 - node1
    length = np.linalg.norm(L)
    l, m = L / length
    k = (E * A) / length
    k_local = k * np.array([
        [ l*l, l*m, -l*l, -l*m],
        [ l*m, m*m, -l*m, -m*m],
        [-l*l, -l*m,  l*l,  l*m],
        [-l*m, -m*m,  l*m,  m*m]
    ])
    return k_local

def assemble_global_stiffness(node_coords, connections, E, A):
    num_nodes = node_coords.shape[0]
    K = np.zeros((2*num_nodes, 2*num_nodes))
    for (n1, n2) in connections:
        node1 = node_coords[n1]
        node2 = node_coords[n2]
        k_local = compute_local_stiffness_2d(node1, node2, E, A)
        dof_indices = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
        for i in range(4):
            for j in range(4):
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

def visualize_truss_2d(node_coords, connections, displacements=None, scale=1.0, title="2D Truss Fabric Visualization"):
    plt.figure(figsize=(12, 10))
    plt.title(title, fontsize=16)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    for (n1, n2) in connections:
        x = [node_coords[n1, 0], node_coords[n2, 0]]
        y = [node_coords[n1, 1], node_coords[n2, 1]]
        plt.plot(x, y, color='blue', linewidth=1)
    if displacements is not None:
        deformed_coords = node_coords + displacements * scale
        for (n1, n2) in connections:
            x = [deformed_coords[n1, 0], deformed_coords[n2, 0]]
            y = [deformed_coords[n1, 1], deformed_coords[n2, 1]]
            plt.plot(x, y, color='green', linewidth=2)
        plt.scatter(deformed_coords[:, 0], deformed_coords[:, 1], color='red', s=50, label='Deformed Nodes')
    plt.scatter(node_coords[:, 0], node_coords[:, 1], color='black', s=20, label='Original Nodes')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def main():
    n = 3
    spacing = 1.0
    E = 210e9
    A = 0.01
    gravity = 9.81
    mass_per_node = 1000
    node_coords = generate_nodes(n, spacing)
    total_nodes = n * n
    connections = get_truss_connections(n)
    K = assemble_global_stiffness(node_coords, connections, E, A)
    F = np.zeros(2 * total_nodes)
    for i in range(total_nodes):
        F[2*i + 1] = -mass_per_node * gravity
    fixed_nodes = [0, total_nodes -1]
    fixed_dofs = []
    for node in fixed_nodes:
        fixed_dofs.extend([2*node, 2*node +1])
    K_reduced, F_reduced, free_dofs = apply_boundary_conditions(K, F, fixed_dofs)
    for row in K_reduced:
        for x in row:
            print(f"{x:.2f}, ", end="")
        print()
    rank = np.linalg.matrix_rank(K_reduced)
    if rank < K_reduced.shape[0]:
        print("Error: The reduced stiffness matrix is singular. Check boundary conditions and connectivity.")
        return
    U_reduced = solve_displacements(K_reduced, F_reduced)
    U = np.zeros(2 * total_nodes)
    U[free_dofs] = U_reduced
    displacements = U.reshape(-1, 2)
    print("\nDisplacement Vector (in meters) for Fixed Nodes:")
    for node in fixed_nodes:
        print(f"Node {node}: x={displacements[node,0]:.6e}, y={displacements[node,1]:.6e}")
    print("\nDisplacement Vector (in meters) for Sample Free Nodes:")
    sample_nodes = [1, n, n*n -2]
    for node in sample_nodes:
        print(f"Node {node}: x={displacements[node,0]:.6e}, y={displacements[node,1]:.6e}")
    visualize_truss_2d(node_coords, connections, displacements, scale=1.0, title="10x10 Truss Fabric Deformation")

if __name__ == "__main__":
    main()

