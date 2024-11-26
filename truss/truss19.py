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
    x1,y1 = node1[0],node1[1]
    x2,y2 = node2[0],node2[1]
    c = (x2-x1)/length
    s = (y2-y1)/length
    k_local = k* np.array([
        [c*c, c*s, -c*c, -c*s],
        [c*s, s*s, -c*s, -s*s],
        [-c*c, -c*s, c*c, c*s],
        [-c*s, -s*s, c*s, s*s],
        ])
    for row in k_local:
        for x in row:
            print(f"{x:<20.2f}, ", end="")
        print()

    print()
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
    for dof in fixed_dofs:
        K[dof, :] = 0
        K[:, dof] = 0
        K[dof, dof] = 1
        F[dof] = 0
    return K, F


def visualize_truss_2d(node_coords, connections, displacements=None):
    plt.figure(figsize=(12, 10))
    for (n1, n2) in connections:
        x = [node_coords[n1, 0], node_coords[n2, 0]]
        y = [node_coords[n1, 1], node_coords[n2, 1]]
        plt.plot(x, y, color='blue', linewidth=1)
    if displacements is not None:
        deformed_coords = node_coords + displacements
        for (n1, n2) in connections:
            x = [deformed_coords[n1, 0], deformed_coords[n2, 0]]
            y = [deformed_coords[n1, 1], deformed_coords[n2, 1]]
            plt.plot(x, y, color='green', linewidth=2)
        plt.scatter(deformed_coords[:, 0], deformed_coords[:, 1], color='red', s=50)
    plt.scatter(node_coords[:, 0], node_coords[:, 1], color='black', s=20)
    plt.legend()
    plt.grid(True)
    plt.show()

n = 3
spacing = 1.0
E = 2000
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
fixed_nodes = [0, total_nodes -1, 1,n-1, total_nodes-5]
fixed_dofs = []

for node in fixed_nodes:
    fixed_dofs.extend([2*node, 2*node +1])
K, F = apply_boundary_conditions(K, F, fixed_dofs)

for row in K:
    for x in row:
        print(f"{x:<2.0f}, ", end="")
    print()
print()

num_dofs=2*n*n
free_dofs = np.setdiff1d(np.arange(num_dofs), fixed_dofs)
K_reduced = K[free_dofs][:, free_dofs]
F_reduced = F[free_dofs]

for row in K_reduced:
    for x in row:
        print(f"{x:<2.0f}, ", end="")
    print()

U = np.linalg.solve(K_reduced, F_reduced)
displacements = U.reshape(-1, 2)
sample_nodes = [1, n, n*n -2]
visualize_truss_2d(node_coords,connections)


