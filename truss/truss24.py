import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

young_modulus = 10
poisson_ratio = -0.3 

length = 1.0
height = 1.0
num_elements_x = 10
num_elements_y = 10

x = np.linspace(0, length, num_elements_x + 1)
y = np.linspace(0, height, num_elements_y + 1)
points = np.array([[i, j] for j in y for i in x])
elements = []

for j in range(num_elements_y):
    for i in range(num_elements_x):
        n1 = j * (num_elements_x + 1) + i
        n2 = n1 + 1
        n3 = n1 + (num_elements_x + 1)
        n4 = n3 + 1
        elements.append((n1, n2))
        elements.append((n1, n3))
        elements.append((n1, n4))

boundary_nodes = list(range(len(points)))

num_dofs = len(points) * 3
K = lil_matrix((num_dofs, num_dofs))
F = np.zeros(num_dofs)

E = young_modulus
nu = poisson_ratio
D = (E / (1 - nu**2)) * np.array([
    [1, nu, 0],
    [nu, 1, 0],
    [0, 0, (1 - nu) / 2]
])

for el in elements:
    i, j = el
    xi, yi = points[i]
    xj, yj = points[j]
    length = np.sqrt((xj - xi)**2 + (yj - yi)**2)
    
    c = (xj - xi) / length
    s = (yj - yi) / length
    k_local = (E / (1 - nu**2) / length) * np.array([
        [c*c, c*s, 0, -c*c, -c*s, 0],
        [c*s, s*s, 0, -c*s, -s*s, 0],
        [0, 0, 0, 0, 0, 0],
        [-c*c, -c*s, 0, c*c, c*s, 0],
        [-c*s, -s*s, 0, c*s, s*s, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    
    dof_map = [3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]
    for a in range(6):
        for b in range(6):
            K[dof_map[a], dof_map[b]] += k_local[a, b]

for node in boundary_nodes:
    dof_x = 3 * node
    dof_y = 3 * node + 1
    dof_z = 3 * node + 2
    K[dof_x, :] = 0
    K[dof_y, :] = 0
    K[dof_z, :] = 0
    K[dof_x, dof_x] = 1
    K[dof_y, dof_y] = 1
    K[dof_z, dof_z] = 1
    F[dof_x] = 0
    F[dof_y] = 0
    F[dof_z] = 0

for i, p in enumerate(points):
    F[3*i + 2] -= 10

U = spsolve(K.tocsc(), F)
displacements = U.reshape(-1, 3)

scale = 10
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(points[:, 0], points[:, 1], np.zeros(len(points)), 'b.', label='Original nodes')
for el in elements:
    i, j = el
    ax.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], [0, 0], 'b--')

for el in elements:
    i, j = el
    ax.plot([points[i, 0] + scale * displacements[i, 0], points[j, 0] + scale * displacements[j, 0]],
            [points[i, 1] + scale * displacements[i, 1], points[j, 1] + scale * displacements[j, 1]],
            [scale * displacements[i, 2], scale * displacements[j, 2]], 'r-')

plt.show()

