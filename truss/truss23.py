import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Parameters for auxetic material properties
young_modulus = 2  # Young's modulus
poisson_ratio = .1  # Negative Poisson's ratio for auxetic material

# Create a mesh for a 2D fabric using trusses
length = 1.0
height = 1.0
num_elements_x = 10
num_elements_y = 10

# Generate mesh points (grid)
x = np.linspace(0, length, num_elements_x + 1)
y = np.linspace(0, height, num_elements_y + 1)
points = np.array([[i, j] for j in y for i in x])
elements = []

# Generate truss elements in the grid
for j in range(num_elements_y):
    for i in range(num_elements_x):
        n1 = j * (num_elements_x + 1) + i
        n2 = n1 + 1
        n3 = n1 + (num_elements_x + 1)
        n4 = n3 + 1
        elements.append((n1, n2))  # Horizontal element
        elements.append((n1, n3))  # Vertical element
        elements.append((n1, n4))  # Diagonal element

# Define boundary conditions (fixed at all edges)
boundary_nodes = [i for i, p in enumerate(points) if np.isclose(p[0], 0) or np.isclose(p[0], length) or np.isclose(p[1], 0) or np.isclose(p[1], height)]

# Assemble stiffness matrix and force vector
num_dofs = len(points) * 3  # 3 DOFs per node (x, y, z)
K = lil_matrix((num_dofs, num_dofs))
F = np.zeros(num_dofs)

# Material matrix for plane stress
E = young_modulus
nu = poisson_ratio
D = (E / (1 - nu**2)) * np.array([
    [1, nu, 0],
    [nu, 1, 0],
    [0, 0, (1 - nu) / 2]
])

# Loop over elements to assemble K and F
for el in elements:
    i, j = el
    xi, yi = points[i]
    xj, yj = points[j]
    length = np.sqrt((xj - xi)**2 + (yj - yi)**2)
    
    # Local stiffness matrix for a 2D truss element considering Poisson's ratio
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
    
    # Assembly into global stiffness matrix
    dof_map = [3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]
    for a in range(6):
        for b in range(6):
            K[dof_map[a], dof_map[b]] += k_local[a, b]

# Apply boundary conditions
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

# Apply external force (gravity-like in z-direction)
for i, p in enumerate(points):
    F[3*i + 2] -= 10  # Apply force in the negative z-direction

# Solve system of equations
U = spsolve(K.tocsc(), F)

# Extract displacements
displacements = U.reshape(-1, 3)

# Plot deformed shape in 3D
scale = 10  # Scaling factor for visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(points[:, 0], points[:, 1], np.zeros(len(points)), 'b.', label='Original nodes')
for el in elements:
    i, j = el
    ax.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], [0, 0], 'b--')

# Plot deformed shape
for el in elements:
    i, j = el
    ax.plot([points[i, 0] + scale * displacements[i, 0], points[j, 0] + scale * displacements[j, 0]],
            [points[i, 1] + scale * displacements[i, 1], points[j, 1] + scale * displacements[j, 1]],
            [scale * displacements[i, 2], scale * displacements[j, 2]], 'r-')

ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_zlabel('Z coordinate')
ax.set_title("Deformation of 2D Fabric Using Trusses (3D Visualization)")
plt.legend(['Original nodes', 'Original elements', 'Deformed elements'])
plt.show()

