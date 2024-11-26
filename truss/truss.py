import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Parameters for auxetic material properties
young_modulus = 1e5  # Young's modulus
poisson_ratio = 0.3  # Negative Poisson's ratio for auxetic material

# Create a mesh for a dom-like shape (semi-sphere in 2D)
radius = 1.0
num_elements = 32

# Generate mesh points (semi-circle)
angles = np.linspace(0, np.pi, num_elements)
points = np.array([[radius * np.cos(angle), radius * np.sin(angle)] for angle in angles])
elements = [(i, i + 1) for i in range(len(points) - 1)]

# Define boundary conditions (fixed at the bottom)
bottom_nodes = [i for i, p in enumerate(points) if np.isclose(p[1], 0)]

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
for node in bottom_nodes:
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

# Apply external force (gravity-like)
for i, p in enumerate(points):
    F[3*i + 1] -= 10  # Apply force in the negative y-direction

# Solve system of equations
U = spsolve(K.tocsc(), F)

# Extract displacements
displacements = U.reshape(-1, 3)

# Plot deformed shape in 3D
scale = 10  # Scaling factor for visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(points[:, 0], points[:, 1], np.zeros(len(points)), 'b--', label='Original shape')
ax.plot(points[:, 0] + scale * displacements[:, 0],
        points[:, 1] + scale * displacements[:, 1],
        scale * displacements[:, 2], 'r-', label='Deformed shape')
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_zlabel('Z coordinate')
ax.set_title("Displacement of Auxetic Material (Dom-like Shape)")
plt.legend()
plt.show()

