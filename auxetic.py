import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Parameters for auxetic material properties
young_modulus = 1e5  # Young's modulus
poisson_ratio = -0.3  # Negative Poisson's ratio for auxetic material

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
num_dofs = len(points) * 2
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
    
    # Local stiffness matrix for a 2D truss element
    c = (xj - xi) / length
    s = (yj - yi) / length
    k_local = (E / length) * np.array([
        [c*c, c*s, -c*c, -c*s],
        [c*s, s*s, -c*s, -s*s],
        [-c*c, -c*s, c*c, c*s],
        [-c*s, -s*s, c*s, s*s]
    ])
    
    # Assembly into global stiffness matrix
    dof_map = [2*i, 2*i+1, 2*j, 2*j+1]
    for a in range(4):
        for b in range(4):
            K[dof_map[a], dof_map[b]] += k_local[a, b]

# Apply boundary conditions
for node in bottom_nodes:
    dof_x = 2 * node
    dof_y = 2 * node + 1
    K[dof_x, :] = 0
    K[dof_y, :] = 0
    K[dof_x, dof_x] = 1
    K[dof_y, dof_y] = 1
    F[dof_x] = 0
    F[dof_y] = 0

# Apply external force (gravity-like)
for i, p in enumerate(points):
    F[2*i + 1] -= 10  # Apply force in the negative y-direction

# Solve system of equations
U = spsolve(K.tocsc(), F)

# Extract displacements
displacements = U.reshape(-1, 2)

# Plot deformed shape
scale = 10  # Scaling factor for visualization
plt.figure()
plt.plot(points[:, 0], points[:, 1], 'b--', label='Original shape')
plt.plot(points[:, 0] + scale * displacements[:, 0], points[:, 1] + scale * displacements[:, 1], 'r-', label='Deformed shape')
plt.axis('equal')
plt.legend()
plt.title("Displacement of Auxetic Material (Dom-like Shape)")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.show()

