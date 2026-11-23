import numpy as np
import trimesh
from scipy import sparse

mesh = trimesh.load('sphere.stl', force='mesh')

# recall mesh M = (V,F,E)
V = mesh.vertices
F = mesh.faces
E = mesh.edges_unique
n = V.shape[0]

i, j, k = F.T   # coordinates of the 1st, 2nd, and 3rd vertex of each triangle, respectively
vi, vj, vk = V[i], V[j], V[k] # corresponding coordinates of the aforementioned vertices

area_2x = np.linalg.norm(np.cross(vj - vi, vk - vi), axis=1)

# cotangents of three angles
# recall cot = a.b / ||a x b|| = a.b / (2*area)
# Einstein summation
# equiv. to np.sum((vj-vi)*(vk-vi), axis=1) / area_2x
# 'ik,ik' <=> multiply corresponding coordinates -> gives us the 'product' part of dot product, then '->i' <=> sum along the second axis (k) to get the dot product for each triangle
cot_i = np.einsum('ik, ik->i', vj-vi, vk-vi) / area_2x  
cot_j = np.einsum('ik, ik->i', vk-vj, vi-vj) / area_2x
cot_k = np.einsum('ik, ik->i', vi-vk, vj-vk) / area_2x

# print(cot_i.shape, cot_j.shape, cot_k.shape) 

# opposite edge handling
a = np.concatenate([j, k, i])
b = np.concatenate([k, i, j])
w = 0.5 * np.concatenate([cot_i, cot_j, cot_k])


# Symmetric weighted adjacency matrix
# construct matrix W, where W[i,j] = (1/2)(cot(alpha_ij) + cot(beta_ij)) for ij an edge in E
# np.r_ is for easy concatenation
W = sparse.coo_matrix(
    (np.r_[w, w], (np.r_[a, b], np.r_[b, a])),
        # the second and third np.r_ specify the row and column indices of non-zero entries in the sparse matrix
    shape=(n, n)
).tocsr()

# diagonal matrix
D = sparse.diags(
    # flatten the array to 1D before passing it to the sparse matrix constructor
    np.asarray(W.sum(axis=1)).ravel()
)

# Cotangent matrix
C = W - D

# Barycentric vertex areas
# A[i] = (1/3) * sum of areas of triangles of which vertex i is a corner
A = np.zeros(n)
for q in range(3):
    np.add.at(A, F[:,q], area_2x / 6)

M = sparse.diags(A)

# Discrete Laplace-Beltrami operator
# LB = np.linalg.inv(M) @ C
LB = sparse.diags(1.0 / A) @ C # yeah, diagonal matrices are fucking awesome


# ================= HEAT EQUATION VISUALIZATION =================
# Change to False to disable
if True:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.sparse.linalg import factorized

    # Initial condition: Gaussian hot spot at the top
    center = V.mean(axis=0)
    R = np.max(np.linalg.norm(V - center, axis=1))
    hotspot = V[np.argmax(V[:, 2])]

    sigma = 0.3 * R
    u = np.exp(-np.sum((V - hotspot)**2, axis=1) / (2*sigma**2))

    # Time discretization
    dt = 0.02 * R**2
    num_steps = 5

    # Backward Euler
    solve = factorized((M - dt*C).tocsc())

    states = [u.copy()]
    for _ in range(num_steps):
        u = solve(M @ u)
        states.append(u.copy())

    # Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.get_cmap('inferno')
    norm = Normalize(vmin=0, vmax=1)

    surface = Poly3DCollection(V[F], linewidths=0)
    ax.add_collection3d(surface)

    ax.set_xlim(center[0]-R, center[0]+R)
    ax.set_ylim(center[1]-R, center[1]+R)
    ax.set_zlim(center[2]-R, center[2]+R)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()

    def update(frame):
        face_temps = states[frame][F].mean(axis=1)
        surface.set_facecolor(cmap(norm(face_temps)))
        ax.set_title(f"Heat equation | t = {frame*dt:.3f}")
        return surface,

    ani = FuncAnimation(
        fig, update,
        frames=len(states),
        interval=60,
        blit=False
    )

    plt.show()
# ===============================================================