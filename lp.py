import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.linalg import eigsh

mesh = trimesh.load('dragon_vrip.ply', force='mesh')

'''
some preprocessing to prevent more downstream errors (e.g. 0 area when calculating cotangent laplacian)
'''
# print("Original vertices", len(mesh.vertices))
# print("Original faces", len(mesh.faces))

# merge vertices at identical positions
mesh.merge_vertices()

# remove faces w zero or near-zero area
mesh.update_faces(mesh.nondegenerate_faces())

# remove duplicate faces
mesh.update_faces(mesh.unique_faces())

# remove vertices that don't belong to any face (after we killed some faces above)
mesh.remove_unreferenced_vertices()

# print("Cleaned vertices", len(mesh.vertices))
# print("Cleaned faces", len(mesh.faces))
'''
that should be all the preprocessing we need
'''

# recall mesh M = (V,F,E)
V = mesh.vertices
F = mesh.faces
E = mesh.edges_unique
n = V.shape[0]

i, j, k = F.T   # coordinates of the 1st, 2nd, and 3rd vertex of each triangle, respectively
vi, vj, vk = V[i], V[j], V[k] # corresponding coordinates of the aforementioned vertices

area_2x = np.linalg.norm(np.cross(vj - vi, vk - vi), axis=1)

assert np.all(area_2x > 0), "Degenerate triangles remaining..."

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

assert np.all(np.isfinite(C.data)), "Invalid cotangent weights present..."

# Barycentric vertex areas
# A[i] = (1/3) * sum of areas of triangles of which vertex i is a corner
A = np.zeros(n)
for q in range(3):
    np.add.at(A, F[:,q], area_2x / 6)

assert np.all(A > 0), "Mass matrix is singular..."


M = sparse.diags(A)

# Discrete Laplace-Beltrami operator
# LB = np.linalg.inv(M) @ C
LB = sparse.diags(1.0 / A) @ C # yeah, diagonal matrices are fucking awesome


### spectral solve

r = 50  # number of eigenvalues to compute

mu, Phi = eigsh(-C, k=r, M=M, sigma=-1e-8, which='LM')
    # LM = 'Largest in Magnitude'

# sort in ascending order
sorted_indices = np.argsort(mu)
mu = np.maximum(mu[sorted_indices], 0.0)
Phi = Phi[:, sorted_indices]

# sample initial condition
center = V.mean(axis=0) # center of our sphere
R = np.mean(np.linalg.norm(V-center, axis=1))   # radius
north = V[np.argmax(V[:,2])]

sigma = 2.4 * R
u0 = np.exp(-np.sum((V - north)**2, axis=1) / (2 * sigma**2))

# projection coefficients
proj_coeffs = Phi.T @ (A * u0)

# diffusivity coefficient
k = 1.0

def u(t):
    # u(t) = \sum_{i} a_i \cdot e^{-k \cdot mu \cdot t}
    return Phi @ (proj_coeffs * np.exp(-k * mu * t))


import plotly.graph_objects as go

# Choose times relative to sphere radius and diffusivity
times = np.linspace(0, 0.5 * R**2 / k, 40)
solutions = [u(t) for t in times]

# Initial 3D mesh
fig = go.Figure(
    data=[go.Mesh3d(
        x=V[:, 0], y=V[:, 1], z=V[:, 2],
        i=F[:, 0], j=F[:, 1], k=F[:, 2],
        intensity=solutions[0],
        intensitymode='vertex',
        colorscale='Inferno',
        cmin=0, cmax=1,
        colorbar=dict(title='Temperature')
    )]
)

# Animation frames
fig.frames = [
    go.Frame(
        data=[go.Mesh3d(intensity=values)],
        traces=[0],
        name=str(q)
    )
    for q, values in enumerate(solutions)
]

# Time slider
fig.update_layout(
    title='Spectral Heat Diffusion on a Sphere',
    scene=dict(aspectmode='data'),
    sliders=[dict(
        currentvalue=dict(prefix='Time: '),
        steps=[
            dict(
                label=f'{t:.3f}',
                method='animate',
                args=[
                    [str(q)],
                    dict(mode='immediate',
                         frame=dict(duration=0, redraw=True),
                         transition=dict(duration=0))
                ]
            )
            for q, t in enumerate(times)
        ]
    )],
    updatemenus=[dict(
        type='buttons',
        buttons=[
            dict(
                label='Play',
                method='animate',
                args=[None, dict(
                    frame=dict(duration=100, redraw=True),
                    fromcurrent=True,
                    transition=dict(duration=0)
                )]
            ),
            dict(
                label='Pause',
                method='animate',
                args=[[None], dict(
                    mode='immediate',
                    frame=dict(duration=0, redraw=False)
                )]
            )
        ]
    )]
)

fig.write_html('sphere_heat_spectral.html')
fig.show()
