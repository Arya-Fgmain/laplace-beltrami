import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized
import trimesh
import pyvista as pv

def tangent_basis(mesh, i):
    """
    Construct an orthonormal tangent basis
    at vertex i.
    """
    n = mesh.vertex_normals[i]
    n = n / np.linalg.norm(n)
    # Choose a vector not parallel to n
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2

def assemble_gfd_matrix(mesh):
    n = len(mesh.vertices)
    rows = []
    cols = []
    values = []
    for i in range(n):
        neighbors = np.asarray(mesh.vertex_neighbors[i])
        if len(neighbors) < 5:
            continue
        e1, e2 = tangent_basis(mesh, i)
        dp = mesh.vertices[neighbors] - mesh.vertices[i]
        x = dp @ e1
        y = dp @ e2
        A = np.column_stack([x, y, 0.5 * x**2, x * y, 0.5 * y**2])
        q = np.array([0, 0, 1, 0, 1])
        # Least-squares coefficients for the Laplacian
        w, _, rank, _ = np.linalg.lstsq(A.T, q, rcond=None)
        if rank < 5:
            continue
        # Off-diagonal coefficients
        rows.extend([i] * len(neighbors))
        cols.extend(neighbors)
        values.extend(w)
        # Diagonal coefficient
        rows.append(i)
        cols.append(i)
        values.append(-np.sum(w))
    L = sp.csr_matrix(
        (values, (rows, cols)),
        shape=(n, n)
    )
    return L

def solve_heat_gfd(mesh, u0, dt, T):
    L = assemble_gfd_matrix(mesh)
    n = len(mesh.vertices)
    steps = int(np.ceil(T / dt))
    dt = T / steps
    # Backward Euler system
    A = sp.eye(n, format="csc") - dt * L.tocsc()
    # Factorize once
    solve = factorized(A)
    u = np.array(u0, dtype=float).copy()
    for _ in range(steps):
        u = solve(u)
    return u

mesh = trimesh.load_mesh("SqPyramid.stl", force = "mesh")
'''Preprocessing'''
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
'''End of Preprocessing'''

V = mesh.vertices

# u0 = np.exp(-10 * ( (V[:, 0] - 0.5)**2 + (V[:, 1] - 0.0)**2 + (V[:, 2] - 0.5)**2 ))
center = V.mean(axis=0) # center of our sphere
R = np.mean(np.linalg.norm(V-center, axis=1))   # radius
north = V[np.argmax(V[:,2])]

sigma = 2.4 * R
u0 = np.exp(-np.sum((V - north)**2, axis=1) / (2 * sigma**2))
dt = 0.001
T = 0.1
u_final = solve_heat_gfd(mesh, u0, dt=dt, T=T)

# Convert trimesh faces into PyVista face format
faces = np.hstack([np.full((len(mesh.faces), 1), 3),mesh.faces]).astype(np.int64)
poly = pv.PolyData(mesh.vertices, faces)
poly.point_data["Heat"] = u0
plotter = pv.Plotter()
plotter.add_mesh(
    poly,
    scalars="Heat",
    cmap="inferno",
    show_edges=False
)

plotter.show()