import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import factorized
import pyvista as pv

def assemble_matrices(vertices, faces):
    N = len(vertices)
    rows = []
    cols = []
    K_data = []
    M_data = []
    for face in faces:
        points = vertices[face]
        K_local, M_local = local_matrices(points)
        for i in range(3):
            for j in range(3):
                rows.append(face[i])
                cols.append(face[j])

                K_data.append(K_local[i, j])
                M_data.append(M_local[i, j])
    K = coo_matrix(
        (K_data, (rows, cols)),
        shape=(N, N)
    ).tocsr()
    M = coo_matrix(
        (M_data, (rows, cols)),
        shape=(N, N)
    ).tocsr()
    return K, M

def local_matrices(points):
    p1, p2, p3 = points
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    twice_area = np.linalg.norm(cross)
    max_edge_sq = max(
        np.dot(v1, v1),
        np.dot(v2, v2),
        np.dot(p3 - p2, p3 - p2)
    )
    # Scale-independent degeneracy check
    if max_edge_sq == 0 or twice_area <= 1e-12 * max_edge_sq:
        raise ValueError("Degenerate triangle encountered.")
    e1 = v1 / np.linalg.norm(v1)
    normal = cross / twice_area
    e2 = np.cross(normal, e1)
    a = np.linalg.norm(v1)
    b = np.dot(v2, e1)
    c = np.dot(v2, e2)
    area = 0.5 * twice_area
    G = np.array([
        [-c, b - a],
        [ c, -b],
        [ 0, a]
    ]) / (2 * area)
    K = area * (G @ G.T)
    M = (area / 12) * np.array([
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2]
    ])
    return K, M

def solve_heat_equation(K, M, u0, dt, steps):
    # Construct system matrix
    A = M + dt * K
    # Factorize once
    solve = factorized(A.tocsc())
    u = u0.copy()
    for n in range(steps):
        rhs = M @ u
        u = solve(rhs)
    return u


mesh = trimesh.load_mesh("stanford-bunny.obj", force = "mesh")

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


vertices = mesh.vertices
faces = mesh.faces

# Approximate sphere center
mesh_center = vertices.mean(axis=0)

# Approximate sphere radius
radius = np.mean(
    np.linalg.norm(vertices - mesh_center, axis=1)
)

# Average edge length
edge_lengths = mesh.edges_unique_length
h = edge_lengths.mean()

K, M = assemble_matrices(vertices, faces)

sigma = 1.0 * radius
center = vertices[0]
distances = np.linalg.norm(vertices - center, axis=1)
u0 = np.exp(-distances**2 / (2 * sigma**2))

print("Started solving heat equation")
# Solve
dt = 1.0
steps = 100
u = solve_heat_equation(K, M, u0, dt, steps)
initial_heat = np.ones(len(u0)) @ (M @ u0)
final_heat = np.ones(len(u)) @ (M @ u)

# Convert trimesh faces into PyVista face format
pv_faces = np.column_stack((
    np.full(len(faces), 3),
    faces
)).ravel()
surface = pv.PolyData(vertices, pv_faces)
surface.point_data["Heat"] = u0
plotter = pv.Plotter()
plotter.add_mesh(
    surface,
    scalars="Heat",
    cmap="inferno",
    clim=[0, 1],
    show_edges=False
)
plotter.show()