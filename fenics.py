import numpy as np
import trimesh
import scipy as sp
from mpi4py import MPI
from dolfinx import mesh, fem, plot
import basix.ufl
import ufl
from dolfinx.fem.petsc import LinearProblem
import pyvista as pv

# For some reason there's too many things named mesh so we gotta change it to fig
fig = trimesh.load("SqPyramid.stl", force = "mesh")

'''Preprocessing'''
# merge vertices at identical positions
fig.merge_vertices()

# remove faces w zero or near-zero area
fig.update_faces(fig.nondegenerate_faces())

# remove duplicate faces
fig.update_faces(fig.unique_faces())

# remove vertices that don't belong to any face (after we killed some faces above)
fig.remove_unreferenced_vertices()

# print("Cleaned vertices", len(mesh.vertices))
# print("Cleaned faces", len(mesh.faces))
'''End of Preprocessing'''

vertices = np.asarray(fig.vertices, dtype=np.float64)
faces = np.asarray(fig.faces, dtype=np.int64)

# Mesh vertices
V_mesh = vertices

# Center and approximate radius
center = V_mesh.mean(axis=0)
R = np.mean(np.linalg.norm(V_mesh - center, axis=1))

# Northernmost point
north = V_mesh[np.argmax(V_mesh[:, 2])]

# Gaussian initial condition
sigma = 2.4 * R

u0 = np.exp(-np.sum((V_mesh - north)**2, axis=1) / (2 * sigma**2))

element = basix.ufl.element(
    "Lagrange",
    "triangle",
    1,
    shape=(3,)
)

# Create FEniCSx mesh
domain = mesh.create_mesh(
    MPI.COMM_WORLD,
    faces,
    element,
    vertices
)

# Function space
V = fem.functionspace(domain, ("Lagrange", 1))

# Trial and test functions
w = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Time-stepping parameters
T = 0.1
dt= 0.001
num_steps = int(np.ceil(T / dt))
dt = T / num_steps

# Initial temperature
u_n = fem.Function(V)

# Interpolate Gaussian initial condition
u_n.interpolate(
    lambda x: np.exp(
        -np.sum((x - north[:, None])**2, axis=0)
        / (2 * sigma**2)
    )
)

u_n.x.scatter_forward()

# Current temperature
u = fem.Function(V)

# Weak formulation
a = (
    ufl.inner(w, v) * ufl.dx
    + dt * ufl.inner(ufl.grad(w), ufl.grad(v)) * ufl.dx
)

L = ufl.inner(u_n, v) * ufl.dx

# Construct linear problem
problem = LinearProblem(
    a,
    L,
    petsc_options_prefix="heat_equation",
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu"
    }
)

# Time-stepping
for n in range(num_steps):
    u = problem.solve()

    # Update previous solution
    u_n.x.array[:] = u.x.array
    u_n.x.scatter_forward()

print("Heat equation solved!")

# Visualize the thing
topology, cell_types, geometry = plot.vtk_mesh(V)

grid = pv.UnstructuredGrid(
    topology,
    cell_types,
    geometry
)

grid.point_data["Temperature"] = u.x.array

plotter = pv.Plotter()

plotter.add_mesh(
    grid,
    scalars="Temperature",
    cmap="inferno",
    show_edges=False,
    scalar_bar_args={"title": "Temperature"}
)

plotter.show()