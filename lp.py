'''Cotangent mesh operators and spectral heat diffusion.'''

from numbers import Integral
from typing import NamedTuple

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, expm_multiply


def preprocess_mesh(mesh):
    '''Return a cleaned copy of a triangle mesh in its original coordinates.'''
    # some preprocessing to prevent more downstream errors (e.g. 0 area when calculating cotangent laplacian)
    mesh = mesh.copy()
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
    if len(mesh.faces) == 0:
        raise ValueError('The mesh has no nondegenerate triangles.')

    # that should be all the preprocessing we need
    return mesh


def compute_cotangent_laplacian(mesh):
    '''Return C and the lumped mass matrix M for M u' = diffusivity * C u.

    C is negative semidefinite. On meshes with boundary, this assembly uses
    natural (zero-flux) boundary conditions.
    '''
    # recall mesh M = (V,F,E)
    V = mesh.vertices
    F = mesh.faces
    n = V.shape[0]

    i, j, k = F.T   # indices of the 1st, 2nd, and 3rd vertex of each triangle, respectively
    vi, vj, vk = V[i], V[j], V[k]  # corresponding coordinates of the aforementioned vertices

    area_2x = np.linalg.norm(np.cross(vj - vi, vk - vi), axis=1)
    if not np.all(np.isfinite(area_2x) & (area_2x > 0)):
        raise ValueError('The mesh contains invalid or degenerate triangles.')

    # cotangents of three angles
    # recall cot = a.b / ||a x b|| = a.b / (2*area)
    # Einstein summation
    # equiv. to np.sum((vj-vi)*(vk-vi), axis=1) / area_2x
    # 'ik,ik' <=> multiply corresponding coordinates -> gives us the 'product' part of dot product,
    # then '->i' <=> sum along the second axis (k) to get the dot product for each triangle
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
        shape=(n, n),
    ).tocsr()

    # diagonal matrix
    D = sparse.diags(
        # flatten the array to 1D before passing it to the sparse matrix constructor
        np.asarray(W.sum(axis=1)).ravel()
    )

    # Cotangent matrix
    C = W - D
    if not np.all(np.isfinite(C.data)):
        raise ValueError('The mesh produced nonfinite cotangent weights.')

    # Barycentric vertex areas
    # A[i] = (1/3) * sum of areas of triangles of which vertex i is a corner
    A = np.zeros(n)
    for q in range(3):
        np.add.at(A, F[:,q], area_2x / 6)

    if n == 0 or not np.all(np.isfinite(A) & (A > 0)):
        raise ValueError('Each vertex must have a positive, finite lumped area.')
    M = sparse.diags(A, format='csr')

    # Discrete Laplace-Beltrami operator
    # LB = np.linalg.inv(M) @ C
    # LB = sparse.diags(1.0 / A) @ C # yeah, diagonal matrices are fucking awesome
    return C, M


### spectral solve

def compute_eigenmodes(C, M, *, n_modes):
    '''Return ascending eigenvalues and M-orthonormal modes of -C phi = mu M phi.'''
    n = C.shape[0]
    r = n_modes  # number of eigenvalues to compute
    if not isinstance(r, Integral) or not 1 <= r <= n:
        raise ValueError(f'n_modes must be an integer between 1 and {n}.')

    if r == n:
        mu, Phi = eigh(-C.toarray(), M.toarray())
    else:
        # Keep the shift below the zero mode and relative to the mesh scale.
        shift = -1e-8 * np.max(-C.diagonal() / M.diagonal())
        start = np.random.default_rng(0).standard_normal(n)
        # LM = 'Largest in Magnitude'
        mu, Phi = eigsh(-C, k=r, M=M, sigma=shift, which='LM', v0=start)

    # sort in ascending order
    sorted_indices = np.argsort(mu)
    mu = np.maximum(mu[sorted_indices], 0.0)
    Phi = Phi[:, sorted_indices]
    return mu, Phi


def _heat_inputs(M, u0, diffusivity):
    u0 = np.asarray(u0, dtype=float)
    if u0.shape != (M.shape[0],) or not np.all(np.isfinite(u0)):
        raise ValueError('u0 must contain one finite value per vertex.')
    if not np.isfinite(diffusivity) or diffusivity < 0:
        raise ValueError('Diffusivity must be finite and nonnegative.')
    return u0


def _heat_times(times):
    times = np.asarray(times, dtype=float)
    if times.ndim > 1 or not np.all(np.isfinite(times) & (times >= 0)):
        raise ValueError('Supply a finite nonnegative time or a 1D array of times.')
    return times


def make_heat_solution(eigenvalues, modes, M, u0, *, diffusivity=1.0):
    '''Project u0 once and return u(t). Array input returns one row per time.'''
    u0 = _heat_inputs(M, u0, diffusivity)

    # projection coefficients
    proj_coeffs = modes.T @ (M @ u0)

    def u(t):
        # u(t) = \sum_{i} a_i \cdot e^{-k \cdot mu \cdot t}
        t = _heat_times(t)
        decays = np.exp(-diffusivity * t[..., None] * eigenvalues)
        return (decays * proj_coeffs) @ modes.T

    return u


def solve_discrete_heat(C, M, u0, times, *, diffusivity=1.0):
    '''Evaluate the full mesh ODE using a sparse matrix exponential action.'''
    u0 = _heat_inputs(M, u0, diffusivity)
    times = _heat_times(times)
    if diffusivity == 0:
        return np.broadcast_to(u0, times.shape + u0.shape).copy()

    # Evolve sqrt(M) u with the symmetric generator sqrt(M)^-1 C sqrt(M)^-1.
    sqrt_mass = np.sqrt(M.diagonal())
    inverse_sqrt_mass = 1.0 / sqrt_mass
    scaling = sparse.diags(inverse_sqrt_mass)
    generator = diffusivity * (scaling @ C @ scaling)
    weighted_u0 = sqrt_mass * u0

    if times.ndim == 0:
        return expm_multiply(float(times) * generator, weighted_u0) * inverse_sqrt_mass
    if times.size >= 2 and times[-1] > times[0] and np.allclose(
        times, np.linspace(times[0], times[-1], times.size), rtol=1e-13, atol=0
    ):
        states = expm_multiply(
            generator, weighted_u0, start=times[0], stop=times[-1], num=times.size
        )
        return states * inverse_sqrt_mass

    return np.array([
        expm_multiply(t * generator, weighted_u0) * inverse_sqrt_mass for t in times
    ]).reshape(times.size, u0.size)


class ErrorNorms(NamedTuple):
    l2: np.ndarray
    linf: np.ndarray


def evaluate_errors(numerical, reference, vertex_areas):
    '''Return area-weighted L2 and maximum nodal errors for each sampled time.'''
    numerical = np.asarray(numerical)
    reference = np.asarray(reference)
    vertex_areas = np.asarray(vertex_areas)
    if numerical.shape != reference.shape or numerical.shape[-1:] != vertex_areas.shape:
        raise ValueError('Solutions must match in shape and have one value per vertex.')

    curr_err = numerical - reference

    # L2 error, weighted by 'area' of each vertex -- see comments for A matrix above for an expanded explanation
    l2_errs = np.sqrt((curr_err**2) @ vertex_areas)

    # L-inf error
    linf_errs = np.max(np.abs(curr_err), axis=-1)
    return ErrorNorms(
        l2=l2_errs,
        linf=linf_errs,
    )
