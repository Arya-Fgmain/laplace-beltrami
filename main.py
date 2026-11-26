'''Run spectral heat-equation accuracy experiments on unit-sphere meshes.'''

import argparse
from numbers import Integral
from pathlib import Path
from typing import NamedTuple

import numpy as np
import trimesh
from scipy.special import eval_legendre

from lp import (
    ErrorNorms,
    compute_cotangent_laplacian,
    compute_eigenmodes,
    evaluate_errors,
    make_heat_solution,
    preprocess_mesh,
    solve_discrete_heat,
)
from stability import DEFAULT_EPSILONS, evaluate_stability, plot_stability, print_stability


class ModeErrors(NamedTuple):
    truncation: ErrorNorms
    total: ErrorNorms | None


class MeshEvaluation(NamedTuple):
    vertex_count: int
    max_edge_length: float
    times: np.ndarray
    discretization: ErrorNorms | None
    errors_by_modes: dict[int, ModeErrors]


### Accuracy Evaluation

def evaluate_mesh(mesh, u0, *, times, mode_counts, diffusivity=1.0, exact_solution=None):
    '''Evaluate a cleaned mesh with vertex-aligned u0 and an optional exact(t).'''
    times = np.atleast_1d(np.asarray(times, dtype=float))
    if times.ndim != 1 or times.size == 0:
        raise ValueError('Supply at least one evaluation time in a 1D array.')
    mode_counts = sorted(set(mode_counts))
    if not mode_counts or any(not isinstance(n, Integral) or n < 1 for n in mode_counts):
        raise ValueError('Supply at least one positive integer mode count.')

    C, M = compute_cotangent_laplacian(mesh)
    mu, Phi = compute_eigenmodes(C, M, n_modes=mode_counts[-1])
    A = M.diagonal()
    discrete = solve_discrete_heat(C, M, u0, times, diffusivity=diffusivity)
    exact = None if exact_solution is None else np.stack([exact_solution(t) for t in times])
    discretization = None if exact is None else evaluate_errors(discrete, exact, A)

    errors_by_modes = {}
    for r in mode_counts:
        u = make_heat_solution(
            mu[:r], Phi[:, :r], M, u0, diffusivity=diffusivity
        )
        approximate = u(times)
        errors_by_modes[r] = ModeErrors(
            truncation=evaluate_errors(approximate, discrete, A),
            total=None if exact is None else evaluate_errors(approximate, exact, A),
        )

    return MeshEvaluation(
        vertex_count=len(mesh.vertices),
        max_edge_length=float(mesh.edges_unique_length.max()),
        times=times,
        discretization=discretization,
        errors_by_modes=errors_by_modes,
    )


def normalize_unit_sphere(mesh):
    '''Normalize a cleaned spherical mesh in place for the analytic benchmark.'''
    # center the mesh
    centered = mesh.vertices - mesh.bounding_box.centroid

    # get original mesh scale, then normalize it to a unit sphere
    radii = np.linalg.norm(centered, axis=1)
    radius = radii.mean()

    # check that the mesh is indeed a unit sphere after rescaling
    if radius <= 0 or not np.allclose(radii / radius, 1, rtol=0, atol=1e-3):
        raise ValueError('The sphere benchmark requires a nearly spherical mesh.')
    mesh.vertices = centered / radii[:, None]


def sphere_problem(vertices, *, diffusivity, case):
    '''Return initial values and exact(t) for a unit-sphere benchmark.'''
    # sample initial condition
    # The Gaussian draft takes V = vertices and has no exact solution here.
    # center = V.mean(axis=0) # center of our sphere
    # R = np.mean(np.linalg.norm(V-center, axis=1))   # radius
    # north = V[np.argmax(V[:,2])]
    # sigma = 2.4 * R
    # complicated IC
    # u0 = np.exp(-np.sum((V - north)**2, axis=1) / (2 * sigma**2))

    # eig-func IC 1
    # u0 = 1 + V[:, 2]

    # eig-func IC 2
    # u0 = 1 + V[:, 0] * V[:, 1]
    # def u_exact(t):
    #     return 1 + V[:, 0] * V[:, 1] * np.exp(-6*diffusivity*t)

    # eig-func IC 3
    # u0 = 1 + V[:, 0] * V[:, 1] * V[:, 2]
    # def u_exact(t):
    #     return 1 + V[:, 0] * V[:, 1] * V[:, 2] * np.exp(-12*diffusivity*t)
    degrees_and_weights = {
        'baseline': [(1, 1.0)],
        'multimode': [(1, 0.5), (4, 0.3), (8, 0.2)],
    }
    degrees, weights = zip(*degrees_and_weights[case])
    degrees = np.asarray(degrees)
    components = np.asarray(weights)[:, None] * eval_legendre(
        degrees[:, None], vertices[:, 2]
    )
    decay_rates = diffusivity * degrees * (degrees + 1)

    def exact(t):
        return 1.0 + np.exp(-t * decay_rates) @ components

    return exact(0.0), exact


def print_evaluation(result):
    print(f"\n{result.vertex_count} vertices, h = {result.max_edge_length:.6g}")
    print(
        f"{'R':>5} {'t':>9} {'trunc L2':>12} {'mesh L2':>12} {'total L2':>12} "
        f"{'trunc Linf':>12} {'mesh Linf':>12} {'total Linf':>12}"
    )
    for n_modes, errors in result.errors_by_modes.items():
        for i, t in enumerate(result.times):
            print(
                f"{n_modes:5d} {t:9.4g} {errors.truncation.l2[i]:12.4e} "
                f"{result.discretization.l2[i]:12.4e} {errors.total.l2[i]:12.4e} "
                f"{errors.truncation.linf[i]:12.4e} {result.discretization.linf[i]:12.4e} "
                f"{errors.total.linf[i]:12.4e}"
            )


def plot_evaluation(result, output_dir):
    # plotting (thank you AI)
    import matplotlib.pyplot as plt

    mode_counts = sorted(result.errors_by_modes)
    errors = [result.errors_by_modes[r].total for r in mode_counts]
    if any(error is None for error in errors):
        raise ValueError('Plotting error against the exact solution requires exact_solution.')

    time_order = np.argsort(result.times)
    times = result.times[time_order]
    l2_errs = np.stack([error.l2[time_order] for error in errors])
    linf_errs = np.stack([error.linf[time_order] for error in errors])

    plots = [
        ('error_vs_time.png', times, 'Time', [f'R = {r}' for r in mode_counts],
         l2_errs, linf_errs),
        ('error_vs_modes.png', mode_counts, 'Retained eigenmodes (R)',
         [f't = {t:g}' for t in times], l2_errs.T, linf_errs.T),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    norms = [r'Area-weighted $L^2$ error', r'Nodal $L^\infty$ error']
    for filename, x, xlabel, labels, l2, linf in plots:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
        fig.suptitle(f'Error vs. exact solution (N = {result.vertex_count}, '
                     f'h = {result.max_edge_length:.4g})')
        for ax, ylabel, curves in zip(axes, norms, [l2, linf]):
            for label, curve in zip(labels, curves):
                ax.plot(x, curve, marker='.', label=label)
            ax.set(xlabel=xlabel, ylabel=ylabel)
            ax.grid(True, alpha=0.3)

        if filename == 'error_vs_modes.png':
            for ax in axes:
                ax.set_xticks(mode_counts)
        handles, _ = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncols=min(5, len(labels)))
        try:
            fig.savefig(output_dir / filename, dpi=200)
        finally:
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--mesh', type=Path, help='Spherical mesh file (default: sphere.stl)')
    source.add_argument('--subdivisions', type=int, nargs='+', help='Icosphere refinement levels')
    parser.add_argument('--modes', type=int, nargs='+', default=[4, 9, 16, 25], help='Mode counts R')
    parser.add_argument('--case', choices=['baseline', 'multimode'], default='baseline')
    # diffusivity coefficient
    parser.add_argument('--diffusivity', type=float, default=1.0)
    parser.add_argument('--times', type=float, nargs='+', default=[0.0, 0.01, 0.1, 1.0])
    parser.add_argument('--plot', action='store_true', help='Save error plots for each mesh resolution')
    stability = parser.add_argument_group('initial condition stability')
    stability.add_argument('--stability', action='store_true', help='Run the initial perturbation experiment')
    stability.add_argument('--epsilons', type=float, nargs='+', default=DEFAULT_EPSILONS)
    stability.add_argument('--trials', type=int, default=25, help='Number of random perturbation directions')
    stability.add_argument('--seed', type=int, default=0, help='Random seed for the stability experiment')
    args = parser.parse_args()

    if args.subdivisions is not None:
        if min(args.subdivisions) < 0:
            parser.error('Icosphere subdivisions must be nonnegative.')
        meshes = (
            (f'subdivision_{n}', trimesh.creation.icosphere(subdivisions=n))
            for n in args.subdivisions
        )
    else:
        path = args.mesh or Path(__file__).with_name('sphere.stl')
        meshes = [(path.stem, trimesh.load_mesh(path))]

    for mesh_name, mesh in meshes:
        mesh = preprocess_mesh(mesh)
        if max(args.modes) > len(mesh.vertices):
            parser.error(f"Mode counts cannot exceed the mesh's {len(mesh.vertices)} vertices.")
        normalize_unit_sphere(mesh)
        u0, exact = sphere_problem(mesh.vertices, diffusivity=args.diffusivity, case=args.case)
        output_dir = Path('plots') / args.case / f'{mesh_name}_N_{len(mesh.vertices)}'
        if args.stability:
            print(f'\n{len(mesh.vertices)} vertices, h = {mesh.edges_unique_length.max():.6g}')
            result = evaluate_stability(
                mesh, u0, times=args.times, mode_counts=args.modes, diffusivity=args.diffusivity,
                epsilons=args.epsilons, trials=args.trials, seed=args.seed,
            )
            print_stability(result)
            if args.plot:
                plot_stability(result, output_dir / 'stability')
                print(f'Saved plots in {output_dir / "stability"}')
            continue

        result = evaluate_mesh(
            mesh, u0, times=args.times, mode_counts=args.modes,
            diffusivity=args.diffusivity, exact_solution=exact,
        )
        print_evaluation(result)
        if args.plot:
            plot_evaluation(result, output_dir)
            print(f'Saved plots in {output_dir}')


'''
Visualization: AI did it
'''
# This draft takes V, F, u(t), sphere radius R, and diffusivity k.
# import plotly.graph_objects as go

# Choose times relative to sphere radius and diffusivity
# times = np.linspace(0, 0.5 * R**2 / k, 40)
# solutions = [u(t) for t in times]

# Initial 3D mesh
# fig = go.Figure(
#     data=[go.Mesh3d(
#         x=V[:, 0], y=V[:, 1], z=V[:, 2],
#         i=F[:, 0], j=F[:, 1], k=F[:, 2],
#         intensity=solutions[0],
#         intensitymode='vertex',
#         colorscale='Inferno',
#         cmin=0, cmax=1,
#         colorbar=dict(title='Temperature')
#     )]
# )

# Animation frames
# fig.frames = [
#     go.Frame(
#         data=[go.Mesh3d(intensity=values)],
#         traces=[0],
#         name=str(q)
#     )
#     for q, values in enumerate(solutions)
# ]

# Time slider
# fig.update_layout(
#     title='Spectral Heat Diffusion on a Sphere',
#     scene=dict(aspectmode='data'),
#     sliders=[dict(
#         currentvalue=dict(prefix='Time: '),
#         steps=[
#             dict(
#                 label=f'{t:.3f}',
#                 method='animate',
#                 args=[
#                     [str(q)],
#                     dict(mode='immediate',
#                          frame=dict(duration=0, redraw=True),
#                          transition=dict(duration=0))
#                 ]
#             )
#             for q, t in enumerate(times)
#         ]
#     )],
#     updatemenus=[dict(
#         type='buttons',
#         buttons=[
#             dict(
#                 label='Play',
#                 method='animate',
#                 args=[None, dict(
#                     frame=dict(duration=100, redraw=True),
#                     fromcurrent=True,
#                     transition=dict(duration=0)
#                 )]
#             ),
#             dict(
#                 label='Pause',
#                 method='animate',
#                 args=[[None], dict(
#                     mode='immediate',
#                     frame=dict(duration=0, redraw=False)
#                 )]
#             )
#         ]
#     )]
# )

# fig.write_html('sphere_heat_spectral.html')
# fig.show()


if __name__ == '__main__':
    main()
