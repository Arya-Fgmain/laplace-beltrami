'''Measure how initial temperature perturbations propagate through the heat solver.'''

from numbers import Integral
from typing import NamedTuple

import numpy as np

from lp import (
    ErrorNorms,
    compute_cotangent_laplacian,
    compute_eigenmodes,
    evaluate_errors,
    make_heat_solution,
)


DEFAULT_EPSILONS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


class PerturbationErrors(NamedTuple):
    random: ErrorNorms  # axes: trial, perturbation size, time
    low_frequency: ErrorNorms  # axes: perturbation size, time


class StabilityEvaluation(NamedTuple):
    times: np.ndarray
    epsilons: np.ndarray
    errors_by_modes: dict[int, PerturbationErrors]


def _unit_perturbation(eta, A):
    # Remove the constant part so u0 and u0 + epsilon * eta have the same total heat.
    eta = eta - (eta @ A / A.sum())[..., None]
    return eta / np.sqrt((eta**2) @ A)[..., None]


def evaluate_stability(
    mesh, u0, *, times, mode_counts, diffusivity=1.0,
    epsilons=DEFAULT_EPSILONS, trials=25, seed=0,
):
    '''Compare perturbed and unperturbed spectral solutions on a fixed mesh.'''
    times = np.atleast_1d(np.asarray(times, dtype=float))
    if times.ndim != 1 or not times.size or not np.all(np.isfinite(times) & (times >= 0)):
        raise ValueError('Supply at least one finite, nonnegative evaluation time.')
    epsilons = np.asarray(epsilons, dtype=float)
    if epsilons.ndim != 1 or not epsilons.size or not np.all(np.isfinite(epsilons) & (epsilons > 0)):
        raise ValueError('Supply at least one finite, positive perturbation size.')
    if not isinstance(trials, Integral) or trials < 1:
        raise ValueError('The number of trials must be a positive integer.')
    mode_counts = sorted(set(mode_counts))
    if not mode_counts or any(not isinstance(r, Integral) or r < 1 for r in mode_counts):
        raise ValueError('Supply at least one positive integer mode count.')
    times = np.unique(times)
    epsilons = np.unique(epsilons)
    u0 = np.asarray(u0, dtype=float)

    C, M = compute_cotangent_laplacian(mesh)
    A = M.diagonal()
    # Include the first nonconstant mode as a test direction even if R = 1.
    mu, Phi = compute_eigenmodes(C, M, n_modes=max(2, mode_counts[-1]))
    rng = np.random.default_rng(seed)
    random_directions = _unit_perturbation(rng.standard_normal((trials, A.size)), A)
    low_frequency = _unit_perturbation(Phi[:, 1], A)
    directions = np.vstack([random_directions, low_frequency])

    errors_by_modes = {}
    for r in mode_counts:
        u = make_heat_solution(mu[:r], Phi[:, :r], M, u0, diffusivity=diffusivity)
        baseline = u(times)
        l2_errs = np.empty((trials + 1, len(epsilons), len(times)))
        linf_errs = np.empty_like(l2_errs)
        for i, eta in enumerate(directions):
            for j, epsilon in enumerate(epsilons):
                perturbed = make_heat_solution(
                    mu[:r], Phi[:, :r], M, u0 + epsilon * eta, diffusivity=diffusivity
                )
                l2_errs[i, j], linf_errs[i, j] = evaluate_errors(perturbed(times), baseline, A)

        errors_by_modes[r] = PerturbationErrors(
            random=ErrorNorms(l2_errs[:-1], linf_errs[:-1]),
            low_frequency=ErrorNorms(l2_errs[-1], linf_errs[-1]),
        )
    return StabilityEvaluation(times, epsilons, errors_by_modes)


def print_stability(result):
    print(f"{'R':>5} {'epsilon':>11} {'t':>9} {'median L2':>12} {'median Linf':>12} "
          f"{'max L2 gain':>12} {'mode L2 gain':>13}")
    for r, errors in result.errors_by_modes.items():
        median_l2 = np.median(errors.random.l2, axis=0)
        median_linf = np.median(errors.random.linf, axis=0)
        max_l2 = errors.random.l2.max(axis=0)
        for j, epsilon in enumerate(result.epsilons):
            for i, t in enumerate(result.times):
                print(f'{r:5d} {epsilon:11.3g} {t:9.4g} {median_l2[j, i]:12.4e} '
                      f'{median_linf[j, i]:12.4e} {max_l2[j, i] / epsilon:12.4e} '
                      f'{errors.low_frequency.l2[j, i] / epsilon:13.4e}')


def plot_stability(result, output_dir):
    import matplotlib.pyplot as plt

    norms = [r'Area-weighted $L^2$', r'Nodal $L^\infty$']
    epsilon = result.epsilons[-1]
    for r, errors in result.errors_by_modes.items():
        mode_dir = output_dir / f'R_{r}'
        mode_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
        fig.suptitle(f'Solution change under random initial perturbations (R = {r})')
        for ax, norm, samples in zip(axes, norms, errors.random):
            lower, median, upper = np.percentile(samples, [5, 50, 95], axis=0)
            for i, t in enumerate(result.times):
                line, = ax.plot(result.epsilons, median[:, i], '.-', label=f't = {t:g}')
                ax.fill_between(result.epsilons, lower[:, i], upper[:, i],
                                color=line.get_color(), alpha=0.18)
            ax.set(xscale='log', xlabel=r'Perturbation size $\varepsilon$', ylabel=f'{norm} change')
            if np.any(upper > 0):
                ax.set_yscale('log', nonpositive='mask')
            ax.grid(True, alpha=0.3)
        axes[0].legend()
        fig.savefig(mode_dir / 'change_vs_perturbation.png', dpi=200)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
        fig.suptitle(rf'Amplification (R = {r}, $\varepsilon$ = {epsilon:g})')
        for ax, norm, samples, mode in zip(axes, norms, errors.random, errors.low_frequency):
            lower, median, upper = np.percentile(samples[:, -1] / epsilon, [5, 50, 95], axis=0)
            line, = ax.plot(result.times, median, '.-', label='Random median')
            ax.fill_between(result.times, lower, upper, color=line.get_color(), alpha=0.18,
                            label='5th–95th percentile')
            ax.plot(result.times, mode[-1] / epsilon, '.--', label='First nonconstant mode')
            ax.set(xlabel='Time', ylabel=rf'{norm} change / $\varepsilon$')
            ax.grid(True, alpha=0.3)
        axes[0].axhline(1, color='gray', linestyle=':', label=r'$L^2$ bound')
        axes[0].legend()
        fig.savefig(mode_dir / 'gain_vs_time.png', dpi=200)
        plt.close(fig)
