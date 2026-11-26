# laplace-beltrami
MACM 416 / MATH 716 Project ft. Arya Faghihy, Martin Su, Artem Bakulin

## Conda environment

With Conda installed, run these commands from the repository root to create
the `laplace` environment from `environment.yml` and activate it.

```sh
conda env create -f environment.yml
conda activate laplace
```

## Heat equation experiments

Run the baseline \(1+z\) initial condition on the supplied sphere, or sweep
icosphere refinement levels and numbers of retained eigenfunctions \(R\).

```sh
python main.py --modes 4 9 16 --times 0 0.01 0.1 1
python main.py --subdivisions 1 2 3 4 --modes 4 9 25 36 --case multimode --plot
```

Use `--diffusivity` to set the diffusion coefficient. Mode counts must fit the
coarsest mesh in the sweep.
To include `R=81`, start the icosphere sweep at subdivision level 2. Each mesh
reports its vertex count and maximum edge length. The table gives area-weighted
L2 and maximum nodal errors for three comparisons.

- **Truncation** compares the retained modes with the full discrete heat solution.
- **Mesh discretization** compares the full discrete heat solution with the analytic solution.
- **Total** compares the retained modes with the analytic solution.

With `--plot`, each mesh has its own folder under `plots/<case>/`.
For example, `plots/multimode/subdivision_2_N_162/` contains two PNGs, each
with area-weighted L2 and nodal Linf error against the exact solution.

- `error_vs_time.png` has one curve per retained mode count.
- `error_vs_modes.png` has one curve per evaluation time.

Rerunning the same case and mesh replaces its two figures.

For a custom problem, clean the mesh with `lp.preprocess_mesh`, then call
`main.evaluate_mesh` with one initial value per cleaned vertex, the evaluation
times, the mode counts, and optionally an `exact_solution(t)` that returns
values on those same vertices. Use matching diffusivity in both solutions.

## Initial condition stability

Use `--stability` to compare perturbed and unperturbed numerical solutions on
the same mesh and eigenbasis. The experiment uses 25 seeded random directions
by default. Each direction has zero area-weighted mean and unit area-weighted
L2 norm, so `epsilon` is the L2 change in the initial data and total heat is
preserved. The same directions are reused at every perturbation size and mode
count. The first nonconstant eigenmode is also tested as a low-frequency check.

```sh
python main.py --subdivisions 1 2 --modes 4 9 25 --case multimode \
  --stability --epsilons 1e-6 1e-4 1e-2 1e-1 --times 0 0.01 0.1 1 --plot
```

Use `--trials` and `--seed` to change the number of random draws and reproduce
a run. With `--plot`, each mesh folder contains `stability/R_<count>/` with two
figures.

- `change_vs_perturbation.png` shows L2 and Linf solution change against
  `epsilon` at the chosen times, with a median and 5th–95th percentile band.
- `gain_vs_time.png` shows solution change divided by the largest tested
  `epsilon`, with the low-frequency check and the L2 gain bound of 1.

Both gains use the input's L2 norm as the denominator. The bound applies to
L2 gain.
