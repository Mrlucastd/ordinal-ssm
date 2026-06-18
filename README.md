# Ordinal SSM — Network-Scale Infrastructure Degradation Modeling

Code for the paper:

> **Network-Scale Infrastructure Degradation Modeling Using Ordinal Visual Inspections and State-Space Models**
> Lucas Alric, Zachary Hamida, James-A. Goulet — Polytechnique Montréal, 2026

---

## The Problem

Bridge management systems record the condition of structural elements using four ordered grades {**A**, **B**, **C**, **D**} — from excellent to poor — where each grade represents the fraction of the element surface exhibiting a given severity of deterioration. These fractions always sum to one. Over time, and in the absence of maintenance, the mass migrates monotonically from A toward D.

Existing approaches either collapse the four grades into a single scalar index (losing distributional information) or use discrete Markov chains calibrated at the network level (losing element-specific histories). Both discard structure that is physically meaningful and statistically exploitable.

---

## The Approach

This work reformulates the ordinal structure as a **hierarchy of three conditionally independent latent branches**, each evolving monotonically on [0, 1]:

| Branch | Physical meaning |
|---|---|
| x_AB | Fraction of surface in grades A or B (non-degraded group) |
| x_A\|AB | Of the AB surface, fraction that is A (excellent) |
| x_D\|CD | Of the CD surface, fraction that is D (severe) |

The four observable grade proportions {A, B, C, D} are recovered at every time step from these three branches:

```
x_A =       x_{A|AB}  ·  x_AB
x_B = (1 - x_{A|AB}) ·  x_AB
x_C = (1 - x_{D|CD}) · (1 - x_AB)
x_D =       x_{D|CD}  · (1 - x_AB)
```

Each branch is tracked by a **constrained kinematic Kalman filter and RTS smoother**. The observation equations are nonlinear (clipping to [0,1]) and all required moment integrals are derived analytically via piecewise Gaussian integration — no Monte Carlo, no numerical quadrature.

Inspector-specific observation noise is learned from data by **Analytical Gaussian Variational Inference (AGVI)**, separating genuine deterioration trends from subjective rating variability across inspectors.

---

## Key Properties

- **Monotonic by construction** — each branch can only move in one direction, matching the physical irreversibility of deterioration without maintenance.
- **Fully analytical** — filtering, smoothing, forecasting, and inspector learning all have closed-form updates. No sampling, no iterative solvers.
- **Scalable** — constant cost per element per time step; tested on networks of thousands of elements.
- **Well-calibrated** — predictive intervals achieve nominal coverage on both synthetic and real inspection data; standardized residuals are approximately N(0,1) in the deterioration range.

---

## Repository Structure

```
ordinal-ssm/
├── src/core/
│   ├── state_models.py          # SSMparam: kinematic state-space model
│   ├── kalman_filter.py         # Kalman filter + RTS smoother
│   ├── inspector_manager.py     # AGVI inspector variance learning
│   ├── utils.py                 # gaussian_product and helpers
│   ├── agvi.py                  # Analytical Gaussian Variational Inference
│   ├── close_form.py            # Closed-form moment integrals
│   ├── inference.py             # Inference routines
│   └── nonlinear_processor.py   # Nonlinear moment propagation
├── data/
│   └── synthetic_bridge_data.py # Synthetic network generator
├── validation_pipeline.py       # Grid search + full pipeline + printed summary
├── environment.yml
└── README.md
```

---

## Quickstart

```bash
conda env create -f environment.yml
conda activate pyopenipmd
python validation_pipeline.py
```

`validation_pipeline.py` performs a two-level grid search over kinematic hyperparameters (velocity prior, acceleration prior, process noise), selects the best configuration by weighted log-likelihood on the validation set, retrains on the training set, and prints a numerical summary on the held-out test set:

- Per-stage test log-likelihoods (regular and weighted)
- Learned inspector σ vs ground-truth MAE
- Mean grade proportions at last time step
- Forecast RMSE vs ground truth at Δt = 1, 2, 3 yr (+ naive last-observation baseline)
- Calibration: z-score standard deviation (should be ≈ 1.0)

Results are also written to `grid_search_results.csv` and `best_summary.txt` in the working directory.

---

## Synthetic Data

The generator (`data/synthetic_bridge_data.py`) simulates a network of bridge elements:

- 100 bridges × 50 elements = 5 000 time series
- 20 inspectors, each with a fixed but unknown observation noise σ ∈ [0.03, 0.12]
- 3–10 inspections per element with inter-observation gaps of 2–4 years
- Ground-truth trajectories stored alongside noisy observations for exact verification

---

## Requirements

See `environment.yml`. Core dependencies: JAX (64-bit), NumPy, pandas.

---

## Citation

```bibtex
@article{alric2026ordinalssm,
  title   = {Network-Scale Infrastructure Degradation Modeling Using
             Ordinal Visual Inspections and State-Space Models},
  author  = {Alric, Lucas and Hamida, Zachary and Goulet, James-A.},
  year    = {2026}
}
```
