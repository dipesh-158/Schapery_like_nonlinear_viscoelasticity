# Linear Viscoelastic Model with g₂ Nonlinearity

A compact, identifiable viscoelastic model for fitting cyclic compressive stress–strain data on soft, fabric-like materials.

The model is a multi-mode Prony series (linear viscoelasticity) augmented with a **single** Schapery-style nonlinear function `g₂(σ)` that modifies the effective stress driving the internal variables. It captures the time-dependent creep, partial recovery, ratcheting, and peak-strain flattening visible in the experimental data, while keeping every parameter individually identifiable from a single loading protocol.

---

## Why this model and not a full Schapery model

A full Schapery formulation has four stress-dependent functions (`g₀`, `g₁`, `g₂`, `aσ`) plus a Prony spectrum, giving roughly 14 parameters when discretised. With a single loading amplitude in the experimental data, those four nonlinearities are not separately identifiable — many parameter combinations produce indistinguishable fits. After working through that issue we deliberately reduced the formulation to the minimum that the data can support:

- `g₁` (output scaling) is redundant with the compliance amplitudes `Dₖ` at one stress level — it can be absorbed into `Dₖ`, so it is removed.
- `aσ` (time-clock shift) is unidentifiable without multi-rate data — earlier fits returned `d_shift = 0`, confirming this.
- `g₂` (effective-stress modifier) changes the **shape** of the response, not just its amplitude. It is **not** redundant with `Dₖ`, and it is what produces the observed peak-strain flattening.

The result is a 13-parameter model (2 nonlinear + 11 linear) with a clean identifiability story.

---

## Model

The strain at time `t` is

```
ε(t) = ε₀ + D₀·σ(t) + Σₖ qₖ(t)
```

with one internal state per Prony mode evolving via the standard exact recursive update:

```
qₖ(t+Δt) = r·qₖ(t) + (1 − r)·Dₖ·σ_eff,        r = exp(−Δt / τₖ)
σ_eff    = g₂(σ_mid) · σ_mid
σ_mid    = ½·(σ(t) + σ(t+Δt))
```

The single nonlinear function is

```
g₂(σ) = 1 / (1 + c₂ · |σ| / σ_ref)
```

Properties of `g₂`:

| σ              | g₂(σ)              | meaning                           |
|----------------|--------------------|-----------------------------------|
| 0              | 1                  | linear-viscoelastic limit (exact) |
| σ_ref          | 1 / (1 + c₂)       | half-saturation                   |
| → ∞            | → 0                | full saturation (peak flattening) |

`σ_ref` controls **where** the transition happens; `c₂` controls **how strong** the saturation is.

---

## Parameters

| Symbol      | Type                  | How it is fitted | Count |
|-------------|-----------------------|------------------|------:|
| `c₂`, `σ_ref` | nonlinear (`g₂`)    | `least_squares` (outer loop) | 2 |
| `D₀`        | instantaneous compliance | NNLS (inner loop)         | 1 |
| `D₁ … D₉`   | Prony amplitudes       | NNLS (inner loop)            | 9 |
| `ε₀`        | strain offset           | NNLS (inner loop)            | 1 |
| **Total**   |                         |                              | **13** |

The retardation times `τₖ` are **fixed**, log-spaced from 0.15 s to 50 000 s:

```
TAU = [0.15, 1.0, 5.0, 20.0, 80.0, 300.0, 1500.0, 8000.0, 50000.0]   # seconds
```

This spectrum spans every timescale visible in a ~550 s test, plus long-term tails for ratcheting and recovery extrapolation. Fixing `τₖ` and fitting only the amplitudes is the standard practice for Prony fitting and is what makes the inner problem linear (and therefore exactly solvable by NNLS with a guaranteed global optimum).

---

## Fitting strategy — nested optimisation

The fit exploits the linear / nonlinear separation in the parameters:

```
outer loop:   least_squares  over  [c₂, σ_ref]                  (2 unknowns, log-space)
                |
                └── for each candidate (c₂, σ_ref):
                      build design matrix A with σ_eff = g₂(σ)·σ
                      inner loop: NNLS over [D₀, D₁..D₉, ε₀]    (11 unknowns, ≥ 0)
                      compute residual = A·x_lin − ε_data
```

This is the same nested approach used in the larger Schapery code, but the outer search space is reduced from 7 dimensions to 2, which makes the fit fast and robust. The inner NNLS step is convex and linear, so for any fixed `(c₂, σ_ref)` the compliance amplitudes are determined exactly.

The non-negativity constraint on `Dₖ` is a **thermodynamic** requirement (negative compliances are non-physical and would correspond to systems that absorb energy on a closed loading cycle), not just a numerical convenience.

---

## Repository layout

```
.
├── linear_ve_g2.py        # main script: fit + plots + predictions
├── README.md              # this file
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT
└── .gitignore
```

---

## Installation

Clone the repository and install the dependencies into a virtual environment.

```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The dependencies are `numpy`, `pandas`, `scipy`, and `matplotlib`. Any reasonably recent version (last 3–4 years) works.

---

## Input data format

The script reads a single CSV file with three columns: time (s), stress (Pa), and strain (dimensionless). Column names are configured at the top of the script:

```python
CSV_PATH   = r"path/to/your/data.csv"
TIME_COL   = "Time"
STRESS_COL = "Stress"
STRAIN_COL = "Strain"
```

The loader tolerates a column named `"Strain "` with a trailing space, which is common in exports from certain test machines — adjust `STRAIN_COL` to match your file exactly.

The data does **not** need to be on a uniform time grid. The script sorts by time, interpolates onto a uniform grid at `DT_UNIFORM = 0.25 s`, and zero-baselines both time and strain. This baselining is what allows `ε₀` to be small but non-zero (it absorbs measurement offset, not real material deformation).

---

## Running

```bash
python linear_ve_g2.py
```

Console output:

```
Loaded N points, t=[0.0, T] s
Stress range: [σ_min, σ_max] Pa
Strain range: [ε_min, ε_max]
Fitting: 2 nonlinear (c2, sigma_ref) + 11 linear (NNLS)...
=========================================================
FIT RESULTS
=========================================================

NONLINEAR parameters (2):
  c2        = ...
  sigma_ref = ... Pa

LINEAR parameters (11):
  D0 = ...
  D1 = ...
  ...
  eps0 = ...

RMSE = ...
Forward simulation consistency check: max |error| = ~1e-15
```

Four PNG plots are written to the working directory:

| File                         | Content                                                            |
|------------------------------|--------------------------------------------------------------------|
| `plot_fit_g2.png`            | Stress input and fitted strain overlaid on experiment              |
| `plot_g2_function.png`       | `g₂(σ)` curve and `σ_eff` vs applied stress                       |
| `plot_predictions_g2.png`    | Frozen-parameter predictions on three new stress histories         |
| `plot_recovery_g2.png`       | Long-time recovery extrapolation (24 h after the test ends)        |

---

## What the fitted parameters mean

After the fit, **all 13 parameters are constants**. They go into `forward_simulate()` unchanged for every prediction. `g₂` is *not* a parameter — it is a function whose internal constants `c₂` and `σ_ref` are frozen, but whose output value changes moment to moment as `σ(t)` changes during the simulation.

A useful sanity check, printed at the end of the fit:

```
g₂ at peak stress (σ_max):     g₂ = ...
σ_eff at peak stress:           σ_eff = ... Pa
g₂ at zero stress:              g₂ = 1.0000   ← by construction
```

If `σ_eff` at peak is dramatically smaller than `σ_max` (say, less than 5 %), the model is in a strongly saturating regime and predictions far above the calibration stress will be unreliable. This is a property of the data, not a bug.

---

## What this model can and cannot do

**Can**

- Fit cyclic stress–strain data with high accuracy on a single peak amplitude.
- Predict strain for any new stress history as long as the peak stress stays close to the calibration range.
- Reproduce ratcheting and partial recovery through the slow Prony modes (`τ` from 1500 s up to 50 000 s).
- Extrapolate recovery far beyond the measurement window (24 h shown in `plot_recovery_g2.png`).

**Cannot (yet)**

- Be quantitatively trusted at stress levels far from the calibration range. With only one peak amplitude in the training data, the values of `c₂` and `σ_ref` describe the *shape* at that amplitude but cannot be cross-validated at other amplitudes.
- Capture loading-rate effects independent of stress amplitude. There is no `aσ`-style time-clock shift in this model.
- Generalise to strain-controlled experiments without modification — the formulation is in compliance form (stress in, strain out).

To lift these limitations, multi-amplitude creep tests and multi-rate ramp tests are needed; with that data, the `g₂` parameters become individually identifiable and additional Schapery functions can be re-introduced one at a time.

---

## Limitations and identifiability caveat

This section is deliberately explicit so that anyone reading the code understands what has and has not been validated.

**The current dataset cannot distinguish this model from a purely linear Prony series.**

The experimental data on which this model has been calibrated consists of triangular ramp / unload / hold cycles at a **single peak stress amplitude**, with the experiment replicated three times. A multi-mode linear Prony series with the same retardation spectrum can already reproduce the visible features of that data — instantaneous response, creep during hold, partial recovery, cycle-to-cycle ratcheting — through the interaction of multiple linear timescales with a triangular loading profile. The "ratcheting" in particular is **not** evidence of nonlinearity; it is exactly what slow Prony modes (`τ ≫` test duration) produce when the test ends before they have finished recovering.

What this means in practice:

- The `g₂` nonlinearity in this code is **structurally** justified (it is the Schapery component most clearly identifiable from a single loading protocol, and it changes the shape of the response in a way `Dₖ` cannot absorb), but it is **empirically** unconfirmed against this dataset alone. The fitted values `c₂` and `σ_ref` describe the curve shape at one peak stress; they are not a tested prediction about how the material behaves at other stress levels.
- The improvement in fit quality (RMSE) over a linear-only Prony series on this dataset has not been formally quantified yet. Expect the gap to be modest. If a future contributor runs the comparison, the result should be added here.
- Because `σ_ref` and the calibration peak stress live on very different scales (earlier fits returned `σ_ref ≈ 3 kPa` against a peak of `~80 kPa`), `g₂` saturates well before the peak. This is consistent with the data the optimiser saw, but it also means the model says almost all of the response is governed by behaviour in the 0–`σ_ref` effective-stress range. Predictions at peak stresses far above the calibration range should be treated as extrapolations, not predictions.

**What would resolve this.**

The minimum experimental campaign that can either confirm or falsify the `g₂` nonlinearity is:

1. Creep tests at three or more stress levels (e.g. 20 %, 50 %, 80 % of the calibration peak). This is the single most informative experiment: it directly tests whether `g₂(σ)` predicts the right shape changes with amplitude.
2. Ramp tests at two or more loading rates. Required if the `aσ` time-clock shift is ever to be re-introduced.
3. A long pure-creep test at constant `σ` for a duration much longer than 1000 s. Validates the slow Prony tail amplitudes (`Dₖ` at `τₖ` ≥ 1500 s), which currently extrapolate beyond the measurement window.
4. A recovery test with strain monitored after unloading, ideally for several hours.

Until at least the multi-amplitude creep test is available, the recommended scientific framing of this model is: *"a Schapery-inspired single-nonlinearity formulation calibrated on cyclic data, structurally consistent with the existing measurements, awaiting multi-amplitude validation."* It is not yet *"a validated nonlinear viscoelastic constitutive model."*

---

## Reproducing predictions on new stress inputs

Reuse the frozen parameters with `forward_simulate`:

```python
from linear_ve_g2 import forward_simulate, TAU
import numpy as np

# After running the script once and recording the printed parameters:
D0       = ...   # from the fit output
Dk       = np.array([..., ..., ...])   # 9 values, in TAU order
eps0     = ...
c2       = ...
sigma_ref = ...

# Define any stress history
t     = np.arange(0, 1000, 0.25)
sigma = np.where(t < 500, 50_000.0, 0.0)   # 50 kPa step, then unload

eps_pred = forward_simulate(t, sigma, D0, Dk, eps0, c2, sigma_ref)
```

The code path executed by `forward_simulate()` is bit-identical to the one used during fitting (verified at the end of `main` by `Forward simulation consistency check: max |error| = ~1e-15`).

---

## Citation and references

The model structure draws directly on the standard viscoelastic literature. The most relevant references are:

1. Schapery, R. A. (1969). *On the characterization of nonlinear viscoelastic materials*. Polymer Engineering and Science, 9(4), 295–310.
2. Chen, T. (2000). *Determining a Prony Series for a Viscoelastic Material From Time Varying Strain Data*. NASA/TM-2000-210123. (Free on NASA NTRS.)
3. Lai, J. and Bakker, A. (1996). *3-D Schapery representation for non-linear viscoelasticity and finite element implementation*. Computational Mechanics, 18, 182–191.
4. Park, S. W. and Schapery, R. A. (1999). *Methods of interconversion between linear viscoelastic material functions*. International Journal of Solids and Structures, 36(11), 1653–1675.
5. Ferry, J. D. (1980). *Viscoelastic Properties of Polymers*, 3rd ed., Wiley. Chapters 3–4.

---

## License

Released under the MIT License — see [LICENSE](LICENSE).
