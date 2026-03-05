"""
================================================================================
LINEAR VISCOELASTIC + g2 NONLINEARITY MODEL
================================================================================

THEORY
------
We start from the linear Prony series that failed to capture peak flattening,
and add ONE nonlinear function: g2(sigma).

The model:
  eps(t) = eps0 + D0 * sigma(t) + sum_k q_k(t)

where each internal variable evolves as:
  q_k(t+dt) = r * q_k(t) + (1-r) * Dk * sigma_eff(t)
  r = exp(-dt / tau_k)

The ONLY nonlinearity is in sigma_eff:
  sigma_eff = g2(sigma) * sigma
  g2(sigma) = 1 / (1 + c2 * |sigma| / sigma_ref)

What g2 does physically:
  - At low stress: g2 ≈ 1, so sigma_eff ≈ sigma (linear behavior)
  - At high stress: g2 < 1, so sigma_eff < sigma (saturating/stiffening)
  - This creates the flattening near peak strain that your data shows

Why not g1 or a(sigma)?
  - g1 scales the OUTPUT of internal variables. With one stress level,
    g1 and Dk are redundant — you can absorb g1 into Dk.
  - a(sigma) shifts the timescale. Your fitted d_shift was 0.000 — the
    optimizer found no use for it.
  - g2 modifies the INPUT (effective stress). This is NOT redundant with
    Dk because it changes the SHAPE of the response, not just the amplitude.

Parameters:
  LINEAR (fitted by NNLS):  D0, D1..D9, eps0  = 11 parameters
  NONLINEAR (fitted by least_squares): c2, sigma_ref = 2 parameters
  TOTAL: 13 parameters

Fitting strategy:
  Outer loop: optimize c2 and sigma_ref (2-parameter nonlinear search)
  Inner loop: given c2 and sigma_ref, solve Dk by NNLS (linear, exact)
  This is the same nested approach as the Schapery code, but with only
  2 nonlinear unknowns instead of 7.

================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, nnls

# ============================================================
# CONFIG
# ============================================================
CSV_PATH   = r"C:/Users/random/folder/csv.csv"
TIME_COL   = "time"
STRESS_COL = "Stress"
STRAIN_COL = "Strain"

DT_UNIFORM = 0.25  # s, resample timestep

# Retardation spectrum (fixed, log-spaced)
TAU = np.array([0.15, 1.0, 5.0, 20.0, 80.0, 300.0, 1500.0, 8000.0, 50000.0])

# ============================================================
# Data helpers
# ============================================================
def load_and_prepare(path, time_col, stress_col, strain_col, dt):
    """Load CSV, sort, resample uniformly, zero-baseline."""
    df = pd.read_csv(path)

    missing = [c for c in [time_col, stress_col, strain_col] if c not in df.columns]
    if missing:
        print("CSV columns found:", list(df.columns))
        raise KeyError(f"Missing columns: {missing}")

    t = df[time_col].to_numpy(float)
    sigma = df[stress_col].to_numpy(float)
    eps = df[strain_col].to_numpy(float)

    # Sort by time
    order = np.argsort(t)
    t, sigma, eps = t[order], sigma[order], eps[order]

    # Resample onto uniform grid
    t_u = np.arange(t[0], t[-1] + dt, dt)
    sigma_u = np.interp(t_u, t, sigma)
    eps_u = np.interp(t_u, t, eps)

    # Zero baseline
    t_u = t_u - t_u[0]
    eps_u = eps_u - eps_u[0]

    return t_u, sigma_u, eps_u


# ============================================================
# g2 function — the ONLY nonlinearity
# ============================================================
def g2(sigma_val, c2, sigma_ref):
    """
    g2(sigma) = 1 / (1 + c2 * |sigma| / sigma_ref)

    Properties:
      g2(0) = 1           (linear at zero stress)
      g2(inf) -> 0         (saturates at high stress)
      sigma_ref controls WHERE the transition happens
      c2 controls HOW MUCH saturation
    """
    x = np.abs(sigma_val) / max(sigma_ref, 1e-12)
    return 1.0 / (1.0 + c2 * x)


# ============================================================
# Build design matrix with g2-modified effective stress
# ============================================================
def build_design_matrix(t, sigma, c2, sigma_ref):
    """
    Build matrix A such that:
        eps(t) ≈ A @ [D0, D1, ..., DN, eps0]

    Column 0:       D0 * sigma(t)              (instantaneous, uses raw sigma)
    Column k=1..N:  q_k(t) with Dk=1           (modes driven by sigma_eff)
    Column N+1:     eps0 * 1                    (offset)

    The key difference from pure linear: modes use sigma_eff = g2(sigma)*sigma
    instead of raw sigma. This makes the basis functions nonlinear in sigma,
    but the problem is still LINEAR in the compliance amplitudes Dk.
    """
    n = len(t)
    n_modes = len(TAU)
    A = np.zeros((n, n_modes + 2))

    # Column 0: instantaneous response (raw sigma — no g2 here)
    # D0*sigma is the elastic part, it should respond linearly
    A[:, 0] = sigma

    # Columns 1..N: mode basis functions driven by sigma_eff
    for k in range(n_modes):
        z = 0.0
        for i in range(1, n):
            dt = t[i] - t[i - 1]
            if dt <= 0:
                dt = 1e-12

            sigma_mid = 0.5 * (sigma[i] + sigma[i - 1])
            sigma_eff = g2(sigma_mid, c2, sigma_ref) * sigma_mid

            r = np.exp(-dt / TAU[k])
            z = r * z + (1.0 - r) * sigma_eff
            A[i, k + 1] = z

    # Last column: constant offset
    A[:, -1] = 1.0

    return A


# ============================================================
# Forward simulation (for prediction after fitting)
# ============================================================
def forward_simulate(t, sigma, D0, Dk, eps0, c2, sigma_ref):
    """
    Forward simulation with frozen parameters.
    Use this for PREDICTION on any new stress input.

    All parameters are constants — only sigma(t) changes between
    fitting and prediction.
    """
    n = len(t)
    n_modes = len(Dk)
    q = np.zeros(n_modes)

    eps = np.zeros(n)
    eps[0] = eps0 + D0 * sigma[0]

    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if dt <= 0:
            dt = 1e-12

        sigma_mid = 0.5 * (sigma[i] + sigma[i - 1])
        sigma_eff = g2(sigma_mid, c2, sigma_ref) * sigma_mid

        for k in range(n_modes):
            r = np.exp(-dt / TAU[k])
            q[k] = r * q[k] + (1.0 - r) * Dk[k] * sigma_eff

        eps[i] = eps0 + D0 * sigma[i] + np.sum(q)

    return eps


# ============================================================
# Fitting: 2 nonlinear params (outer) + NNLS for Dk (inner)
# ============================================================
def fit_model(t, sigma, eps_data):
    """
    Nested optimization:
      Outer: least_squares over [c2, sigma_ref]  (2 unknowns)
      Inner: NNLS over [D0, D1..D9, eps0]        (11 unknowns)
    """

    def residuals(x_nl):
        # Unpack nonlinear params
        c2_val = np.exp(x_nl[0])        # enforce c2 > 0
        sref_val = np.exp(x_nl[1])      # enforce sigma_ref > 0

        # Build design matrix with current g2 parameters
        A = build_design_matrix(t, sigma, c2_val, sref_val)

        # Solve for compliances (linear, exact, global optimum)
        x_lin, _ = nnls(A, eps_data)

        # Predicted strain
        eps_pred = A @ x_lin

        return eps_pred - eps_data

    # Initial guess for nonlinear params (log-space)
    # c2 ~ 2 and sigma_ref ~ 3000 Pa (from your Schapery results)
    x0 = np.array([np.log(2.0), np.log(3000.0)])

    # Bounds: c2 in [0.01, 1000], sigma_ref in [10, 1e8]
    lb = np.array([np.log(0.01), np.log(10.0)])
    ub = np.array([np.log(1000.0), np.log(1e8)])

    print("Fitting: 2 nonlinear (c2, sigma_ref) + 11 linear (NNLS)...")
    sol = least_squares(
        residuals, x0, bounds=(lb, ub),
        ftol=1e-12, xtol=1e-12, gtol=1e-12,
        max_nfev=500,
        verbose=1
    )

    # Extract final parameters
    c2_fit = np.exp(sol.x[0])
    sref_fit = np.exp(sol.x[1])

    A = build_design_matrix(t, sigma, c2_fit, sref_fit)
    x_lin, _ = nnls(A, eps_data)

    D0 = x_lin[0]
    Dk = x_lin[1:-1]
    eps0 = x_lin[-1]

    eps_pred = A @ x_lin
    rmse = np.sqrt(np.mean((eps_pred - eps_data) ** 2))

    return D0, Dk, eps0, c2_fit, sref_fit, eps_pred, rmse, sol


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # ---- Load data ----
    t, sigma, eps_data = load_and_prepare(
        CSV_PATH, TIME_COL, STRESS_COL, STRAIN_COL, DT_UNIFORM
    )

    print(f"Loaded {len(t)} points, t=[{t[0]:.1f}, {t[-1]:.1f}] s")
    print(f"Stress range: [{sigma.min():.0f}, {sigma.max():.0f}] Pa")
    print(f"Strain range: [{eps_data.min():.4f}, {eps_data.max():.4f}]")

    # ---- Fit ----
    D0, Dk, eps0, c2_fit, sref_fit, eps_pred, rmse, sol = fit_model(
        t, sigma, eps_data
    )

    # ---- Print results ----
    print(f"\n{'='*60}")
    print("FIT RESULTS")
    print(f"{'='*60}")
    print(f"\nNONLINEAR parameters (2):")
    print(f"  c2        = {c2_fit:.6f}")
    print(f"  sigma_ref = {sref_fit:.2f} Pa")
    print(f"\n  g2 at peak stress ({sigma.max():.0f} Pa):")
    print(f"    g2 = {g2(sigma.max(), c2_fit, sref_fit):.4f}")
    print(f"    sigma_eff = {g2(sigma.max(), c2_fit, sref_fit) * sigma.max():.0f} Pa")
    print(f"  g2 at zero stress: g2 = 1.0000 (by construction)")

    print(f"\nLINEAR parameters (11):")
    print(f"  D0 = {D0:.6e} [1/Pa]")
    for k, (tau_k, dk) in enumerate(zip(TAU, Dk)):
        active = "  <-- active" if dk > 1e-15 else ""
        print(f"  D{k+1} = {dk:.6e} [1/Pa]   (tau={tau_k:>8.1f} s){active}")
    print(f"  eps0 = {eps0:.6e}")

    print(f"\nRMSE = {rmse:.6e}")
    print(f"Solver: {sol.message}")
    print(f"Total parameters: 13  (2 nonlinear + 11 linear)")

    # ---- Verify: forward simulation matches ----
    eps_check = forward_simulate(t, sigma, D0, Dk, eps0, c2_fit, sref_fit)
    check_err = np.max(np.abs(eps_check - eps_pred))
    print(f"\nForward simulation consistency check: max |error| = {check_err:.2e}")

    # ---- Plot 1: Fit quality ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(t, sigma / 1000, "b-", lw=2)
    axes[0].set_ylabel("Stress [kPa]")
    axes[0].set_title("Input Stress (from CSV)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, eps_data, "k.", ms=1, alpha=0.5, label="Experimental (CSV)")
    axes[1].plot(t, eps_pred, "r-", lw=2, label=f"Linear VE + g2 (RMSE={rmse:.4e})")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Strain [-]")
    axes[1].set_title("Fit: Linear VE + g2 Nonlinearity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plot_fit_g2.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ---- Plot 2: g2 function over stress range ----
    s_range = np.linspace(0, sigma.max() * 1.5, 500)
    g2_vals = np.array([g2(s, c2_fit, sref_fit) for s in s_range])
    seff_vals = g2_vals * s_range

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(s_range / 1000, g2_vals, "b-", lw=2)
    axes[0].axhline(1.0, color="gray", ls="--", alpha=0.5)
    axes[0].axvline(sigma.max() / 1000, color="r", ls="--", alpha=0.5,
                    label=f"Peak stress = {sigma.max()/1000:.0f} kPa")
    axes[0].set_xlabel("Stress [kPa]")
    axes[0].set_ylabel("g2 [-]")
    axes[0].set_title("g2(sigma): nonlinear scaling function")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(s_range / 1000, seff_vals / 1000, "r-", lw=2, label="sigma_eff")
    axes[1].plot(s_range / 1000, s_range / 1000, "k--", lw=1, label="linear (sigma_eff=sigma)")
    axes[1].set_xlabel("Applied Stress [kPa]")
    axes[1].set_ylabel("Effective Stress [kPa]")
    axes[1].set_title("Effective stress vs applied stress")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plot_g2_function.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ---- Plot 3: Predictions on new stress inputs ----
    dt = DT_UNIFORM

    # Prediction 1: Creep + recovery at same peak stress
    t1 = np.arange(0, 1000, dt)
    s1 = np.where(t1 < 500, sigma.max(), 0.0)
    e1 = forward_simulate(t1, s1, D0, Dk, eps0, c2_fit, sref_fit)

    # Prediction 2: Creep at HALF peak stress (tests the nonlinearity)
    t2 = np.arange(0, 1000, dt)
    s2 = np.where(t2 < 500, sigma.max() * 0.5, 0.0)
    e2 = forward_simulate(t2, s2, D0, Dk, eps0, c2_fit, sref_fit)

    # Prediction 3: Trapezoidal loading
    t3 = np.arange(0, 800, dt)
    s3 = np.zeros_like(t3)
    for i, ti in enumerate(t3):
        if ti < 100:
            s3[i] = sigma.max() * ti / 100
        elif ti < 400:
            s3[i] = sigma.max()
        elif ti < 500:
            s3[i] = sigma.max() * (1 - (ti - 400) / 100)
        else:
            s3[i] = 0
    e3 = forward_simulate(t3, s3, D0, Dk, eps0, c2_fit, sref_fit)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    axes[0, 0].plot(t1, s1 / 1000, "b-", lw=2)
    axes[0, 0].set_title("Pred 1: Creep at full load")
    axes[0, 0].set_ylabel("Stress [kPa]")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t1, e1, "r-", lw=2)
    axes[0, 1].set_title("Pred 1: Strain response")
    axes[0, 1].set_ylabel("Strain [-]")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t2, s2 / 1000, "b-", lw=2)
    axes[1, 0].set_title("Pred 2: Creep at HALF load")
    axes[1, 0].set_ylabel("Stress [kPa]")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(t2, e2, "r-", lw=2)
    axes[1, 1].set_title("Pred 2: Strain response (nonlinear effect visible)")
    axes[1, 1].set_ylabel("Strain [-]")
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 0].plot(t3, s3 / 1000, "b-", lw=2)
    axes[2, 0].set_title("Pred 3: Trapezoidal loading")
    axes[2, 0].set_ylabel("Stress [kPa]")
    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(t3, e3, "r-", lw=2)
    axes[2, 1].set_title("Pred 3: Strain response")
    axes[2, 1].set_ylabel("Strain [-]")
    axes[2, 1].set_xlabel("Time [s]")
    axes[2, 1].grid(True, alpha=0.3)

    plt.suptitle("PREDICTIONS with frozen parameters", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plot_predictions_g2.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ---- Plot 4: Long-time recovery ----
    t_long = np.arange(0, t[-1] + 24 * 3600, dt)
    s_long = np.zeros_like(t_long)
    s_long[:len(sigma)] = sigma
    e_long = forward_simulate(t_long, s_long, D0, Dk, eps0, c2_fit, sref_fit)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, eps_data, "k.", ms=1, alpha=0.5, label="Experimental")
    ax.plot(t, eps_pred, "r-", lw=2, label="Model (fit window)")
    ax.plot(t_long / 3600, e_long, "g-", lw=1.5, alpha=0.7,
            label="Recovery extrapolation (24h)")
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Strain [-]")
    ax.set_title("Long-time Recovery Prediction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot_recovery_g2.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nAll plots saved.")
