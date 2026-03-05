"""
================================================================================
CROSS-VALIDATION: Fit on Rep 1, Predict Reps 2 & 3
================================================================================

This script:
  1. Loads all 3 replicate CSV files
  2. Fits the model on Rep 1 only (finds D0, Dk, eps0, c2, sigma_ref)
  3. Uses forward_simulate with FROZEN parameters to predict Reps 2 & 3
  4. Compares predictions vs experimental data

If predictions are good → model captures real material behavior
If predictions are bad  → model was overfitting noise in Rep 1
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, nnls

# ============================================================
# CONFIG — EDIT THESE
# ============================================================
# Paths to your 3 replicate CSV files
CSV_REP1 = r"C:/Users/random/folder/rep1.csv"
CSV_REP2 = r"C:/Users/random/folder/rep2.csv"
CSV_REP3 = r"C:/Users/random/folder/rep3.csv"

# Column names (same for all files)
TIME_COL   = "time"
STRESS_COL = "Stress"
STRAIN_COL = "Strain"

DT_UNIFORM = 0.25  # s

# Retardation spectrum (same as fitting code)
TAU = np.array([0.15, 1.0, 5.0, 20.0, 80.0, 300.0, 1500.0, 8000.0, 50000.0])


# ============================================================
# Functions (copied from the fitting code — keep them identical)
# ============================================================
def load_and_prepare(path, time_col, stress_col, strain_col, dt):
    df = pd.read_csv(path)
    missing = [c for c in [time_col, stress_col, strain_col] if c not in df.columns]
    if missing:
        print("CSV columns found:", list(df.columns))
        raise KeyError(f"Missing columns: {missing}")

    t = df[time_col].to_numpy(float)
    sigma = df[stress_col].to_numpy(float)
    eps = df[strain_col].to_numpy(float)

    order = np.argsort(t)
    t, sigma, eps = t[order], sigma[order], eps[order]

    t_u = np.arange(t[0], t[-1] + dt, dt)
    sigma_u = np.interp(t_u, t, sigma)
    eps_u = np.interp(t_u, t, eps)

    t_u = t_u - t_u[0]
    eps_u = eps_u - eps_u[0]

    return t_u, sigma_u, eps_u


def g2(sigma_val, c2, sigma_ref):
    x = np.abs(sigma_val) / max(sigma_ref, 1e-12)
    return 1.0 / (1.0 + c2 * x)


def build_design_matrix(t, sigma, c2, sigma_ref):
    n = len(t)
    n_modes = len(TAU)
    A = np.zeros((n, n_modes + 2))
    A[:, 0] = sigma

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

    A[:, -1] = 1.0
    return A


def forward_simulate(t, sigma, D0, Dk, eps0, c2, sigma_ref):
    """
    Forward simulation with ALL parameters FROZEN.
    This is the prediction function — no fitting happens here.
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


def fit_model(t, sigma, eps_data):
    def residuals(x_nl):
        c2_val = np.exp(x_nl[0])
        sref_val = np.exp(x_nl[1])
        A = build_design_matrix(t, sigma, c2_val, sref_val)
        x_lin, _ = nnls(A, eps_data)
        eps_pred = A @ x_lin
        return eps_pred - eps_data

    x0 = np.array([np.log(2.0), np.log(3000.0)])
    lb = np.array([np.log(0.01), np.log(10.0)])
    ub = np.array([np.log(1000.0), np.log(1e8)])

    sol = least_squares(
        residuals, x0, bounds=(lb, ub),
        ftol=1e-12, xtol=1e-12, gtol=1e-12,
        max_nfev=500
    )

    c2_fit = np.exp(sol.x[0])
    sref_fit = np.exp(sol.x[1])

    A = build_design_matrix(t, sigma, c2_fit, sref_fit)
    x_lin, _ = nnls(A, eps_data)

    D0 = x_lin[0]
    Dk = x_lin[1:-1]
    eps0 = x_lin[-1]

    eps_pred = A @ x_lin
    rmse = np.sqrt(np.mean((eps_pred - eps_data) ** 2))

    return D0, Dk, eps0, c2_fit, sref_fit, eps_pred, rmse


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # ---- Step 1: Load all 3 replicates ----
    print("Loading replicates...")
    t1, s1, e1 = load_and_prepare(CSV_REP1, TIME_COL, STRESS_COL, STRAIN_COL, DT_UNIFORM)
    t2, s2, e2 = load_and_prepare(CSV_REP2, TIME_COL, STRESS_COL, STRAIN_COL, DT_UNIFORM)
    t3, s3, e3 = load_and_prepare(CSV_REP3, TIME_COL, STRESS_COL, STRAIN_COL, DT_UNIFORM)

    print(f"  Rep 1: {len(t1)} points, t=[{t1[0]:.1f}, {t1[-1]:.1f}] s")
    print(f"  Rep 2: {len(t2)} points, t=[{t2[0]:.1f}, {t2[-1]:.1f}] s")
    print(f"  Rep 3: {len(t3)} points, t=[{t3[0]:.1f}, {t3[-1]:.1f}] s")

    # ---- Step 2: Fit on Rep 1 ONLY ----
    print(f"\n{'='*60}")
    print("FITTING on Rep 1 only")
    print(f"{'='*60}")
    D0, Dk, eps0, c2_fit, sref_fit, e1_pred, rmse1 = fit_model(t1, s1, e1)

    print(f"\nFitted parameters (from Rep 1):")
    print(f"  c2        = {c2_fit:.6f}")
    print(f"  sigma_ref = {sref_fit:.2f} Pa")
    print(f"  D0        = {D0:.6e}")
    for k, (tau_k, dk) in enumerate(zip(TAU, Dk)):
        if dk > 1e-15:
            print(f"  D{k+1}        = {dk:.6e}  (tau={tau_k:.1f} s)")
    print(f"  eps0      = {eps0:.6e}")
    print(f"  RMSE (fit on Rep 1) = {rmse1:.6e}")

    # ---- Step 3: PREDICT Reps 2 & 3 (no re-fitting!) ----
    print(f"\n{'='*60}")
    print("PREDICTING Rep 2 & Rep 3 with FROZEN parameters")
    print(f"{'='*60}")

    # Predict Rep 2 using Rep 2's stress input + Rep 1's parameters
    e2_pred = forward_simulate(t2, s2, D0, Dk, eps0, c2_fit, sref_fit)
    rmse2 = np.sqrt(np.mean((e2_pred - e2) ** 2))
    print(f"  RMSE (prediction on Rep 2) = {rmse2:.6e}")

    # Predict Rep 3 using Rep 3's stress input + Rep 1's parameters
    e3_pred = forward_simulate(t3, s3, D0, Dk, eps0, c2_fit, sref_fit)
    rmse3 = np.sqrt(np.mean((e3_pred - e3) ** 2))
    print(f"  RMSE (prediction on Rep 3) = {rmse3:.6e}")

    # ---- Step 4: Compare ----
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"  Rep 1 (FIT):        RMSE = {rmse1:.6e}")
    print(f"  Rep 2 (PREDICT):    RMSE = {rmse2:.6e}")
    print(f"  Rep 3 (PREDICT):    RMSE = {rmse3:.6e}")
    print(f"  Ratio Rep2/Rep1:    {rmse2/rmse1:.2f}x")
    print(f"  Ratio Rep3/Rep1:    {rmse3/rmse1:.2f}x")

    if max(rmse2, rmse3) < 3 * rmse1:
        print("\n  RESULT: Prediction RMSE is within 3x of fit RMSE.")
        print("  The model is capturing real material behavior, not noise.")
    else:
        print("\n  WARNING: Prediction RMSE is much larger than fit RMSE.")
        print("  The model may be overfitting Rep 1.")

    # ---- Step 5: Plots ----

    # Plot 1: All three reps with fit/predictions
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

    # Rep 1 (FIT)
    axes[0].plot(t1, e1, "k.", ms=1, alpha=0.4, label="Rep 1 (experimental)")
    axes[0].plot(t1, e1_pred, "r-", lw=2, label=f"FIT (RMSE={rmse1:.4e})")
    axes[0].set_ylabel("Strain [-]")
    axes[0].set_title("Rep 1 — FITTED (parameters calibrated here)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Rep 2 (PREDICT)
    axes[1].plot(t2, e2, "k.", ms=1, alpha=0.4, label="Rep 2 (experimental)")
    axes[1].plot(t2, e2_pred, "b-", lw=2, label=f"PREDICTION (RMSE={rmse2:.4e})")
    axes[1].set_ylabel("Strain [-]")
    axes[1].set_title("Rep 2 — PREDICTED (frozen parameters from Rep 1)")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    # Rep 3 (PREDICT)
    axes[2].plot(t3, e3, "k.", ms=1, alpha=0.4, label="Rep 3 (experimental)")
    axes[2].plot(t3, e3_pred, "b-", lw=2, label=f"PREDICTION (RMSE={rmse3:.4e})")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Strain [-]")
    axes[2].set_title("Rep 3 — PREDICTED (frozen parameters from Rep 1)")
    axes[2].legend(loc="lower right")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Cross-Validation: Fit on Rep 1, Predict Reps 2 & 3",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("crossval_all_reps.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Plot 2: Overlay all three on same axes
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t1, e1, "k.", ms=1, alpha=0.3, label="Rep 1 (exp)")
    ax.plot(t2, e2, "gray", ms=1, alpha=0.3, label="Rep 2 (exp)")
    ax.plot(t3, e3, "silver", ms=1, alpha=0.3, label="Rep 3 (exp)")
    ax.plot(t1, e1_pred, "r-", lw=2, label=f"Rep 1 FIT (RMSE={rmse1:.4e})")
    ax.plot(t2, e2_pred, "b-", lw=2, label=f"Rep 2 PRED (RMSE={rmse2:.4e})")
    ax.plot(t3, e3_pred, "g-", lw=2, label=f"Rep 3 PRED (RMSE={rmse3:.4e})")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Strain [-]")
    ax.set_title("All Replicates: Fit vs Predictions (overlay)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("crossval_overlay.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nPlots saved: crossval_all_reps.png, crossval_overlay.png")
