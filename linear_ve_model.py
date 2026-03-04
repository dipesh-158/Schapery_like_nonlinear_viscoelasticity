"""
================================================================================
LINEAR VISCOELASTIC MODEL (Generalized Kelvin-Voigt / Prony Series)
================================================================================

THEORY
------
This is the simplest model that can capture what your experimental data shows.

Your material exhibits:
  1. Instantaneous elastic response (strain jumps when stress is applied)
  2. Delayed/creep response (strain continues to grow under constant stress)
  3. Incomplete recovery (strain doesn't return to zero when stress is removed)
  4. Cycle-to-cycle ratcheting (baseline strain drifts upward each cycle)

All of these are classic LINEAR viscoelastic features. You do NOT need
nonlinearity to explain them — you just need multiple relaxation timescales.

THE MODEL (compliance form):
  
  The constitutive equation in integral form is:
  
    eps(t) = D(t) * sigma(t)   [convolution]
  
  where D(t) is the creep compliance function:
  
    D(t) = D0 + sum_{k=1}^{N} Dk * (1 - exp(-t/tau_k))
  
  D0 = instantaneous (glassy) compliance [1/Pa]
  Dk = retardation strength of mode k [1/Pa]
  tau_k = retardation time of mode k [s]

  In incremental/recursive form (what we actually compute), each mode k
  has an internal variable q_k(t) that evolves as:
  
    q_k(t+dt) = exp(-dt/tau_k) * q_k(t) + (1 - exp(-dt/tau_k)) * Dk * sigma_mid
  
  where sigma_mid = 0.5*(sigma(t) + sigma(t+dt)).
  
  Total strain:
    eps(t) = eps0 + D0 * sigma(t) + sum_k q_k(t)

  This recursive form is EXACT for piecewise-linear stress histories and
  numerically stable for any timestep.

WHY MULTIPLE TIMESCALES MATTER:
  - Fast modes (tau ~ 0.1-10 s): capture the rapid loading/unloading response
  - Medium modes (tau ~ 50-200 s): capture the partial recovery during hold
  - Slow modes (tau ~ 1000+ s): capture the ratcheting / permanent-looking set
  
  The "incomplete recovery" you see is NOT permanent deformation — it's just
  very slow recovery from modes with tau >> your test duration. A mode with
  tau=50000s needs ~14 hours to recover 63%.

FITTING STRATEGY:
  - Fix the retardation times tau_k (log-spaced to cover the timescales in
    your data). This is standard practice.
  - Fit ONLY the compliance amplitudes D0, Dk, and offset eps0.
  - Since all Dk >= 0 (thermodynamic requirement), we use Non-Negative
    Least Squares (NNLS). This is a LINEAR problem — no iterative
    optimization, no local minima, guaranteed global optimum.

WHAT THIS MODEL CAN AND CANNOT DO:
  CAN:    Predict strain for ANY stress input at the SAME stress level
          (since it's linear, actually any stress level — but that's the
          assumption, not a proven fact with your data).
  CANNOT: Capture true stress-level-dependent nonlinearity (if it exists).
          For that, you'd need multi-stress-level data to justify adding
          Schapery nonlinearity on top of this.

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls

# ============================================================
# CONFIGURATION
# ============================================================
# We don't have access to your CSV, so we'll generate synthetic data
# that matches your experimental profile. Replace this section with
# your actual data loading when you run it.

DT = 0.25  # s, uniform timestep

# Retardation spectrum: log-spaced from 0.1s to 50000s
# This covers all timescales visible in your ~550s test plus long-term
TAU = np.array([0.15, 1.0, 5.0, 20.0, 80.0, 300.0, 1500.0, 8000.0, 50000.0])
# 9 modes: you could use fewer, but this gives good coverage

# ============================================================
# STEP 1: Generate synthetic stress that matches your experiment
# ============================================================
def make_triangular_stress(t, peak_stress=80000.0):
    """
    Approximate your experimental stress profile:
    3 triangular ramp-unload cycles with hold periods.
    
    From your plot:
      Cycle 1: ramp up 0-40s, ramp down 40-80s, hold 80-130s
      Cycle 2: ramp up 130-200s, ramp down 200-250s, hold 250-300s  
      Cycle 3: ramp up 300-370s, ramp down 370-420s, hold 420-550s
    (approximate timings from your stress plot)
    """
    sigma = np.zeros_like(t)
    
    # Define cycles as (start_ramp_up, peak_time, end_ramp_down, end_hold)
    cycles = [
        (0, 40, 80, 130),
        (130, 200, 250, 300),
        (300, 370, 420, 550),
    ]
    
    for t_start, t_peak, t_end, t_hold in cycles:
        for i, ti in enumerate(t):
            if t_start <= ti < t_peak:
                # Ramp up
                sigma[i] += peak_stress * (ti - t_start) / (t_peak - t_start)
            elif t_peak <= ti < t_end:
                # Ramp down
                sigma[i] += peak_stress * (1.0 - (ti - t_peak) / (t_end - t_peak))
            # else: hold at zero (already 0)
    
    return sigma


# ============================================================
# STEP 2: THE MODEL — Forward simulation
# ============================================================
def forward_linear_ve(t, sigma, D0, Dk, eps0=0.0):
    """
    Forward simulation of linear viscoelastic model.
    
    Parameters (ALL FIXED after fitting, NONE change during prediction):
        t      : time array [s]
        sigma  : stress array [Pa] — THIS is the only input that changes
        D0     : instantaneous compliance [1/Pa]
        Dk     : array of retardation strengths [1/Pa], shape (n_modes,)
        eps0   : strain offset
    
    Returns:
        eps_total : predicted strain array
        eps_modes : (n_modes,) contribution of each mode at each timestep
    
    The physics at each timestep:
        1. Compute dt and sigma_mid (midpoint stress)
        2. For each mode k:
           - Compute decay factor r = exp(-dt/tau_k)
           - Update: q_k = r*q_k + (1-r) * Dk * sigma_mid
           This is the analytical solution of: tau_k * dq_k/dt + q_k = Dk*sigma
        3. Total strain: eps = eps0 + D0*sigma + sum(q_k)
    """
    n = len(t)
    n_modes = len(Dk)
    
    # Internal state variables (one per mode)
    q = np.zeros(n_modes)
    
    # Output arrays
    eps_total = np.zeros(n)
    eps_modes = np.zeros((n_modes, n))
    
    # Initial condition
    eps_total[0] = eps0 + D0 * sigma[0]
    
    for i in range(1, n):
        dt = t[i] - t[i-1]
        sigma_mid = 0.5 * (sigma[i] + sigma[i-1])
        
        for k in range(n_modes):
            r = np.exp(-dt / TAU[k])
            q[k] = r * q[k] + (1.0 - r) * Dk[k] * sigma_mid
            eps_modes[k, i] = q[k]
        
        eps_total[i] = eps0 + D0 * sigma[i] + np.sum(q)
    
    return eps_total, eps_modes


# ============================================================
# STEP 3: FITTING — Build design matrix and solve with NNLS
# ============================================================
def build_design_matrix(t, sigma):
    """
    Build matrix A such that:
        eps(t) = A @ [D0, D1, D2, ..., DN, eps0]
    
    This turns fitting into a LINEAR problem: find x >= 0 such that A@x ≈ eps.
    
    Column 0: D0 contribution  -> sigma(t)           (instantaneous)
    Column k: Dk contribution  -> q_k(t) with Dk=1   (each mode's basis function)
    Column N+1: eps0           -> 1                   (constant offset)
    """
    n = len(t)
    n_modes = len(TAU)
    
    # A has n_modes + 2 columns: [D0, D1, ..., DN, eps0]
    A = np.zeros((n, n_modes + 2))
    
    # Column 0: instantaneous compliance basis = sigma(t)
    A[:, 0] = sigma
    
    # Columns 1..N: retardation mode basis functions
    # Run forward with Dk=1 for each mode independently
    for k in range(n_modes):
        z = 0.0  # internal variable for mode k with unit compliance
        for i in range(1, n):
            dt = t[i] - t[i-1]
            sigma_mid = 0.5 * (sigma[i] + sigma[i-1])
            r = np.exp(-dt / TAU[k])
            z = r * z + (1.0 - r) * sigma_mid  # note: Dk=1 here, so basis
            A[i, k + 1] = z
    
    # Last column: constant offset (eps0)
    A[:, -1] = 1.0
    
    return A


def fit_linear_ve(t, sigma, eps_data):
    """
    Fit the linear viscoelastic model using NNLS.
    
    Returns:
        D0, Dk (array), eps0, eps_pred, rmse
    """
    A = build_design_matrix(t, sigma)
    
    # NNLS enforces all coefficients >= 0
    # This is physically correct: compliances must be non-negative
    # Note: eps0 column is also constrained >= 0. If you need negative eps0,
    # you can handle it separately, but for your data eps0 ~ 0 or positive.
    
    x, rnorm = nnls(A, eps_data)
    
    D0 = x[0]
    Dk = x[1:-1]
    eps0 = x[-1]
    
    eps_pred = A @ x
    rmse = np.sqrt(np.mean((eps_pred - eps_data) ** 2))
    
    return D0, Dk, eps0, eps_pred, rmse


# ============================================================
# STEP 4: Generate synthetic experimental data for demonstration
# ============================================================
def generate_synthetic_data():
    """
    Create synthetic data that mimics your experimental strain response.
    
    Key features to reproduce:
      - Peak strain ~0.55-0.6 at peak stress
      - Recovery to ~0.28-0.35 during hold (not zero!)
      - Slight upward drift cycle-to-cycle
      - Smooth, not noisy
    
    We use a "true" linear VE model with known parameters to generate this.
    In your case, replace this with your actual CSV data.
    """
    t = np.arange(0, 551, DT)
    sigma = make_triangular_stress(t)
    
    # "True" parameters (chosen to mimic your plots)
    D0_true = 2e-7        # small instantaneous compliance
    Dk_true = np.array([
        3.5e-5,   # tau=0.15s  (very fast)
        2.0e-5,   # tau=1.0s
        1.5e-5,   # tau=5.0s
        1.0e-5,   # tau=20s
        5.0e-6,   # tau=80s
        2.0e-6,   # tau=300s   (medium — partial recovery)
        1.0e-6,   # tau=1500s  (slow — contributes to ratchet)
        8.0e-7,   # tau=8000s  (very slow)
        5.0e-5,   # tau=50000s (quasi-permanent on test timescale)
    ])
    eps0_true = 0.0
    
    eps_true, _ = forward_linear_ve(t, sigma, D0_true, Dk_true, eps0_true)
    
    # Add small noise
    np.random.seed(42)
    eps_noisy = eps_true + np.random.normal(0, 0.002, len(t))
    
    return t, sigma, eps_noisy, eps_true


# ============================================================
# MAIN: Fit + Predict + Plots
# ============================================================
if __name__ == "__main__":
    
    # ---- PART A: Generate data (replace with your CSV loading) ----
    # ================================================================
    # TO USE YOUR ACTUAL DATA, replace this block with:
    #
    #   import pandas as pd
    #   df = pd.read_csv("your_file.csv")
    #   t_raw = df["Time"].to_numpy(float)
    #   sigma_raw = df["Stress"].to_numpy(float)
    #   eps_raw = df["Strain "].to_numpy(float)
    #   # sort, resample, zero-baseline as in your original code
    # ================================================================
    
    t, sigma, eps_data, eps_true = generate_synthetic_data()
    
    print("=" * 60)
    print("LINEAR VISCOELASTIC MODEL (Prony Series)")
    print("=" * 60)
    print(f"Data points: {len(t)}")
    print(f"Retardation times (fixed): {TAU}")
    print(f"Number of modes: {len(TAU)}")
    
    # ---- PART B: FIT ----
    print("\nFitting (NNLS — linear, one-shot, global optimum)...")
    D0, Dk, eps0, eps_pred, rmse = fit_linear_ve(t, sigma, eps_data)
    
    print(f"\n{'='*60}")
    print("FIT RESULTS")
    print(f"{'='*60}")
    print(f"  D0 (instantaneous) = {D0:.6e} [1/Pa]")
    for k, (tau_k, dk) in enumerate(zip(TAU, Dk)):
        print(f"  D{k+1} = {dk:.6e} [1/Pa]   (tau = {tau_k:>8.1f} s)")
    print(f"  eps0               = {eps0:.6e}")
    print(f"\n  RMSE = {rmse:.6e}")
    print(f"  Number of parameters: {len(TAU) + 2} (all linear, no iteration needed)")
    
    # ---- PART C: PREDICT on new stress input ----
    # This is the whole point: use SAME parameters, DIFFERENT stress
    print(f"\n{'='*60}")
    print("PREDICTION: Using fitted parameters on new stress inputs")
    print(f"{'='*60}")
    
    # Prediction 1: Single sustained creep at half the peak stress
    t_creep = np.arange(0, 1000, DT)
    sigma_creep = np.where(t_creep < 500, 40000.0, 0.0)  # step on, then off
    eps_creep, _ = forward_linear_ve(t_creep, sigma_creep, D0, Dk, eps0)
    
    # Prediction 2: Different triangular profile (slower, lower amplitude)
    t_slow = np.arange(0, 800, DT)
    sigma_slow = np.zeros_like(t_slow)
    # Single slow triangle: ramp over 200s, hold 200s, ramp down 200s, recover 200s
    for i, ti in enumerate(t_slow):
        if ti < 200:
            sigma_slow[i] = 50000.0 * ti / 200.0
        elif ti < 400:
            sigma_slow[i] = 50000.0
        elif ti < 600:
            sigma_slow[i] = 50000.0 * (1.0 - (ti - 400) / 200.0)
        else:
            sigma_slow[i] = 0.0
    eps_slow, _ = forward_linear_ve(t_slow, sigma_slow, D0, Dk, eps0)
    
    # Prediction 3: Sinusoidal stress
    t_sin = np.arange(0, 600, DT)
    sigma_sin = 40000.0 * np.sin(2 * np.pi * t_sin / 100.0)
    sigma_sin = np.maximum(sigma_sin, 0)  # only positive (compression test)
    eps_sin, _ = forward_linear_ve(t_sin, sigma_sin, D0, Dk, eps0)
    
    # ---- PART D: PLOTS ----
    
    # Plot 1: Fit quality
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    
    ax1 = axes[0]
    ax1.plot(t, sigma / 1000, 'b-', lw=2)
    ax1.set_ylabel("Stress [kPa]")
    ax1.set_title("Input Stress (Training Data)")
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(t, eps_data, 'k.', ms=1, label="Experimental", alpha=0.5)
    ax2.plot(t, eps_pred, 'r-', lw=2, label=f"Linear VE Model (RMSE={rmse:.4e})")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Strain [-]")
    ax2.set_title("Fit: Linear Viscoelastic Model")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("/home/claude/plot_fit.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Predictions on new stress inputs
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # Creep + recovery
    axes[0, 0].plot(t_creep, sigma_creep / 1000, 'b-', lw=2)
    axes[0, 0].set_ylabel("Stress [kPa]")
    axes[0, 0].set_title("Prediction 1: Creep + Recovery (stress)")
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(t_creep, eps_creep, 'r-', lw=2)
    axes[0, 1].set_ylabel("Strain [-]")
    axes[0, 1].set_title("Prediction 1: Creep + Recovery (strain)")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Trapezoidal loading
    axes[1, 0].plot(t_slow, sigma_slow / 1000, 'b-', lw=2)
    axes[1, 0].set_ylabel("Stress [kPa]")
    axes[1, 0].set_title("Prediction 2: Trapezoidal Loading (stress)")
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(t_slow, eps_slow, 'r-', lw=2)
    axes[1, 1].set_ylabel("Strain [-]")
    axes[1, 1].set_title("Prediction 2: Trapezoidal Loading (strain)")
    axes[1, 1].grid(True, alpha=0.3)
    
    # Sinusoidal
    axes[2, 0].plot(t_sin, sigma_sin / 1000, 'b-', lw=2)
    axes[2, 0].set_ylabel("Stress [kPa]")
    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 0].set_title("Prediction 3: Half-Sine Pulses (stress)")
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(t_sin, eps_sin, 'r-', lw=2)
    axes[2, 1].set_ylabel("Strain [-]")
    axes[2, 1].set_xlabel("Time [s]")
    axes[2, 1].set_title("Prediction 3: Half-Sine Pulses (strain)")
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.suptitle("PREDICTIONS: Same parameters, different stress inputs", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("/home/claude/plot_predictions.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Retardation spectrum (bar chart)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(TAU)), Dk * 1e6, tick_label=[f"{t:.1f}" for t in TAU])
    ax.set_xlabel("Retardation time tau [s]")
    ax.set_ylabel("Dk [μPa⁻¹]")
    ax.set_title("Fitted Retardation Spectrum")
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("/home/claude/plot_spectrum.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Long-time recovery extrapolation
    t_long = np.arange(0, t[-1] + 24*3600, DT)
    sigma_long = np.zeros_like(t_long)
    sigma_long[:len(sigma)] = sigma
    eps_long, _ = forward_linear_ve(t_long, sigma_long, D0, Dk, eps0)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, eps_data, 'k.', ms=1, alpha=0.5, label="Experimental")
    ax.plot(t, eps_pred, 'r-', lw=2, label="Model (fit window)")
    ax.plot(t_long, eps_long, 'g-', lw=1.5, alpha=0.7, label="Recovery extrapolation")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Strain [-]")
    ax.set_title("Long-time Recovery Prediction (24h extrapolation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/home/claude/plot_recovery.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nPlots saved.")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
This linear viscoelastic model:
  - Has {n_params} parameters (all fitted by NNLS in one shot)
  - No iterative optimization, no local minima
  - Every parameter is physically meaningful and identifiable
  - Can predict strain for ANY stress input using the same frozen parameters
  - The 'incomplete recovery' comes naturally from slow modes (tau >> test time)
  
LIMITATIONS:
  - Assumes linearity (proportional scaling with stress)
  - Fitted at one stress level — predicting at very different stress levels
    is an extrapolation that should be validated experimentally
  - Does not capture true permanent/plastic deformation (if any exists)
  
NEXT STEP:
  If this model fits your data well, test it by:
  1. Fitting on cycle 1-2 only, predicting cycle 3
  2. If you get more data at different stress levels, compare predictions
  3. Only if linear model FAILS, add Schapery nonlinearity on top
""".format(n_params=len(TAU) + 2))
