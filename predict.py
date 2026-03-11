"""
================================================================================
PREDICTION SCRIPT
================================================================================
Uses frozen parameters from experimental fit to predict strain
for any constant stress magnitude and duration.

Just change STRESS_KPA and DURATION_SEC below and run.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# FROZEN PARAMETERS (from your fit — never change these)
# ============================================================
TAU = np.array([0.15, 1.0, 5.0, 20.0, 80.0, 300.0, 1500.0, 8000.0, 50000.0])
Dk  = np.array([1.053909e-4, 3.485715e-5, 5.073150e-5, 0.0, 7.369150e-5, 0.0, 0.0, 0.0, 3.222032e-3])
D0  = 2.901145e-7
EPS0 = 2.220802e-1
C2  = 2.038971
SIGMA_REF = 3262.85

# ============================================================
# USER CONTROLS — CHANGE THESE
# ============================================================
STRESS_KPA    = 85.0     # Applied stress in kPa (try: 1, 10, 40, 85, 120)
DURATION_SEC  = 60.0     # How long stress is applied [s]
RECOVERY_SEC  = 60.0     # Recovery time after unloading [s] (set to 0 for creep only)

# ============================================================
# Model functions
# ============================================================
def g2(sigma_val):
    x = np.abs(sigma_val) / max(SIGMA_REF, 1e-12)
    return 1.0 / (1.0 + C2 * x)


def forward_simulate(t, sigma):
    n = len(t)
    q = np.zeros(len(TAU))
    eps = np.zeros(n)
    eps[0] = EPS0 + D0 * sigma[0]

    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if dt <= 0:
            dt = 1e-12
        sigma_mid = 0.5 * (sigma[i] + sigma[i - 1])
        sigma_eff = g2(sigma_mid) * sigma_mid

        for k in range(len(TAU)):
            r = np.exp(-dt / TAU[k])
            q[k] = r * q[k] + (1.0 - r) * Dk[k] * sigma_eff

        eps[i] = EPS0 + D0 * sigma[i] + np.sum(q)

    return eps


# ============================================================
# Build time and stress arrays
# ============================================================
stress_pa = STRESS_KPA * 1000.0
total_time = DURATION_SEC + RECOVERY_SEC

# Choose dt based on duration
if total_time <= 1:
    dt = 0.001
elif total_time <= 10:
    dt = 0.005
elif total_time <= 100:
    dt = 0.025
else:
    dt = 0.25

t = np.arange(0, total_time + dt, dt)
sigma = np.where(t <= DURATION_SEC, stress_pa, 0.0)

# ============================================================
# Run prediction
# ============================================================
eps = forward_simulate(t, sigma)

# ============================================================
# Print summary
# ============================================================
g2_val = g2(stress_pa)
sigma_eff_val = g2_val * stress_pa

print(f"{'='*50}")
print(f"PREDICTION SUMMARY")
print(f"{'='*50}")
print(f"  Applied stress:  {STRESS_KPA} kPa ({stress_pa:.0f} Pa)")
print(f"  g2(sigma):       {g2_val:.6f}")
print(f"  sigma_eff:       {sigma_eff_val:.1f} Pa")
print(f"  Creep duration:  {DURATION_SEC} s")
print(f"  Recovery time:   {RECOVERY_SEC} s")
print(f"")
print(f"  Strain at t=0:          {eps[0]:.6f}")

# Strain at end of creep
idx_creep_end = np.searchsorted(t, DURATION_SEC)
print(f"  Strain at end of creep: {eps[idx_creep_end]:.6f}")

if RECOVERY_SEC > 0:
    print(f"  Strain at end of recovery: {eps[-1]:.6f}")
    print(f"  Residual strain:           {eps[-1] - EPS0:.6f}")

print(f"  Peak strain:            {np.max(eps):.6f}")

# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axes[0].plot(t, sigma / 1000, "b-", lw=2)
axes[0].set_ylabel("Stress [kPa]")
axes[0].set_title(f"Prediction: {STRESS_KPA} kPa for {DURATION_SEC}s" +
                  (f" + {RECOVERY_SEC}s recovery" if RECOVERY_SEC > 0 else ""))
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, eps, "r-", lw=2)
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Strain [-]")
axes[1].set_title(f"Predicted Strain (g2={g2_val:.4f}, σ_eff={sigma_eff_val:.1f} Pa)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("prediction.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# BONUS: Multi-stress comparison
# ============================================================
stress_levels = [1, 5, 10, 20, 40, 60, 85, 120]  # kPa

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Creep at different stress levels
t_creep = np.arange(0, DURATION_SEC + dt, dt)
for s_kpa in stress_levels:
    s_pa = s_kpa * 1000.0
    sigma_creep = np.full_like(t_creep, s_pa)
    eps_creep = forward_simulate(t_creep, sigma_creep)
    axes[0].plot(t_creep, eps_creep, lw=2, label=f"{s_kpa} kPa")

axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Strain [-]")
axes[0].set_title(f"Creep at different stress levels ({DURATION_SEC}s)")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# g2 and sigma_eff vs stress
s_range = np.linspace(0, 150000, 500)
g2_range = np.array([g2(s) for s in s_range])
seff_range = g2_range * s_range

axes[1].plot(s_range / 1000, seff_range / 1000, "r-", lw=2, label="σ_eff")
axes[1].plot(s_range / 1000, s_range / 1000, "k--", lw=1, alpha=0.5, label="Linear (σ_eff = σ)")
axes[1].set_xlabel("Applied Stress [kPa]")
axes[1].set_ylabel("Effective Stress [kPa]")
axes[1].set_title("Nonlinear stress mapping (g2)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("multi_stress_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nPlots saved: prediction.png, multi_stress_comparison.png")
