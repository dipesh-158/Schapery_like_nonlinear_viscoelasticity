"""
================================================================================
SYSTEMATIC MODEL ANALYSIS
================================================================================
Sweeps across a grid of stress levels and time durations.
For each (stress, time) pair, runs the model and records:
  - Strain at end of creep
  - Strain at end of recovery
  - Residual strain (unrecovered)

Outputs:
  1. Heatmap of strain vs (stress, time)
  2. Creep curve families at fixed stresses
  3. Creep curve families at fixed durations
  4. CSV table of all results for your records
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

# ============================================================
# FROZEN PARAMETERS
# ============================================================
TAU = np.array([0.15, 1.0, 5.0, 20.0, 80.0, 300.0, 1500.0, 8000.0, 50000.0])
Dk  = np.array([1.053909e-4, 3.485715e-5, 5.073150e-5, 0.0, 7.369150e-5, 0.0, 0.0, 0.0, 3.222032e-3])
D0  = 2.901145e-7
EPS0 = 2.220802e-1
C2  = 2.038971
SIGMA_REF = 3262.85

# ============================================================
# SWEEP CONFIGURATION — EDIT THESE
# ============================================================
# Stress levels to test (kPa)
STRESS_LEVELS = [0.5, 1, 2, 5, 10, 20, 40, 60, 85, 100, 120, 150]

# Creep durations to test (seconds)
TIME_LEVELS = [0.1, 0.5, 1, 5, 10, 30, 60, 120, 300, 500]

# Recovery time after each creep (set to 0 to skip recovery analysis)
RECOVERY_SEC = 300.0

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
        dt_step = t[i] - t[i - 1]
        if dt_step <= 0:
            dt_step = 1e-12
        sigma_mid = 0.5 * (sigma[i] + sigma[i - 1])
        sigma_eff = g2(sigma_mid) * sigma_mid

        for k in range(len(TAU)):
            r = np.exp(-dt_step / TAU[k])
            q[k] = r * q[k] + (1.0 - r) * Dk[k] * sigma_eff

        eps[i] = EPS0 + D0 * sigma[i] + np.sum(q)

    return eps


def choose_dt(total_time):
    if total_time <= 0.5:
        return 0.0005
    elif total_time <= 5:
        return 0.005
    elif total_time <= 50:
        return 0.025
    elif total_time <= 500:
        return 0.1
    else:
        return 0.25


def run_single(stress_kpa, creep_sec, recovery_sec):
    """Run one creep (+recovery) simulation. Return dict of results."""
    stress_pa = stress_kpa * 1000.0
    total = creep_sec + recovery_sec
    dt = choose_dt(total)

    t = np.arange(0, total + dt, dt)
    sigma = np.where(t <= creep_sec, stress_pa, 0.0)
    eps = forward_simulate(t, sigma)

    idx_creep_end = np.searchsorted(t, creep_sec, side="right") - 1
    strain_creep = eps[idx_creep_end]
    strain_recovery = eps[-1] if recovery_sec > 0 else np.nan
    residual = strain_recovery - EPS0 if recovery_sec > 0 else np.nan

    return {
        "stress_kpa": stress_kpa,
        "duration_s": creep_sec,
        "recovery_s": recovery_sec,
        "g2": g2(stress_pa),
        "sigma_eff_Pa": g2(stress_pa) * stress_pa,
        "strain_creep_end": strain_creep,
        "strain_recovery_end": strain_recovery,
        "residual_strain": residual,
        "t": t,
        "eps": eps,
        "sigma": sigma,
    }


# ============================================================
# RUN THE SWEEP
# ============================================================
print("Running systematic sweep...")
print(f"  Stress levels: {STRESS_LEVELS} kPa")
print(f"  Time levels:   {TIME_LEVELS} s")
print(f"  Recovery:      {RECOVERY_SEC} s")
print(f"  Total runs:    {len(STRESS_LEVELS) * len(TIME_LEVELS)}")

results = []
strain_grid = np.zeros((len(STRESS_LEVELS), len(TIME_LEVELS)))
residual_grid = np.zeros((len(STRESS_LEVELS), len(TIME_LEVELS)))

for i, s_kpa in enumerate(STRESS_LEVELS):
    for j, t_sec in enumerate(TIME_LEVELS):
        res = run_single(s_kpa, t_sec, RECOVERY_SEC)
        results.append(res)
        strain_grid[i, j] = res["strain_creep_end"]
        residual_grid[i, j] = res["residual_strain"]
        print(f"  σ={s_kpa:>6.1f} kPa, t={t_sec:>6.1f} s  →  ε_creep={res['strain_creep_end']:.6f}  ε_residual={res['residual_strain']:.6f}")

# ============================================================
# SAVE CSV TABLE
# ============================================================
csv_path = "systematic_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Stress_kPa", "Duration_s", "Recovery_s",
        "g2", "Sigma_eff_Pa",
        "Strain_creep_end", "Strain_recovery_end", "Residual_strain"
    ])
    for r in results:
        writer.writerow([
            r["stress_kpa"], r["duration_s"], r["recovery_s"],
            f"{r['g2']:.6f}", f"{r['sigma_eff_Pa']:.2f}",
            f"{r['strain_creep_end']:.6f}",
            f"{r['strain_recovery_end']:.6f}" if not np.isnan(r["strain_recovery_end"]) else "",
            f"{r['residual_strain']:.6f}" if not np.isnan(r["residual_strain"]) else "",
        ])
print(f"\nResults saved to {csv_path}")

# ============================================================
# PLOT 1: Heatmap — strain at end of creep
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Creep strain heatmap
im1 = axes[0].imshow(strain_grid, aspect="auto", origin="lower", cmap="hot")
axes[0].set_xticks(range(len(TIME_LEVELS)))
axes[0].set_xticklabels([str(t) for t in TIME_LEVELS], rotation=45, fontsize=8)
axes[0].set_yticks(range(len(STRESS_LEVELS)))
axes[0].set_yticklabels([str(s) for s in STRESS_LEVELS], fontsize=8)
axes[0].set_xlabel("Creep Duration [s]")
axes[0].set_ylabel("Stress [kPa]")
axes[0].set_title("Strain at End of Creep")
plt.colorbar(im1, ax=axes[0], label="Strain [-]")

# Add text annotations
for i in range(len(STRESS_LEVELS)):
    for j in range(len(TIME_LEVELS)):
        val = strain_grid[i, j]
        color = "white" if val < (strain_grid.max() + strain_grid.min()) / 2 else "black"
        axes[0].text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=6, color=color)

# Residual strain heatmap
if RECOVERY_SEC > 0:
    im2 = axes[1].imshow(residual_grid, aspect="auto", origin="lower", cmap="YlOrRd")
    axes[1].set_xticks(range(len(TIME_LEVELS)))
    axes[1].set_xticklabels([str(t) for t in TIME_LEVELS], rotation=45, fontsize=8)
    axes[1].set_yticks(range(len(STRESS_LEVELS)))
    axes[1].set_yticklabels([str(s) for s in STRESS_LEVELS], fontsize=8)
    axes[1].set_xlabel("Creep Duration [s]")
    axes[1].set_ylabel("Stress [kPa]")
    axes[1].set_title(f"Residual Strain after {RECOVERY_SEC}s Recovery")
    plt.colorbar(im2, ax=axes[1], label="Residual Strain [-]")

    for i in range(len(STRESS_LEVELS)):
        for j in range(len(TIME_LEVELS)):
            val = residual_grid[i, j]
            color = "white" if val < (residual_grid.max() + residual_grid.min()) / 2 else "black"
            axes[1].text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=6, color=color)

plt.tight_layout()
plt.savefig("heatmaps.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# PLOT 2: Creep curves — fixed stress, varying time
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

selected_stresses = [1, 10, 40, 60, 85, 120]
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(TIME_LEVELS)))

for idx, s_kpa in enumerate(selected_stresses):
    ax = axes[idx]
    for j, t_sec in enumerate(TIME_LEVELS):
        # Find matching result
        for r in results:
            if r["stress_kpa"] == s_kpa and r["duration_s"] == t_sec:
                ax.plot(r["t"], r["eps"], color=colors[j], lw=1.5,
                        label=f"{t_sec}s" if idx == 0 else "")
                break

    ax.set_xlabel("Time [s]", fontsize=9)
    ax.set_ylabel("Strain [-]", fontsize=9)
    ax.set_title(f"σ = {s_kpa} kPa (g2={g2(s_kpa*1000):.4f})", fontsize=10)
    ax.grid(True, alpha=0.3)

# Add shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(TIME_LEVELS),
           fontsize=8, title="Creep duration", title_fontsize=9,
           bbox_to_anchor=(0.5, 1.02))

plt.suptitle("Creep + Recovery Curves at Different Stress Levels", fontsize=13,
             fontweight="bold", y=1.05)
plt.tight_layout()
plt.savefig("creep_families_by_stress.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# PLOT 3: Creep curves — fixed duration, varying stress
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

selected_times = [0.5, 5, 30, 60, 120, 300]
colors_s = plt.cm.plasma(np.linspace(0.1, 0.9, len(STRESS_LEVELS)))

for idx, t_sec in enumerate(selected_times):
    ax = axes[idx]
    for i, s_kpa in enumerate(STRESS_LEVELS):
        for r in results:
            if r["stress_kpa"] == s_kpa and r["duration_s"] == t_sec:
                ax.plot(r["t"], r["eps"], color=colors_s[i], lw=1.5,
                        label=f"{s_kpa} kPa" if idx == 0 else "")
                break

    ax.set_xlabel("Time [s]", fontsize=9)
    ax.set_ylabel("Strain [-]", fontsize=9)
    ax.set_title(f"Creep = {t_sec} s", fontsize=10)
    ax.grid(True, alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=6,
           fontsize=7, title="Stress level", title_fontsize=9,
           bbox_to_anchor=(0.5, 1.02))

plt.suptitle("Creep + Recovery Curves at Different Durations", fontsize=13,
             fontweight="bold", y=1.05)
plt.tight_layout()
plt.savefig("creep_families_by_time.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# PLOT 4: Nonlinearity summary
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# g2 vs stress
s_arr = np.linspace(0, 150, 300)
g2_arr = np.array([g2(s * 1000) for s in s_arr])
axes[0].plot(s_arr, g2_arr, "b-", lw=2)
for s_kpa in STRESS_LEVELS:
    axes[0].plot(s_kpa, g2(s_kpa * 1000), "ro", ms=5)
axes[0].set_xlabel("Stress [kPa]")
axes[0].set_ylabel("g2 [-]")
axes[0].set_title("g2(σ) at each test point")
axes[0].grid(True, alpha=0.3)

# sigma_eff vs stress
seff_arr = g2_arr * s_arr * 1000
axes[1].plot(s_arr, seff_arr, "r-", lw=2)
axes[1].plot(s_arr, s_arr * 1000, "k--", lw=1, alpha=0.4, label="Linear")
for s_kpa in STRESS_LEVELS:
    axes[1].plot(s_kpa, g2(s_kpa * 1000) * s_kpa * 1000, "ro", ms=5)
axes[1].set_xlabel("Applied Stress [kPa]")
axes[1].set_ylabel("Effective Stress [Pa]")
axes[1].set_title("σ_eff vs σ_applied")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Strain at end of creep vs stress (for longest duration)
longest_idx = -1  # last time level
strains_at_longest = strain_grid[:, longest_idx]
axes[2].plot(STRESS_LEVELS, strains_at_longest, "ro-", lw=2, ms=6)
# Linear extrapolation from lowest stress
linear_scale = np.array(STRESS_LEVELS) / STRESS_LEVELS[0] * strains_at_longest[0]
axes[2].plot(STRESS_LEVELS, linear_scale, "k--", lw=1, alpha=0.4, label="If linear")
axes[2].set_xlabel("Stress [kPa]")
axes[2].set_ylabel(f"Strain at t={TIME_LEVELS[-1]}s [-]")
axes[2].set_title("Strain vs Stress (saturation visible)")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("nonlinearity_summary.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# Print summary table
# ============================================================
print(f"\n{'='*80}")
print("STRAIN AT END OF CREEP (rows=stress, cols=duration)")
print(f"{'='*80}")
header = f"{'kPa':>8}" + "".join([f"{t:>10.1f}s" for t in TIME_LEVELS])
print(header)
print("-" * len(header))
for i, s in enumerate(STRESS_LEVELS):
    row = f"{s:>8.1f}" + "".join([f"{strain_grid[i,j]:>10.4f}" for j in range(len(TIME_LEVELS))])
    print(row)

if RECOVERY_SEC > 0:
    print(f"\n{'='*80}")
    print(f"RESIDUAL STRAIN AFTER {RECOVERY_SEC}s RECOVERY")
    print(f"{'='*80}")
    print(header)
    print("-" * len(header))
    for i, s in enumerate(STRESS_LEVELS):
        row = f"{s:>8.1f}" + "".join([f"{residual_grid[i,j]:>10.6f}" for j in range(len(TIME_LEVELS))])
        print(row)

print(f"\nAll plots saved. CSV saved to {csv_path}")
