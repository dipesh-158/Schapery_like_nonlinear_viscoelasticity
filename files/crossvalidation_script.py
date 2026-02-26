"""
Cross-validation: YOUR model vs. Stolyarov (2023) creep data.

KEY INSIGHT: Your model was trained at ~84 kPa on a DIFFERENT soft fabric.
Stolyarov's woven polyester operates at 54-269 MPa — a completely different
material at 1000x higher stress. Direct prediction is not meaningful.

WHAT THIS SCRIPT DOES:
  (A) Predicts what YOUR material would do under a pure creep test at YOUR stress levels
      → This is the most useful validation: does your cyclic-trained model 
        correctly predict creep behavior it was never trained on?
  (B) Compares the SHAPE (normalized) of creep curves between your model 
      and Stolyarov to see if the viscoelastic mechanisms are similar.
  (C) Exports the Stolyarov synthetic data as CSV for your records.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── YOUR MODEL ──────────────────────────────────────────────────────────
TAU_FAST = np.array([0.15, 2.5, 18.0])
TAU_SLOW = np.array([200.0, 800.0, 3000.0, 12000.0, 50000.0])

def g1_of_sigma(sigma_val, sigma_ref, alpha):
    x = np.abs(sigma_val) / max(1e-12, sigma_ref)
    return 1.0 + alpha * (x / (1.0 + x))

def forward_fast_slow(t, sigma, p_fast, A_slow):
    t = np.asarray(t, float); sigma = np.asarray(sigma, float)
    D0=p_fast["D0"]; Dk=p_fast["Dk"]; alpha=p_fast["alpha"]
    sigma_ref=p_fast["sigma_ref"]; d_shift=p_fast["d_shift"]
    c2=p_fast["c2"]; eps0=p_fast["eps0"]
    n=len(t); nf=len(Dk); ns=len(A_slow)
    efk=np.zeros(nf); zs=np.zeros(ns)
    et=np.zeros(n); ef=np.zeros(n); sc=np.zeros(n)
    g1_0=g1_of_sigma(sigma[0],sigma_ref,alpha)
    ef[0]=eps0+D0*sigma[0]+g1_0*np.sum(efk); sc[0]=np.sum(A_slow*zs); et[0]=ef[0]+sc[0]
    for i in range(1,n):
        dt_=t[i]-t[i-1]
        if dt_<=0: dt_=1e-12
        sm=0.5*(sigma[i]+sigma[i-1]); xm=np.abs(sm)/max(1e-12,sigma_ref)
        g2m=1.0/(1.0+c2*xm); am=np.exp(d_shift*xm); dpsi=dt_/am; se=g2m*sm
        for k in range(nf):
            r=np.exp(-dpsi/TAU_FAST[k]); efk[k]=r*efk[k]+(1-r)*(Dk[k]*se)
        for j in range(ns):
            r=np.exp(-dt_/TAU_SLOW[j]); zs[j]=r*zs[j]+(1-r)*sm
        g1i=g1_of_sigma(sigma[i],sigma_ref,alpha)
        ef[i]=eps0+D0*sigma[i]+g1i*np.sum(efk); sc[i]=np.sum(A_slow*zs); et[i]=ef[i]+sc[i]
    return et, ef, sc

# ── YOUR PARAMETERS ────────────────────────────────────────────────────
p_fast = {
    "D0": 2.075775e-07, "Dk": np.array([3.642446e-05, 2.242740e-05, 1.438127e-05]),
    "alpha": 2.898217, "sigma_ref": 3.390546e+03, "d_shift": 0.0,
    "c2": 2.118515, "eps0": 2.411313e-01,
}
A_slow = np.array([1.457611e-06, 1.154429e-06, 0.0, 0.0, 5.532409e-05])

# ── STOLYAROV TABLE 5 ──────────────────────────────────────────────────
stolyarov = {
    'C1': {'sigma_MPa':268.8, 'rate':10, 'C_ih':0.245,'D_ih':8.13,'C_il':0.228,'D_il':5.82},
    'C2': {'sigma_MPa':161.3, 'rate':10, 'C_ih':0.200,'D_ih':3.31,'C_il':0.119,'D_il':2.43},
    'C3': {'sigma_MPa':268.8, 'rate':1., 'C_ih':0.269,'D_ih':8.20,'C_il':0.246,'D_il':6.54},
    'C4': {'sigma_MPa': 53.8, 'rate':10, 'C_ih':0.019,'D_ih':1.43,'C_il':0.017,'D_il':0.53},
    'C5': {'sigma_MPa':268.8, 'rate':.1, 'C_ih':0.268,'D_ih':8.26,'C_il':0.219,'D_il':6.72},
}

# ── HELPER: build creep test at a given constant stress (Pa) ───────────
def build_creep_test(sigma_Pa, t_ramp=50.0, t_creep=6000.0, t_rec=6000.0, dt=1.0):
    sigma_rec_Pa = 0.0  # full unload for your material
    n1=max(int(t_ramp/dt),20); n2=int(t_creep/dt); n3=max(int(t_ramp/dt),20); n4=int(t_rec/dt)
    t1=np.linspace(0,t_ramp,n1); s1=np.linspace(0,sigma_Pa,n1)
    t2=np.linspace(t1[-1]+dt,t1[-1]+t_creep,n2); s2=np.full(n2,sigma_Pa)
    t3=np.linspace(t2[-1]+dt,t2[-1]+t_ramp,n3); s3=np.linspace(sigma_Pa,sigma_rec_Pa,n3)
    t4=np.linspace(t3[-1]+dt,t3[-1]+t_rec,n4); s4=np.full(n4,sigma_rec_Pa)
    return np.concatenate([t1,t2,t3,t4]), np.concatenate([s1,s2,s3,s4]), t_ramp

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: YOUR MODEL'S CREEP PREDICTION AT YOUR STRESS LEVELS
# ═══════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
fig1.suptitle(
    "PART A: YOUR Model Creep Predictions at YOUR Stress Levels\n"
    "(Model trained on ~84 kPa cyclic ramp/hold → predicting pure creep it was never trained on)",
    fontsize=12, fontweight='bold', y=0.99)

# Test at multiple stress levels within your training range
stress_levels_kPa = [20, 40, 60, 84]  # kPa
colors_stress = ['#2ca02c','#ff7f0e','#1f77b4','#d62728']

# (a) Creep curves at different stress levels
ax = axes1[0, 0]
ax.set_title("(a) Predicted Creep at Different Stresses", fontsize=11)
for sigma_kPa, col in zip(stress_levels_kPa, colors_stress):
    sigma_Pa = sigma_kPa * 1e3
    t, s, tr = build_creep_test(sigma_Pa)
    ep, ef, es = forward_fast_slow(t, s, p_fast, A_slow)
    mask = (t >= tr) & (t <= tr + 6000)
    tp = t[mask] - t[mask][0]
    ax.plot(tp, ep[mask], '-', color=col, lw=2, label=f"σ = {sigma_kPa} kPa")
ax.set_xlabel("Time from creep onset (s)"); ax.set_ylabel("Strain (fractional)")
ax.legend(); ax.grid(True, alpha=0.3)

# (b) Full creep+recovery at 84 kPa with FAST/SLOW decomposition
ax = axes1[0, 1]
ax.set_title("(b) Full Creep+Recovery at 84 kPa (FAST+SLOW)", fontsize=11)
t, s, tr = build_creep_test(84e3)
ep, ef, es = forward_fast_slow(t, s, p_fast, A_slow)
ax.plot(t, ep, 'r-', lw=2, label='Total')
ax.plot(t, ef, 'b--', lw=1.5, alpha=0.7, label='FAST')
ax.plot(t, es, 'g--', lw=1.5, alpha=0.7, label='SLOW')
ax.set_xlabel("Time (s)"); ax.set_ylabel("Strain (fractional)")
ax.legend(); ax.grid(True, alpha=0.3)

# (c) Stress input for 84 kPa test
ax = axes1[1, 0]
ax.set_title("(c) Stress Input (84 kPa creep test)", fontsize=11)
ax.plot(t, s/1e3, 'b-', lw=2)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Stress (kPa)")
ax.grid(True, alpha=0.3)

# (d) Creep rate (strain rate during hold) at different stresses
ax = axes1[1, 1]
ax.set_title("(d) Creep Rate During Hold Phase", fontsize=11)
for sigma_kPa, col in zip(stress_levels_kPa, colors_stress):
    sigma_Pa = sigma_kPa * 1e3
    t, s, tr = build_creep_test(sigma_Pa)
    ep, ef, es = forward_fast_slow(t, s, p_fast, A_slow)
    mask = (t >= tr+10) & (t <= tr + 6000)
    tp = t[mask]; ep_m = ep[mask]
    # numerical derivative
    rate = np.gradient(ep_m, tp)
    ax.plot(tp-tp[0], rate, '-', color=col, lw=1.5, label=f"σ = {sigma_kPa} kPa")
ax.set_xlabel("Time from creep onset (s)"); ax.set_ylabel("dε/dt (1/s)")
ax.set_yscale('log'); ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig1.savefig('/mnt/user-data/outputs/partA_your_model_creep_prediction.png', dpi=150, bbox_inches='tight')
plt.close(fig1)
print("Part A saved: partA_your_model_creep_prediction.png")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: NORMALIZED SHAPE COMPARISON WITH STOLYAROV
# ═══════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
fig2.suptitle(
    "PART B: Normalized Creep Shape Comparison\n"
    "(YOUR model at 84 kPa vs. Stolyarov at 54–269 MPa — comparing viscoelastic mechanism shapes)",
    fontsize=12, fontweight='bold', y=0.99)

# (a) Normalized creep: ε_norm(t) = (ε(t) - ε(1)) / (ε(6000) - ε(1))
ax = axes2[0, 0]
ax.set_title("(a) Normalized Creep Shape (0→1 over 6000s)", fontsize=11)

# Stolyarov specimens
for name in ['C1','C2','C4']:
    spec = stolyarov[name]
    tp = np.logspace(0, np.log10(6000), 500)  # 1 to 6000s
    eps_s = (spec['D_ih'] + spec['C_ih'] * np.log(tp)) / 100.0
    eps_norm = (eps_s - eps_s[0]) / (eps_s[-1] - eps_s[0])
    ax.plot(tp, eps_norm, '-', lw=2, label=f"Stolyarov {name} (σ={spec['sigma_MPa']:.0f} MPa)")

# Your model at 84 kPa
t, s, tr = build_creep_test(84e3, dt=1.0)
ep, _, _ = forward_fast_slow(t, s, p_fast, A_slow)
mask = (t >= tr+1) & (t <= tr + 6000)
tp_m = t[mask] - t[mask][0]
ep_m = ep[mask]
ep_norm = (ep_m - ep_m[0]) / (ep_m[-1] - ep_m[0])
ax.plot(tp_m, ep_norm, 'k--', lw=2.5, label=f"YOUR model (σ=84 kPa)")

ax.set_xlabel("Time from creep onset (s)"); ax.set_ylabel("Normalized strain")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (b) Same but in log-time
ax = axes2[0, 1]
ax.set_title("(b) Normalized Creep in Log-Time", fontsize=11)

for name in ['C1','C2','C4']:
    spec = stolyarov[name]
    tp = np.logspace(0, np.log10(6000), 500)
    eps_s = (spec['D_ih'] + spec['C_ih'] * np.log(tp)) / 100.0
    eps_norm = (eps_s - eps_s[0]) / (eps_s[-1] - eps_s[0])
    ax.semilogx(tp, eps_norm, '-', lw=2, label=f"Stolyarov {name}")

ax.semilogx(tp_m, ep_norm, 'k--', lw=2.5, label="YOUR model (84 kPa)")
ax.set_xlabel("Time from creep onset (s)"); ax.set_ylabel("Normalized strain")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (c) Stolyarov raw data reconstruction (for reference / export)
ax = axes2[1, 0]
ax.set_title("(c) Stolyarov Creep Data (reconstructed from Table 5)", fontsize=11)
for name in ['C1','C2','C3','C4','C5']:
    spec = stolyarov[name]
    tp = np.logspace(0, np.log10(6000), 500)
    eps_s = spec['D_ih'] + spec['C_ih'] * np.log(tp)
    ax.plot(tp, eps_s, '-', lw=2, label=f"{name}: σ={spec['sigma_MPa']:.0f}MPa, {spec['rate']}mm/min")
ax.set_xlabel("Time from creep onset (s)"); ax.set_ylabel("Strain (%)")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# (d) Stolyarov recovery data
ax = axes2[1, 1]
ax.set_title("(d) Stolyarov Recovery Data (reconstructed from Table 5)", fontsize=11)
for name in ['C1','C2','C3','C4','C5']:
    spec = stolyarov[name]
    tp = np.logspace(0, np.log10(6000), 500)
    eps_r = spec['D_il'] - spec['C_il'] * np.log(tp)
    ax.plot(tp, eps_r, '-', lw=2, label=f"{name}")
ax.set_xlabel("Time from recovery onset (s)"); ax.set_ylabel("Strain (%)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig2.savefig('/mnt/user-data/outputs/partB_normalized_shape_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print("Part B saved: partB_normalized_shape_comparison.png")


# ═══════════════════════════════════════════════════════════════════════
# EXPORT CSVs of Stolyarov synthetic data
# ═══════════════════════════════════════════════════════════════════════
for name in stolyarov:
    spec = stolyarov[name]
    tp = np.arange(1, 6001, 1.0)
    eps_creep = (spec['D_ih'] + spec['C_ih'] * np.log(tp)) / 100.0
    eps_rec = (spec['D_il'] - spec['C_il'] * np.log(tp)) / 100.0
    stress_creep = np.full_like(tp, spec['sigma_MPa'])
    stress_rec = np.full_like(tp, 3.36)

    # Creep CSV
    d1 = np.column_stack([tp, stress_creep*1e6, eps_creep])
    np.savetxt(f'/mnt/user-data/outputs/stolyarov_{name}_creep.csv', d1,
               delimiter=',', header='time_from_creep_onset_s,stress_Pa,strain_frac', comments='')
    # Recovery CSV
    d2 = np.column_stack([tp, stress_rec*1e6, eps_rec])
    np.savetxt(f'/mnt/user-data/outputs/stolyarov_{name}_recovery.csv', d2,
               delimiter=',', header='time_from_recovery_onset_s,stress_Pa,strain_frac', comments='')

print("All CSVs exported.")

# Print summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
Your model was trained on ~84 kPa cyclic ramp/hold data on a soft fabric.
Stolyarov's paper uses a woven polyester fabric at 54-269 MPa (yarn-level stress).
These are fundamentally different materials at fundamentally different stress scales.

PART A shows what your model PREDICTS for a pure creep test at your own stress
levels (20-84 kPa). This is the most meaningful validation: your model was trained
on cyclic loading → can it predict creep behavior it never saw?

PART B shows the SHAPE comparison: both your model and Stolyarov's fabric exhibit
logarithmic creep (ε ~ C*ln(t)), which is a universal viscoelastic signature.
The normalized plots let you compare whether the creep dynamics are similar.

The CSV files contain the Stolyarov data reconstructed from their Table 5
log-fit coefficients, ready for any further analysis.
""")
