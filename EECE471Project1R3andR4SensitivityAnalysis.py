import numpy as np
import matplotlib.pyplot as plt
import os

# Transfer mag function, metrics
def transfer_mag(R1, R2, R3, R4, C1, C2, omega):
    omega = np.asarray(omega, dtype=float)
    Av = 1 + (R4 / R3)
    hp = omega * R1 * C1
    lp = omega * R2 * C2
    return Av * (hp / np.sqrt(1 + hp**2)) * (1 / np.sqrt(1 + lp**2))

def metrics_from_curve(omega, T):
    T = np.asarray(T)
    omega = np.asarray(omega)

    idx_peak = np.argmax(T)
    T_max = T[idx_peak]
    omega0 = omega[idx_peak]

    target = T_max / np.sqrt(2)
    above = np.where(T >= target)[0]
    if len(above) == 0:
        return T_max, omega0, np.nan, np.nan, np.nan

    w1 = omega[above[0]]
    w2 = omega[above[-1]]
    bw = w2 - w1
    return T_max, omega0, w1, w2, bw

# Settings
base = dict(R1=10_000, R2=10_000, R3=10_000, R4=30_000, C1=0.01, C2=0.01)
omega = np.logspace(-6, 2, 3000)

tol = 0.10
N = 5000
rng = np.random.default_rng(0)



# Nominal
T_nom = transfer_mag(**base, omega=omega)
Tmax_nom, w0_nom, _, _, bw_nom = metrics_from_curve(omega, T_nom)
Av_nom = 1 + base["R4"] / base["R3"]


# Monte Carlo: vary only R3 and R4
R3_s = base["R3"] * (1 + rng.uniform(-tol, tol, N))
R4_s = base["R4"] * (1 + rng.uniform(-tol, tol, N))

Av = 1 + (R4_s / R3_s)

Tmax = np.empty(N)
w0   = np.empty(N)
bw   = np.empty(N)

for k in range(N):
    T = transfer_mag(base["R1"], base["R2"], R3_s[k], R4_s[k], base["C1"], base["C2"], omega)
    Tmax[k], w0[k], _, _, bw[k] = metrics_from_curve(omega, T)


#summarize
def summary(name, arr, nominal):
    arr = np.asarray(arr)
    print(f"{name}: nominal={nominal:.6g}  mean={arr.mean():.6g}  std={arr.std():.6g}  "
          f"min={arr.min():.6g}  max={arr.max():.6g}  (std/nom={100*arr.std()/nominal:.2f}%)")

print("\n=== R3/R4-only ±10% Monte Carlo Sensitivity ===")
summary("Av",   Av,   Av_nom)
summary("Tmax", Tmax, Tmax_nom)
summary("w0 (rad/s)", w0, w0_nom)
summary("BW (rad/s)", bw, bw_nom)


# Plots
plt.figure()
plt.hist(Av, bins=50)
plt.axvline(Av_nom, linestyle="--")
plt.xlabel("Av = 1 + R4/R3")
plt.ylabel("Count")
plt.title("Monte Carlo (±10%): Av distribution (R3/R4 only)")
plt.grid(True, which="both")
plt.savefig("montecarlo_R3R4_Av_hist.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure()
plt.hist(Tmax, bins=50)
plt.axvline(Tmax_nom, linestyle="--")
plt.xlabel("Tmax")
plt.ylabel("Count")
plt.title("Monte Carlo (±10%): Tmax distribution (R3/R4 only)")
plt.grid(True, which="both")
plt.savefig("montecarlo_R3R4_Tmax_hist.png", dpi=300, bbox_inches="tight")
plt.show()