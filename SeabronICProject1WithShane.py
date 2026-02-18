import numpy as np
import matplotlib.pyplot as plt

def transfer_mag(R1, R2, R3, R4, C1, C2, omega):
    omega = np.asarray(omega, dtype=float)
    gain = 1 + (R4 / R3)
    hp = omega * R1 * C1
    lp = omega * R2 * C2
    return gain * (hp / np.sqrt(1 + hp**2)) * (1 / np.sqrt(1 + lp**2))

def metrics_from_curve(omega, T):
    T = np.asarray(T)
    omega = np.asarray(omega)

    idx_peak = np.argmax(T)
    T_max = T[idx_peak]
    omega0 = omega[idx_peak]

    target = T_max / np.sqrt(2)  # -3 dB in magnitude
    above = np.where(T >= target)[0]
    if len(above) == 0:
        return T_max, omega0, np.nan, np.nan, np.nan

    w_low = omega[above[0]]
    w_high = omega[above[-1]]
    bw = w_high - w_low
    return T_max, omega0, w_low, w_high, bw

# Nominal values
base = dict(R1=10_000, R2=10_000, R3=10_000, R4=30_000, C1=0.01, C2=0.01)
omega = np.logspace(-6, 2, 2000)

# Sweep R1
R1_vals = [1_000, 5_000, 10_000, 50_000, 100_000]

plt.figure()
for R1 in R1_vals:
    T = transfer_mag(R1, base["R2"], base["R3"], base["R4"], base["C1"], base["C2"], omega)
    plt.semilogx(omega, 20*np.log10(T), label=f"R1={R1:g} Ω")

plt.xlabel("ω (rad/s)")
plt.ylabel("|T(jω)| (dB)")
plt.grid(True, which="both")
plt.legend()
plt.title("Frequency Response Sweeping R1")
plt.savefig("R1_sweep_frequency_response.png", dpi=300, bbox_inches="tight")
print("R1 sweep metrics:")
for R1 in R1_vals:
    T = transfer_mag(R1, base["R2"], base["R3"], base["R4"], base["C1"], base["C2"], omega)
    T_max, omega0, w1, w2, bw = metrics_from_curve(omega, T)
    print(f"R1={R1:8g} Ω | Tmax={T_max:.4f}  omega0={omega0:.4e}  BW={bw:.4e}")


plt.show()


