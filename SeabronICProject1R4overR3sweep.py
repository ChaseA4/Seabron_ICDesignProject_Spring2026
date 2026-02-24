import numpy as np
import matplotlib.pyplot as plt
import os

# magnitude only
def transfer_mag(R1, R2, R3, R4, C1, C2, omega):
    omega = np.asarray(omega, dtype=float)
    Av = 1 + (R4 / R3)                 # non-inverting gain
    hp = omega * R1 * C1               # ωR1C1
    lp = omega * R2 * C2               # ωR2C2
    return Av * (hp / np.sqrt(1 + hp**2)) * (1 / np.sqrt(1 + lp**2))


def bode_plot_sweep(omega, curves, title, filename, legend_loc="best"):
    """
    curves: list of tuples (label, T_mag_array)
    filename: base filename (no extension)
    """

    plt.figure()
    for label, T in curves:
        plt.semilogx(omega, 20*np.log10(np.maximum(T, 1e-300)), label=label)

    plt.xlabel("ω (rad/s)")
    plt.ylabel("|T(jω)| (dB)")
    plt.grid(True, which="both")
    plt.title(title)
    plt.legend(loc=legend_loc)

    # Save files
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")

    plt.show()



# Baseline parameters (edit as needed)
base = dict(
    R1=10_000,
    R2=10_000,
    R3=10_000,
    R4=30_000,     # Av = 4
    C1=0.01,
    C2=0.01
)

omega = np.logspace(-6, 2, 2000)

# ============================================================
# (A) "Constraint-satisfying" plot: keep Av > 3
#     Approach: choose several Av values all > 3, and for each,
#     set R4 = (Av - 1) * R3_base.
# ============================================================
Av_targets = [3.2, 4.0, 6.0, 10.0]  # all > 3 (meets requirement)

curves_A = []
for Av in Av_targets:
    R3 = base["R3"]
    R4 = (Av - 1.0) * R3  # ensures Av = 1 + R4/R3 exactly
    T = transfer_mag(base["R1"], base["R2"], R3, R4, base["C1"], base["C2"], omega)
    curves_A.append((f"Av={Av:g} (R4={(Av-1):g}·R3)", T))

bode_plot_sweep(
    omega,
    curves_A,
    title="Frequency Response (Gain Constraint Satisfied: Av > 3)",
    filename="gain_constraint_held"
)


# (B) Sweep R3 with R4 held constant (gain may violate threshold)
R3_vals = np.logspace(3, 6, 6)   # 1k to 1M
R4_const = base["R4"]            # hold R4 fixed

curves_B = []
for R3 in R3_vals:
    Av = 1 + R4_const / R3
    T = transfer_mag(base["R1"], base["R2"], R3, R4_const, base["C1"], base["C2"], omega)
    curves_B.append((f"R3={R3:g}Ω (Av={Av:.2f})", T))

bode_plot_sweep(
    omega,
    curves_B,
    title=f"Frequency Response Sweeping R3 (R4 fixed at {R4_const:g} Ω)",
    filename="sweep_R3_fixed_R4"
)

# ============================================================
# (C) Sweep R4 with R3 held constant (gain may violate threshold)
R4_vals = np.logspace(3, 6, 6)   # 1k to 1M
R3_const = base["R3"]            # hold R3 fixed

curves_C = []
for R4 in R4_vals:
    Av = 1 + R4 / R3_const
    T = transfer_mag(base["R1"], base["R2"], R3_const, R4, base["C1"], base["C2"], omega)
    curves_C.append((f"R4={R4:g}Ω (Av={Av:.2f})", T))

bode_plot_sweep(
    omega,
    curves_C,
    title=f"Frequency Response Sweeping R4 (R3 fixed at {R3_const:g} Ω)",
    filename="sweep_R4_fixed_R3"
)




