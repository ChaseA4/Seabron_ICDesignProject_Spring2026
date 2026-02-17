'''
1. solve general transfer function of each part of the design, 
then cascade to find total transfer function

 parameters are C1, C2, R1, R2, R3, R4

 High pass filter stage:
 T = R1/(R1 + ZC1)

 Low pass filter stage:
 T = ZC2/(R2+ZC2)

 Non-inverting amplifier:
 Av = 1 + (R4/R3)

 Total transfer function Ttotal:
 Ttotal = (R1/(R1+ZC1))*(1+(R4/R3))*(ZC2/(R2+ZC2))
'''


'''
2. Choose R3 and R4 Values such that Av > 3.

If Av = 1 + (R4/R3), then (R4/R3) > 2, which implies either 
R4 > 2R3 or R3 < R4/2

Assume R3 = 50K, then R4 > 100K, so then R4 could range from 100K
to 1000K by order of 100K?

'''
import matplotlib.pyplot as plt
import numpy as np


def transfer_mag(R1, R2, R3, R4, C1, C2, omega):
    R3 = np.asarray(R3, dtype=float)
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

    # -3 dB point = magnitude / sqrt(2)
    target = T_max / np.sqrt(2)

    # Find indices where curve is above target
    above = np.where(T >= target)[0]
    if len(above) == 0:
        return T_max, omega0, np.nan, np.nan, np.nan

    w_low = omega[above[0]]
    w_high = omega[above[-1]]
    bw = w_high - w_low

    return T_max, omega0, w_low, w_high, bw


# Fixed values (example)
R1 = R2 = 10_000
R3 = 10_000
R4 = 30_000   # Av = 4 (>3)
C1 = C2 = 0.01

omega = np.logspace(-6, 2, 1000)  # rad/s sweep
T = transfer_mag(R1, R2, R3, R4, C1, C2, omega)

T_max, omega0, w1, w2, bw = metrics_from_curve(omega, T)
print("Tmax =", T_max)
print("omega0 =", omega0)
print("w1, w2 =", w1, w2)
print("bandwidth =", bw)


plt.figure()
plt.semilogx(omega, 20*np.log10(T))
plt.xlabel("ω (rad/s)")
plt.ylabel("|T(jω)| (dB)")
plt.grid(True, which="both")
plt.show()
