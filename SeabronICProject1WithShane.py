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


import math
import numpy as np

def transfer_function(R1, R2, R3, R4, C1, C2, omega):
    gain = 1 + (R4 / R3)
    hp = (omega * R1 * C1)
    lp = (omega * R2 * C2)
    
    magnitude = gain * (hp / math.sqrt(1 + hp**2)) * (1 / math.sqrt(1 + lp**2))
    return magnitude


# Fixed component values
R1 = R2 = 10_000
R4 = 10_000
C1 = C2 = 0.01
omega = 0.01  # rad/s

# Sweep R3 (log-spaced is usually best for resistors)
R3_sweep = np.logspace(3, 6, 25)  # 1k to 1M, 25 points

T_vals = transfer_function(R1, R2, R3_sweep, R4, C1, C2, omega)

print(T_vals)

