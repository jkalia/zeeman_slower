# Jasmine Kalia
# July 2nd, 2021
# propagation.py
# Propagate some atoms through the magnetic field created by 
# zeeman_slower_configuration.py.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.constants 
from scipy.integrate import odeint

import zeeman_slower_configuration as zs 
import ideal_field as ideal
import coil_configuration as coil 


# Parameters from zeeman_slower_configuration.py to use for propagating atoms
densities = [7, 6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1.25, 1, 0.5, 1, 
             0.5, 0.25, 0]
fixed_densities = [2]
fixed_lengths = [6]
fixed_overlap = 0
final = [-7.18428594e+00, -2.85549832e-06, -9.70206319e-07,  5.69787469e-07,
         -6.03636938e-07,  6.99444598e+00,  8.14020624e+00,  7.59039476e+00,
          9.51184227e+00,  1.04877096e+01, -1.19425779e+01, -1.03911870e+01,
         -5.35495084e+00,  8.84634912e+00, -2.46417535e+00,  2.51996538e+00,
          9.14485245e+00, -7.18603523e+00,  1.20000000e+01,  2.98967317e+01,
          1.28602604e+02]

discretized_slower_adjusted, ideal_B_field_adjusted, z_long, num_coils = \
    zs.discretize(fixed_lengths, fixed_overlap)

coil_winding, current_for_coils = \
  coil.give_coil_winding_and_current(num_coils, fixed_densities, densities, 
                                     fixed_lengths, np.round(final[0:-2]), 
                                     final[-2], final[-1])

s = 2


def calc_B_field(z):
    return (coil.calculate_B_field_coil(coil_winding, current_for_coils, 
                                        np.array([z]))[0] * 10**(-4))


def ideal_B_field(z):
    return ideal.get_ideal_B_field(ideal.ideal_B_field, 
                                   np.array([z]))[0] * 10**(-4)


def calc_w0(z):
    return ideal.omega_li + calc_B_field(z) * ideal.mu0_li / ideal.hbar


def ideal_w0(z):
    return ideal.omega_li + ideal_B_field(z) * ideal.mu0_li / ideal.hbar


def w(v):
    return ideal.laser_detuning_li + ideal.k_li * v


def calc_a_zs(z, v):
    return (-1 
            * (ideal.hbar * ideal.k_li / ideal.m_li * ideal.linewidth_li / 2) 
            * s 
            / (1 + s + (2 * (w(v) - calc_w0(z)) / ideal.linewidth_li)**2))


def ideal_a_zs(z, v):
    return (-1 
            * (ideal.hbar * ideal.k_li / ideal.m_li * ideal.linewidth_li / 2) 
            * s 
            / (1 + s + (2 * (w(v) - ideal_w0(z)) / ideal.linewidth_li)**2))


# V[0] = z
# V[1] = v 
def calc_differential(V, t):
    return [V[1], calc_a_zs(V[0], V[1])]


# V[0] = z
# V[1] = v 
def ideal_differential(V, t):
    return [V[1], ideal_a_zs(V[0], V[1])]



#initial conditions
t0=0
T=1
num=int(1e6)
t = np.linspace(t0, T, num)
V0 = [0,ideal.initial_velocity_li*.8]

#Solve
calc_V = odeint(calc_differential, V0, t)
calc_z = calc_V[:,0]
calc_v = calc_V[:,1]

ideal_V = odeint(ideal_differential, V0, t)
ideal_z = ideal_V[:,0]
ideal_v = ideal_V[:,1]


fig, ax = plt.subplots()

ax.plot(ideal_z, ideal_v, label="ideal")
ax.plot(calc_z, calc_v, label="calculated")
ax.set_xlabel("Position (m)")
ax.set_ylabel("Velocity (m/s)")
ax.set_title("Motion of Li atom in the Slower")
ax.legend()

plt.show()

