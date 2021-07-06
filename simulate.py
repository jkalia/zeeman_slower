# Jasmine Kalia
# June 25th, 2021
# propagation.py
# Propagate some atoms through the magnetic field created by 
# zeeman_slower_configuration.py.

import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
import scipy.integrate

import ideal_field as ideal
import coil_configuration as coil 


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def acceleration(m, linewidth, k, mu0, s, laser_detuning, v, B):
    delta = (laser_detuning * 2 * np.pi + k * v + mu0 * B / ideal.hbar)
    # delta = 0
    return (ideal.hbar * k / m * linewidth / 2 
            * s / (1 + s + (2 * delta / linewidth)**2))
 

# Simulates the motion of the atoms in the B field 
# TODO: doesn't work for the ideal field, which it definitely should 
def simulate_atom(atom, intensity, v_initial, coil_winding=[0], current_for_coils=[0], 
                  dt=1e-9, z_max=1, max_steps=2000000, optimized=True):

    # Choose atom to simulate
    if atom=="Er":
        m = ideal.m_er
        linewidth = ideal.linewidth_er
        k = ideal.k_er
        mu0 = ideal.mu0_er
        Isat = ideal.Isat_er
        laser_detuning = ideal.laser_detuning_er
        # v_initial = ideal.initial_velocity_er
        v_final = ideal.final_velocity_er
    else:
        m = ideal.m_li
        linewidth = ideal.linewidth_li
        k = ideal.k_li
        mu0 = ideal.mu0_li
        Isat = ideal.Isat_li_d2
        laser_detuning = ideal.laser_detuning_li
        # v_initial = ideal.initial_velocity_li
        v_final = ideal.final_velocity_li

    s = intensity / Isat

    ts = np.arange(0, max_steps*dt, dt)
    zs = np.zeros(max_steps)
    vs = np.zeros(max_steps)
    acs = np.zeros(max_steps)

    v = v_initial
    z = 0.0005
    counter = 0

    while (v >= v_final) and (z <= z_max) and (counter < max_steps):

        if optimized:
            a = acceleration(m, linewidth, k, mu0, s, laser_detuning, v, 
                             (coil.calculate_B_field_coil(coil_winding, 
                                                          current_for_coils, 
                                                          np.array([z]))[0] 
                                                          * 10**(-4)))
        else:
            a = acceleration(m, linewidth, k, mu0, s, laser_detuning, v, 
                             (ideal.get_ideal_B_field(ideal.ideal_B_field, 
                                                      np.array([z]))[0] 
                                                      * 10**(-4)))

        z += v * dt + 0.5 * a * dt**2
        v -= a * dt 
        
        acs[counter] = a
        vs[counter] = v 
        zs[counter] = z

        counter += 1

    print("v_final = {}, z = {}, counter = {}".format(v, z, counter))

    return ts[0:counter], zs[0:counter], vs[0:counter], acs[0:counter]




