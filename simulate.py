# Jasmine Kalia
# June 25th, 2021
# propagation.py
# Propagate atoms through the magnetic field created in 
# zeeman_slower_configuration.py.

import numpy as np 

import ideal_field as ideal
import coil_configuration as coil 


def acceleration(m, linewidth, k, mu0, s, laser_detuning, v, B):
    delta = (laser_detuning * 2 * np.pi + k * v + mu0 * B / ideal.hbar)
    return (ideal.hbar * k / m * linewidth / 2 
            * s / (1 + s + (2 * delta / linewidth)**2))

def saturation(s_init, z):
    waist_init = 0.00711 #m
    waist_size = 0.0045 + 0.00201544 * z
    return s_init * waist_init**2 / waist_size**2
 

# Simulates the motion of the atoms in the B field 
def simulate_atom(atom, s_init, v_initial, laser_detuning, coil_winding=[0], 
                  current_for_coils=[0], positions=[0], data=[0], dt=1e-7, 
                  z_max=1, max_steps=200000, optimized=True, observed=False, 
                  full_output=True):

    v = v_initial
    z = -0.05
    counter = 0

    if full_output:

        ts = np.arange(0, max_steps*dt, dt)
        zs = np.zeros(max_steps)
        vs = np.zeros(max_steps)
        acs = np.zeros(max_steps)

        while (z <= z_max) and (counter < max_steps):

            if optimized:
                a = acceleration(atom.m, atom.linewidth, atom.k, atom.mu0, s, 
                                  laser_detuning, v, 
                                  (coil.calculate_B_field_coil(coil_winding, 
                                                              current_for_coils, 
                                                              np.array([z]))[0] 
                                                              * 10**(-4)))
            elif observed:
                a = acceleration(atom.m, atom.linewidth, atom.k, atom.mu0, s, 
                                  laser_detuning, v, 
                                  np.interp(z, positions, data))
            else:
                a = acceleration(atom.m, atom.linewidth, atom.k, atom.mu0, s, 
                                  laser_detuning, v, 
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

    else:

        while (z <= z_max) and (counter < max_steps):

            if optimized:
                a = acceleration(atom.m, atom.linewidth, atom.k, atom.mu0, s, 
                                  laser_detuning, v, 
                                  (coil.calculate_B_field_coil(coil_winding, 
                                                              current_for_coils, 
                                                              np.array([z]))[0] 
                                                              * 10**(-4)))
            elif observed:
                a = acceleration(atom.m, atom.linewidth, atom.k, atom.mu0, s, 
                                  laser_detuning, v, 
                                  np.interp(z, positions, data))
            else:
                a = acceleration(atom.m, atom.linewidth, atom.k, atom.mu0, s, 
                                  laser_detuning, v, 
                                  (ideal.get_ideal_B_field(ideal.ideal_B_field, 
                                                          np.array([z]))[0] 
                                                          * 10**(-4)))

            z += v * dt + 0.5 * a * dt**2
            v -= a * dt 

            counter += 1
            
        print("v_final = {}, z = {}, counter = {}".format(v, z, counter))

        return v

