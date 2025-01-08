# Jasmine Kalia
# June 17th, 2021
# solenoid_configuration.py
# Module for zeeman_slower_configuration
# Contains all functions and variables relevant to the solenoid configuration.


import numpy as np
import scipy.constants

import parameters


# B field of solenoid as a function of z
# coil_density is the number of turns per unit length
def B_z_solenoid(z, coil_density, current, radius, z1, z2):
    return ((scipy.constants.mu_0 * coil_density * current / 2) 
            * ((z - z1) / (np.sqrt((z - z1)**2 + radius**2)) 
             - (z - z2) / (np.sqrt((z - z2)**2 + radius**2))))


# Helper function for calculate_B_field_solenoid
# Calculates B field for solenoid with arbitrary thickness and possible 
# half winding 
def calculate_B_field_solenoid_helper(z, coil_density, current, winding_num, 
                                      z1, z2, partial_winding):
    total_B_field = 0
    for i in range(winding_num):
        radius = (i * parameters.wire_height + parameters.wire_height / 2 
                  + parameters.slower_diameter / 2)

        if i == (winding_num - 1):
            current = current * partial_winding

        total_B_field += B_z_solenoid(z, coil_density, current, radius, z1, z2)
    
    return total_B_field


# This function calculates the total B field from an array. The array tells us 
# the solenoids we use to represent the coil winding. The free
# parameters are the end position of the solenoid and its length.  
# We determine the configuration using the array
# lengths = [l0, l1, ...] and densities = [d0, d1, ...], where l0 indicates the 
# the length of the last solenoid with thickness (or number of coils) d0.
# By giving only a physical densities array, we are automatically 
# limited to physically valid coil configurations. 
# We also fix the location of the high current section so that the length of 
# the slower is fixed.
def calculate_B_field_solenoid(z, num_coils, fixed_densities, densities, 
                               fixed_lengths, lengths, 
                               low_current, high_current):

    total_B_field = 0
    end_position = num_coils * parameters.wire_width
    start_position = end_position
    high_current_check = len(fixed_lengths)
    total_lengths = np.concatenate((fixed_lengths, lengths))
    total_densities = np.concatenate((fixed_densities, densities))

    for i, length in np.ndenumerate(total_lengths):

        start_position -= np.abs(length) * parameters.wire_width

        # sets the winding number and if there is a partial winding 
        winding_num = np.ceil(total_densities[i[0]]).astype(int)
        if winding_num == total_densities[i[0]]:
            partial_winding = 1
        else:
            partial_winding = (total_densities[i[0]] 
                               - np.floor(total_densities[i[0]]))

        # sets whether we are in the low or high current section of the slower
        if i[0] < high_current_check:
            current = high_current
        else:
            current = low_current

        solenoid_B_field = calculate_B_field_solenoid_helper(z, parameters.n, 
                                                             current,
                                                             winding_num, 
                                                             start_position,
                                                             end_position, 
                                                             partial_winding) 
        total_B_field += solenoid_B_field * 10**4
        end_position = start_position

    return total_B_field

