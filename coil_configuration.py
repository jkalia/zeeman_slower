# Jasmine Kalia
# June 17th, 2021
# coil_configuration.py
# Module for zeeman_slower_configuration
# Contains all functions and variables relevant to the coil winding.


import numpy as np 
import scipy.constants

import parameters


# B field of coil as a function of z
def B_z_single_coil(current, radius, z_location):
    return lambda z : ((scipy.constants.mu_0 * current * radius**2) / 
                      (2 * ((radius**2 + (z_location - z)**2)**(3/2))))


# This function gives the coil winding from a discretized array 
# lengths = [l0, l1, ...] and densities = [d0, d1, ...], where l0 indicates the 
# the length of the last solenoid with thickness (or number of coils) d0. 
# This automatically limits us to coil configurations which are physically 
# valid. 
def give_coil_winding_and_current(num_coils, fixed_densities, 
                                  densities, fixed_lengths, lengths, 
                                  low_current, high_current):

    # Starts coil winding at zero initially
    coil_winding = np.zeros(num_coils)
    current_for_coils = np.zeros(num_coils) + low_current
    end_position = num_coils
    start_position = end_position
    high_current_check = len(fixed_lengths)

    total_lengths = np.concatenate((fixed_lengths, lengths))
    total_densities = np.concatenate((fixed_densities, densities))

    for i, length in np.ndenumerate(total_lengths):

        start_position -= np.abs(length)
        if start_position < 0:
                start_position = 0

        # sets the winding number
        winding_num = total_densities[i[0]]

        # sets whether we are in the low or high current section of the slower
        if i[0] < high_current_check:
            current = high_current
        else:
            current = low_current

        # set coil winding and current 
        for j in range(num_coils):
            if ((j >= start_position) and (j < end_position)):
                coil_winding[j] = winding_num
                current_for_coils[j] = current 

        end_position = start_position

    return coil_winding, current_for_coils


# This function calculates the total B field from the coil winding, and works 
# with partial integer values
def calculate_B_field_coil(coil_winding, current_for_coils, discretization):

    # Starts total B field to zero initially
    total_B_field = np.zeros(len(coil_winding))

    for position, coil_num in np.ndenumerate(coil_winding):
        axial_position =  (position[0] * parameters.wire_width 
                           + parameters.wire_width / 2)

        # get integer number of coils
        full_coils = np.floor(coil_num)

        for coil in range(full_coils.astype(int)):
            radial_position = ((parameters.slower_diameter / 2) 
                               + (parameters.wire_height / 2) 
                               + coil * parameters.wire_height)
            current = current_for_coils[position[0]]
            field_from_single_coil = \
                B_z_single_coil(current, radial_position, axial_position) 
            total_B_field += field_from_single_coil(discretization)

        # get partial coil winding
        if coil_num != full_coils:

            partial_winding = coil_num - np.floor(coil_num)
            coil = np.ceil(coil_num).astype(int)
            radial_position = ((parameters.slower_diameter / 2) 
                               + (parameters.wire_height / 2) 
                               + coil * parameters.wire_height)
            current = current_for_coils[position[0]] * partial_winding
            field_from_single_coil = \
                B_z_single_coil(current, radial_position, axial_position)
            total_B_field += field_from_single_coil(discretization)

    total_B_field = total_B_field * 10**4

    return total_B_field


# Add coils to the end of the slower 
def add_coils(num_coils, num_positions):
    added_coils = np.zeros(num_coils)
    for i in range(len(added_coils)):
        added_coils[i] = ((num_positions + i) * parameters.wire_width 
                          + parameters.wire_width / 2)
    return added_coils

