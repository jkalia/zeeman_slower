# Jasmine Kalia
# 7/27/21
# winding.py
# Zeeman slower schematic for winding purposes

import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
import pickle 
import os

import ideal_field as ideal
import coil_configuration as coil 
import solenoid_configuration as solenoid 
import parameters
import plotting
import heatmap_script as heatmap
import zeeman_slower_configuration as zs

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


file_path = os.path.join("C:\\", "Users", "Lithium", "Documents", 
                         "zeeman_slower", "figs")

z = np.linspace(0, ideal.slower_length_val, 100000)
y_data = ideal.get_ideal_B_field(ideal.ideal_B_field, z)

fixed_densities = [2]
densities = [7, 6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1.25, 1, 0.5, 1, 
             0.5, 0.25, 0]
fixed_lengths = [6]
fixed_overlap = 0
guess = [-7.22898856e+00, -1.92519981e-06, -6.34518412e-07, -8.82164728e-07,
          7.01947561e-07,  7.06642982e+00,  8.12184856e+00,  7.59530427e+00,
          9.50767008e+00,  1.04795059e+01, -1.19299365e+01, -1.03797288e+01,
         -5.34390819e+00, -8.83375563e+00,  2.46071163e+00,  2.51653805e+00,
         -9.12990925e+00, 7.16913954e+00,  1.10000000e+02,  2.98418721e+01,
          1.28736289e+02]
final = [-7.12653878e+00, -3.73971016e-07, -6.34518412e-07, -8.82164728e-07,
          7.01947561e-07,  6.91609592e+00,  8.16322065e+00,  7.57713685e+00,
          9.52046922e+00,  1.04963877e+01, -1.19580619e+01, -1.04047639e+01,
         -5.36808583e+00, -8.86173341e+00,  2.46843583e+00, 2.52389398e+00,
         -9.16285867e+00,  7.20514955e+00,  1.10000000e+02,  2.99625224e+01,
          1.28534803e+02]
flag = 1


rmse, li_deviation = zs.post_optimization(fixed_densities, densities, 
                                        fixed_lengths, fixed_overlap, 
                                        z, y_data, guess, final, flag,
                                        folder_location)


discretized_slower_adjusted, ideal_B_field_adjusted, z_long, num_coils = \
        discretize(fixed_lengths, fixed_overlap)

z_result = np.linspace(0, ideal.slower_length_val, 10000)
y_result = ideal.get_ideal_B_field(ideal.ideal_B_field, z_result)

coil_winding, current_for_coils = \
  coil.give_coil_winding_and_current(num_coils, fixed_densities, densities, 
                                     fixed_lengths, np.round(final[0:-2]), 
                                     final[-2], final[-1])

total_length = coil.calculate_total_length(coil_winding)
high_current_length = coil.calculate_high_current_section_length(coil_winding, 
    current_for_coils)
low_current_length = coil.calculate_low_current_section_length(coil_winding, 
    current_for_coils)

total_field = coil.calculate_B_field_coil(coil_winding, current_for_coils, 
                                          z_result)

# I think we basically want to make a scatter plot of this
fig, ax = plt.subplots()

