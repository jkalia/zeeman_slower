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


# Location to save data
# file_path = "/Users/jkalia/Documents/research/fletcher_lab/zeeman_slower"
folder_location = os.path.join("C:\\", "Users", "Lithium", "Documents", 
                               "zeeman_slower", "3.5mm", 
                               "optimization_plots")


# Arrays which define the solenoid configuration for the low current section. 
densities = [7, 6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1.25, 1, 0.5, 1, 
             0.5, 0.25, 0]

# Arrays which define the solenoid configuration for the high current section.
fixed_densities = [2]
fixed_lengths = [6]
fixed_overlap = 0

final = [-7.12653878e+00, -3.73971016e-07, -6.34518412e-07, -8.82164728e-07,
          7.01947561e-07,  6.91609592e+00,  8.16322065e+00,  7.57713685e+00,
          9.52046922e+00,  1.04963877e+01, -1.19580619e+01, -1.04047639e+01,
         -5.36808583e+00, -8.86173341e+00,  2.46843583e+00,  2.52389398e+00,
         -9.16285867e+00,  7.20514955e+00,  1.10000000e+02,  30.8086634 , 
          130.84645074]

discretized_slower_adjusted, ideal_B_field_adjusted, z_long, num_coils = \
        zs.discretize(fixed_lengths, fixed_overlap, ideal.eta_er)

z_result = np.linspace(0, ideal.slower_length_val, 10000)
y_result = ideal.get_ideal_B_field(ideal.ideal_B_field, z_result)

coil_winding, current_for_coils = \
  coil.give_coil_winding_and_current(num_coils, fixed_densities, densities, 
                                     fixed_lengths, np.round(final[0:-2]), 
                                     final[-2], final[-1])

# total_length = coil.calculate_total_length(coil_winding)
# high_current_length = coil.calculate_high_current_section_length(coil_winding, 
#     current_for_coils)
# low_current_length = coil.calculate_low_current_section_length(coil_winding, 
#     current_for_coils)

# total_field = coil.calculate_B_field_coil(coil_winding, current_for_coils, 
#                                           z_result)

# Make schematic of Zeeman slower
fig, ax = plt.subplots()

# Turn into a list of x positions and radii
for position, coil_num in np.ndenumerate(coil_winding):

    axial_position = (position[0] * parameters.wire_width 
                           + parameters.wire_width / 2)

    # get integer number of coils
    full_coils = np.ceil(coil_num)

    for c in range(full_coils.astype(int)):
        radial_position = c + 1

        print(radial_position, axial_position)

        # Add points to scatter plot
        if radial_position < coil_num:
            ax.scatter(axial_position, radial_position, s=20, marker="s", facecolors="None", edgecolors="b")

        elif (radial_position - coil_num) == 0.5:
            ax.scatter(axial_position, radial_position, s=20, marker="o", facecolors="None", edgecolors="b")

        elif (radial_position - coil_num) == 0.75:
            ax.scatter(axial_position, radial_position, s=20, marker="D", facecolors="None", edgecolors="b")

        else:
            ax.scatter(axial_position, radial_position, s=20, marker="s", facecolors="None", edgecolors="b")


fig.set_size_inches(14, 2)

ax.set_xlabel("Position (m)")
ax.set_ylabel("Layer #")
ax.set_title("ZS Schematic")

fig.savefig(os.path.join(folder_location, "schematic.pdf"), bbox_inches="tight")











