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
import parameters
# import heatmap_script as heatmap
import zeeman_slower_configuration as zs

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


# This function calculates the total B field from the coil winding with the 
# half gap in between sections, and works with partial integer values
def calculate_B_field_coil_gap(coil_winding, current_for_coils, 
                               discretization, sections):

    # Starts total B field to zero initially
    total_B_field = np.zeros(len(discretization))

    for position, coil_num in np.ndenumerate(coil_winding):
        
        axial_position =  (position[0] * parameters.wire_width 
                           + parameters.wire_width / 2)
        if position[0] > sections[0]:
            axial_position += parameters.wire_width / 2 
        if position[0] > sections[1]:
            axial_position += parameters.wire_width / 2 
        if position[0] > sections[2]:
            axial_position += parameters.wire_width / 2 

        # get integer number of coils
        full_coils = np.floor(coil_num)

        for c in range(full_coils.astype(int)):
            radial_position = ((parameters.slower_diameter / 2) 
                               + (parameters.wire_height / 2) 
                               + c * parameters.wire_height)
            current = current_for_coils[position[0]]
            field_from_single_coil = \
                coil.B_z_single_coil(current, radial_position, axial_position) 
            total_B_field += field_from_single_coil(discretization)

        # get partial coil winding
        if coil_num != full_coils:

            partial_winding = coil_num - np.floor(coil_num)
            c = np.ceil(coil_num).astype(int)
            radial_position = ((parameters.slower_diameter / 2) 
                               + (parameters.wire_height / 2) 
                               + c * parameters.wire_height)
            current = current_for_coils[position[0]] * partial_winding
            field_from_single_coil = \
                coil.B_z_single_coil(current, radial_position, axial_position)
            total_B_field += field_from_single_coil(discretization)

    total_B_field = total_B_field * 10**4

    return total_B_field


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

z_result = np.linspace(0, ideal.slower_length_val+.1, 10000) 
y_result = ideal.get_ideal_B_field(ideal.ideal_B_field, z_result)

coil_winding, current_for_coils = \
  coil.give_coil_winding_and_current(num_coils, fixed_densities, densities, 
                                     fixed_lengths, np.round(final[0:-2]), 
                                     final[-2], final[-1])
total_field = coil.calculate_B_field_coil(coil_winding, current_for_coils, 
                                          z_result)


sections = [62, 98, 112]
total_field_gap = calculate_B_field_coil_gap(coil_winding, current_for_coils, 
                                             z_result, sections)

coil_winding_edited = [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.25, 
                       0.25, 0.25, 0.25, 0.25, 0.25, 0.5 , 0.5 , 0.5 , 
                       0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 1.  , 1.  , 
                       1.  , 0.5 , 0.5 , 1.  , 1.  , 1.  , 1.  , 1.  , 
                       1.  , 1.  , 1.  , 1.  , 1.25, 1.25, 1.25, 1.25, 
                       1.25, 1.5 , 1.5 , 1.5 , 1.5 , 1.5 , 1.5 , 1.5 , 
                       1.5 , 1.5 , 1.5 , 2.  , 2.  , 2.  , 2.  , 2.  , 
                       2.  , 2.  , 2.  , 2.  , 2.  , 2.  , 2.  , 3   , 
                       3   , 2.5 , 2.5 , 2.5 , 2.5 , 2.5 , 2.5 , 2.5 , 
                       2.5 , 3.  , 3.  , 3.  , 3.  , 3.  , 3.  , 3.  , 
                       3.  , 3.  , 3.  , 3.5 , 3.5 , 3.5 , 3.5 , 3.5 , 
                       3.5 , 3.5 , 3.5 , 4.  , 4.  , 4.  , 4.  , 4.  , 
                       4.  , 4.  , 4.  , 6 , 6 , 6 , 4.5 , 4.5 , 4.5 , 
                       4.5 , 7.  , 7.  , 7.  , 7.  , 7.  , 7.  , 7.  , 
                       2.  , 2.  , 2.  , 2.  , 2.  , 2.  ]

current_for_coils_edited = [30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 , 
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 , 
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 , 
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 , 
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 , 
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 , 
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 , 
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 , 
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 ,  30.8086634 , 
                            30.8086634 ,  30.8086634 ,  30.8086634 ,  
                            30.8086634 ,  30.8086634 , 160, 160, 160, 160, 
                            160, 160]

total_field_edited = calculate_B_field_coil_gap(coil_winding_edited, 
                                                current_for_coils_edited, 
                                                z_result, sections)


# Calculate the length of each section and additional metrics from the
# winding
# All sections are 0 indexed
# Section 3: 6-62
# section 2: 63-98
# Section 1: 99-112
# Section 4: 113-118


section0 = coil_winding_edited[0:6]
section3 = coil_winding_edited[6:63]
section2 = coil_winding_edited[63:99]
section1 = coil_winding_edited[99:113]
section4 = coil_winding_edited[113:119]

section0_current = current_for_coils_edited[0:6]
section3_current = current_for_coils_edited[6:63]
section2_current = current_for_coils_edited[63:99]
section1_current = current_for_coils_edited[99:113]
section4_current = current_for_coils_edited[113:119]

s3_length = coil.calculate_section_length(coil_winding_edited, 6, 62)
s2_length = coil.calculate_section_length(coil_winding_edited, 63, 98)
s1_length = coil.calculate_section_length(coil_winding_edited, 99, 112)
s4_length = coil.calculate_section_length(coil_winding_edited, 113, 118)

s3_resistance = parameters.resistance(s3_length)
s2_resistance = parameters.resistance(s2_length)
s1_resistance = parameters.resistance(s1_length)
s4_resistance = parameters.resistance(s4_length)

s3_power = parameters.power(section0_current[0], s3_length)
s2_power = parameters.power(section0_current[0], s2_length)
s1_power = parameters.power(section0_current[0], s1_length)
s4_power = parameters.power(section4_current[0], s4_length)

s3_voltage = parameters.voltage(section0_current[0], s3_length)
s2_voltage = parameters.voltage(section0_current[0], s2_length)
s1_voltage = parameters.voltage(section0_current[0], s1_length)
s4_voltage = parameters.voltage(section4_current[0], s4_length)

s3_temp_change = parameters.temp_increase_per_minute(section0_current[0], 
                                                     s3_length)
s2_temp_change = parameters.temp_increase_per_minute(section0_current[0], 
                                                     s2_length)
s1_temp_change = parameters.temp_increase_per_minute(section0_current[0], 
                                                     s1_length)
s4_temp_change = parameters.temp_increase_per_minute(section4_current[0], 
                                                     s4_length)

print("total length: ", coil.calculate_total_length(coil_winding))
print("section 3 length: ", s3_length)
print("section 2 length: ", s2_length)
print("section 1 length: ", s1_length)
print("section 4 length: ", s4_length)

print("section 3 resistance: ", s3_resistance)
print("section 2 resistance: ", s2_resistance)
print("section 1 resistance: ", s1_resistance)
print("section 4 resistance: ", s4_resistance)

print("section 3 power: ", s3_power)
print("section 2 power: ", s2_power)
print("section 1 power: ", s1_power)
print("section 4 power: ", s4_power)

print("section 3 voltage: ", s3_voltage)
print("section 2 voltage: ", s2_voltage)
print("section 1 voltage: ", s1_voltage)
print("section 4 voltage: ", s4_voltage)

print("section 3 temp change: ", s3_temp_change)
print("section 2 temp change: ", s2_temp_change)
print("section 1 temp change: ", s1_temp_change)
print("section 4 temp change: ", s4_temp_change)



total_field_edited = \
    coil.calculate_B_field_coil(section0+section3+section2+section1+section4, 
                                np.concatenate(
                                    (section0_current, 
                                     np.multiply(section3_current,1),
                                     np.multiply(section2_current,1),
                                     np.multiply(section1_current,1),
                                     np.multiply(section4_current,1))), 
                                z_result)




# fig, ax = plt.subplots()

# ax.plot(z_result, y_result, label="ideal B field", color="m", linestyle="--")
# ax.plot(z_result, total_field, label="coil winding", color="k", linestyle="-")
# ax.plot(z_result, total_field_gap, label="coil winding gapped", color="g", 
#         linestyle="-")
# ax.plot(z_result, total_field_edited, label="coil winding gapped edited", 
#         color="b", linestyle="-")

# ax.set_xlabel("Position (m)")
# ax.set_ylabel("B field (G)")
# ax.legend()

# fig.set_size_inches(12, 8)
# fig.savefig(os.path.join(folder_location, "gapped_winding.pdf"), 
#             bbox_inches="tight")



# # Make schematic of Zeeman slower
# fig, ax = plt.subplots()

# # Turn into a list of x positions and radii
# for position, coil_num in np.ndenumerate(coil_winding_edited):

#     axial_position = (position[0] * parameters.wire_width 
#                             + parameters.wire_width / 2)

#     # get integer number of coils
#     full_coils = np.ceil(coil_num)

#     for c in range(full_coils.astype(int)):
#         radial_position = c + 1

#         print(radial_position, axial_position)

#         # Add points to scatter plot
#         if radial_position < coil_num:
#             ax.scatter(axial_position, radial_position, s=20, marker="s", 
#                        facecolors="None", edgecolors="b")

#         elif (radial_position - coil_num) == 0.5:
#             ax.scatter(axial_position, radial_position, s=20, marker="o", 
#                        facecolors="None", edgecolors="b")

#         elif (radial_position - coil_num) == 0.75:
#             ax.scatter(axial_position, radial_position, s=20, marker="D", 
#                        facecolors="None", edgecolors="b")

#         else:
#             ax.scatter(axial_position, radial_position, s=20, marker="s", 
#                        facecolors="None", edgecolors="b")


# ax.set_xlabel("Position (m)")
# ax.set_ylabel("Layer #")
# ax.set_title("ZS Schematic")

# fig.set_size_inches(14, 2)
# fig.savefig(os.path.join(folder_location, "schematic_edited.pdf"), bbox_inches="tight")

##############################################################################
# Analyze data from 10/5/21 measurements


total_field_edited_lc = \
    calculate_B_field_coil_gap(section0+section3+section2+section1+section4, 
                               np.concatenate((section0_current, 
                                               np.multiply(section3_current,1),
                                               np.multiply(section2_current,1),
                                               np.multiply(section1_current,1),
                                               np.multiply(section4_current,0))), 
                               z_result, sections)
total_field_edited_hc = \
    calculate_B_field_coil_gap(section0+section3+section2+section1+section4, 
                               np.concatenate((section0_current, 
                                               np.multiply(section3_current,0),
                                               np.multiply(section2_current,0),
                                               np.multiply(section1_current,0),
                                               np.multiply(section4_current,1))), 
                               z_result, sections)
total_field_edited = calculate_B_field_coil_gap(coil_winding_edited, 
                                                current_for_coils_edited, 
                                                z_result, sections)

file_location = os.path.join("C:\\", "Users", "Lithium", "Documents", 
                               "zeeman_slower", "data_10.5.21")
position, background, lc, hc = \
    np.genfromtxt(os.path.join(file_location, "10.5.21_ZS_testing_data.csv"), 
                  dtype=float, delimiter=",", skip_header=1, unpack=True)


fig, ax = plt.subplots()

ax.plot(z_result, y_result, label="ideal B field", color="k", linestyle="--")
ax.plot(z_result, total_field_edited_lc, label="expected lc B field", 
        color="red")
ax.plot(z_result, total_field_edited_hc, label="expected hc B field", 
        color="blue")
ax.plot(z_result, total_field_edited, label="expected total B field", 
        color="green")


ax.plot((position*.01)-0.2516, -1*(lc-background)*30.81/2, linestyle="None", 
        marker=".", color="r", label="observed lc B field")
ax.plot((position*.01)-0.2516, -1*(hc-background)*195/2, linestyle="None", 
        marker=".", color="b", label="observed hc B field")
ax.plot(((position*.01)-0.2516), 
        (-1*(lc-background)*30.81/2-1*(hc-background)*195/2), marker=".", 
        color="k", label="observed total B field")

ax.set_xlabel("Position (m)") 
ax.set_ylabel("B field (G)")
ax.legend()

fig.set_size_inches(12, 8)
fig.savefig(os.path.join(os.path.join(file_location), "observed_data.pdf"), bbox_inches="tight")


# Calculate deviations 
B_field_range = (len(discretized_slower_adjusted) 
                      - (np.sum(fixed_lengths) - fixed_overlap) + 1)
total_field_final = calculate_B_field_coil_gap(coil_winding_edited, 
                                                current_for_coils_edited, 
                                                discretized_slower_adjusted, 
                                                sections)
ideal_B_field_comp = ideal.get_ideal_B_field(ideal.ideal_B_field, 
                                ((position*.01)-0.2516)[0:85])

fig1, ax1 = plt.subplots()
ax1.plot(discretized_slower_adjusted[0:B_field_range], 
              (total_field_final[0:B_field_range] 
              - ideal_B_field_adjusted[0:B_field_range]) * 10**(-4) 
              * ideal.mu0_li / ideal.hbar / ideal.linewidth_li, label="expected Li deviations")

ax1.plot(((position*.01)-0.2516)[0:85], 
              ((-1*(lc-background)*30.81/2-1*(hc-background)*195/2)[0:85]
              -ideal_B_field_comp) * 10**(-4) 
              * ideal.mu0_li / ideal.hbar / ideal.linewidth_li, label="observed Li deviations")
ax1.set_ylim(-4, 4)


ax1.set_xlabel("Position (m)")
ax1.set_ylabel("frequency shift / linewidth")
ax1.legend()

fig1.set_size_inches(12, 8)
fig1.savefig(os.path.join(os.path.join(file_location), "observed_data_deviations.pdf"), bbox_inches="tight")
