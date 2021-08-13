# Jasmine Kalia
# August 2nd, 2021
# Zeeman slower compensation code     

import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
import pickle 
import os

import ideal_field as ideal
import coil_configuration as coil
import parameters
import plotting
import zeeman_slower_configuration as zs
import simulate
import atom


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


file_path = os.path.join("C:\\", "Users", "Lithium", "Documents", 
                          "zeeman_slower", "figs", "compensation")

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


discretized_slower_adjusted, ideal_B_field_adjusted, z_long, num_coils = \
        zs.discretize(fixed_lengths, fixed_overlap)
coil_winding, current_for_coils = \
  coil.give_coil_winding_and_current(num_coils, fixed_densities, densities, 
                                      fixed_lengths, np.round(final[0:-2]), 
                                      final[-2], final[-1])

slower_length = len(coil_winding) * parameters.wire_width

low_current = -1 * current_for_coils[0]
high_current = -1 * current_for_coils[-1]

# First we deal with the high current section
# We will then add in the low current section, but we will have to watch the 
# radiuses for the coils 
# added_coils = [0, 0, 0, 0]

new_coils = [0, 0, 0, 0]
new_coils_positions = [0, 0, 0, 0]

MOT_distance = (slower_length + bleh + 
                + parameters.length_to_MOT_from_ZS)

z = np.linspace(0, MOT_distance + .1, 10000)
y = ideal.get_ideal_B_field(ideal.ideal_B_field, z)

total_field = coil.calculate_B_field_coil(coil_winding, current_for_coils, z)
zprime = np.gradient(total_field)


# Plot of total B field
fig, ax = plt.subplots()

ax.plot(z, total_field, label="calculated B field")
ax.plot(z, y, label="ideal B field")
ax.axvline(x=MOT_distance, linestyle="--", color="k", label="MOT location")
ax.legend()

fig.set_size_inches(12, 8)
fig.savefig(os.path.join(file_path, "total_field.pdf"), bbox_inches="tight")


# Zoomed in plot of B field and gradient at MOT
fig1, ax1 = plt.subplots()

ax1.set_xlabel("Position (m)")
ax1.set_ylabel("B field (Gauss)", color="tab:red")
ax1.plot(z, total_field, color="tab:red")
ax1.plot(z, y, linestyle="--", color="tab:red")
ax1.axvline(x=MOT_distance, linestyle="--", color="k", label="MOT location")
ax1.set_xlim(MOT_distance-0.02, MOT_distance+0.02)
ax1.set_ylim(-2, 10)
ax1.tick_params(axis="y", labelcolor="tab:red")

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel("Gradient (Gauss/cm)", color="tab:blue")  # we already handled the x-label with ax1
ax2.plot(z, zprime*100, color="tab:blue")
ax2.set_ylim(-10, 1)
ax2.tick_params(axis="y", labelcolor="tab:blue")

ax1.legend()
fig1.set_size_inches(12, 8)
fig1.tight_layout()
fig1.savefig(os.path.join(file_path, "gradient.pdf"), bbox_inches="tight")


# # Plot motion of atoms through ZS
# li_atom = atom.Atom("Li")
# er_atom = atom.Atom("Er")
# s = 2
# laser_detuning_li = ideal.laser_detuning_li
# laser_detuning_er = ideal.laser_detuning_er

# t_i_li, z_i_li, v_i_li, a_i_li = simulate.simulate_atom(li_atom, s, ideal.initial_velocity_li, 
#                                                         laser_detuning_li, 
#                                                         optimized=False)
# t_i_er, z_i_er, v_i_er, a_i_er = simulate.simulate_atom(er_atom, s, ideal.initial_velocity_er, 
#                                                         laser_detuning_er, 
#                                                         optimized=False)
# t_li, z_li, v_li, a_li = simulate.simulate_atom(li_atom, s, ideal.initial_velocity_li, 
#                                                 laser_detuning_li,
#                                                 coil_winding=coil_winding,
#                                                 current_for_coils=current_for_coils)
# t_er, z_er, v_er, a_er = simulate.simulate_atom(er_atom, s, ideal.initial_velocity_er, 
#                                                 laser_detuning_er,
#                                                 coil_winding=coil_winding,
#                                                 current_for_coils=current_for_coils)
 

# fig_sim, ax_sim = plt.subplots()

# ax_sim.plot(z_i_li, v_i_li, "k--", label="ideal B field (v_i_li = {:.0f})".format(
#             ideal.initial_velocity_li))
# ax_sim.plot(z_i_er, v_i_er, "k--", label="ideal B field (v_i_er = {:.0f})".format(
#             ideal.initial_velocity_er))

# ax_sim.plot(z_li, v_li, label="v_initial = {:.0f}".format(v_i_li))
# ax_sim.plot(z_er, v_er, label="v_initial = {:.0f}".format(v_i_er))
            
# ax_sim.set_xlabel("Position [m]")
# ax_sim.set_ylabel("Velocity [m/s]")
# ax_sim.set_title("Motion of atoms in the Slower")
# ax_sim.legend()

# fig_sim.set_size_inches(12, 8)
# fig_sim.tight_layout()
# fig_sim.savefig(os.path.join(file_path, "simulation.pdf"), bbox_inches="tight")



# fig, ax = plt.subplots()
# z = np.linspace(-0.4, 0.4, 10000)
# ax.plot(z, coil.B_total_rect_coil(30, 0.127, 0.127, 0, z) * 10**4)
# ax.plot(z, coil.B_total_rect_coil(30, 0.127, 0.127, .1, z) * 10**4)
# ax.plot(z, coil.B_total_rect_coil(30, 0.127, 0.127, -0.1, z) * 10**4)

# plt.show()





# file_path = os.path.join("C:\\", "Users", "Lithium", "Documents", 
#                           "zeeman_slower", "figs")

# fixed_densities = [2]
# densities = [7, 6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1.25, 1, 0.5, 1, 
#              0.5, 0.25, 0]
# fixed_lengths = [6]
# fixed_overlap = 0
# guess = [-7.22898856e+00, -1.92519981e-06, -6.34518412e-07, -8.82164728e-07,
#           7.01947561e-07,  7.06642982e+00,  8.12184856e+00,  7.59530427e+00,
#           9.50767008e+00,  1.04795059e+01, -1.19299365e+01, -1.03797288e+01,
#          -5.34390819e+00, -8.83375563e+00,  2.46071163e+00,  2.51653805e+00,
#          -9.12990925e+00, 7.16913954e+00,  1.10000000e+02,  2.98418721e+01,
#           1.28736289e+02]
# final = [-7.12653878e+00, -3.73971016e-07, -6.34518412e-07, -8.82164728e-07,
#           7.01947561e-07,  6.91609592e+00,  8.16322065e+00,  7.57713685e+00,
#           9.52046922e+00,  1.04963877e+01, -1.19580619e+01, -1.04047639e+01,
#          -5.36808583e+00, -8.86173341e+00,  2.46843583e+00, 2.52389398e+00,
#          -9.16285867e+00,  7.20514955e+00,  1.10000000e+02,  2.99625224e+01,
#           1.28534803e+02]
# flag = 1


# discretized_slower_adjusted, ideal_B_field_adjusted, z_long, num_coils = \
#         zs.discretize(fixed_lengths, fixed_overlap)
# coil_winding, current_for_coils = \
#   coil.give_coil_winding_and_current(num_coils, fixed_densities, densities, 
#                                      fixed_lengths, np.round(final[0:-2]), 
#                                      final[-2], final[-1])
  
# current_for_coils = [ 29.9625224,  29.9625224,  29.9625224,  29.9625224,  29.9625224,
#         29.9625224,  29.9625224,  29.9625224,  29.9625224,  29.9625224,
#         29.9625224,  29.9625224,  29.9625224,  29.9625224,  29.9625224,
#         29.9625224,  29.9625224,  29.9625224,  29.9625224,  29.9625224,
#         29.9625224,  29.9625224,  29.9625224,  29.9625224,  29.9625224,
#         29.9625224,  29.9625224,  29.9625224,  29.9625224,  29.9625224,
#         29.9625224,  29.9625224,  29.9625224,  29.9625224,  29.9625224,
#         29.9625224,  29.9625224,  29.9625224,  29.9625224,  29.9625224,
#         29.9625224,  29.9625224,  29.9625224,  29.9625224,  29.9625224,
#         29.9625224,  29.9625224,  29.9625224,  29.9625224,  29.9625224,
#         29.9625224,  29.9625224,  29.9625224,  29.9625224,  29.9625224,
#         29.9625224,  29.9625224,  29.9625224,  29.9625224,  29.9625224,
#         29.9625224,  29.9625224,  29.9625224,  29.9625224,  29.9625224,
#         29.9625224,  29.9625224,  29.9625224,  29.9625224,  29.9625224,
#         29.9625224,  29.9625224,  29.9625224,  29.9625224,  29.9625224,
#         29.9625224,  29.9625224,  29.9625224,  29.9625224,  29.9625224,
#         29.9625224,  29.9625224,  29.9625224,  29.9625224,  29.9625224,
#         29.9625224,  29.9625224,  29.9625224,  29.9625224,  29.9625224,
#         29.9625224,  29.9625224,  29.9625224,  29.9625224,  29.9625224,
#         29.9625224,  29.9625224,  29.9625224,  29.9625224,  29.9625224,
#         29.9625224,  29.9625224,  29.9625224,  29.9625224,  29.9625224,
#         29.9625224,  29.9625224,  29.9625224,  29.9625224,  29.9625224,
#         29.9625224,  29.9625224,  29.9625224, 128.534803 , 128.534803 ,
#         128.534803 , 128.534803 , 128.534803 , 128.534803 ]
  
# coil_winding_total = [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.25, 0.25, 0.25, 0.25,
#         0.25, 0.25, 0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 0.5 ,
#         1.  , 1.  , 1.  , 0.5 , 0.5 , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  ,
#         1.  , 1.  , 1.  , 1.25, 1.25, 1.25, 1.25, 1.25, 1.5 , 1.5 , 1.5 ,
#         1.5 , 1.5 , 1.5 , 1.5 , 1.5 , 1.5 , 1.5 , 2.  , 2.  , 2.  , 2.  ,
#         2.  , 2.  , 2.  , 2.  , 2.  , 2.  , 2.  , 2.  , 2.5 , 2.5 , 2.5 ,
#         2.5 , 2.5 , 2.5 , 2.5 , 2.5 , 2.5 , 2.5 , 3.  , 3.  , 3.  , 3.  ,
#         3.  , 3.  , 3.  , 3.  , 3.  , 3.  , 3.5 , 3.5 , 3.5 , 3.5 , 3.5 ,
#         3.5 , 3.5 , 3.5 , 4.  , 4.  , 4.  , 4.  , 4.  , 4.  , 4.  , 4.  ,
#         4.5 , 4.5 , 4.5 , 4.5 , 4.5 , 4.5 , 4.5 , 7.  , 7.  , 7.  , 7.  ,
#         7.  , 7.  , 7.  , 2.  , 2.  , 2.  , 2.  , 2.  , 2. ]

# coil_winding_lc = [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.25, 0.25, 0.25, 0.25,
#         0.25, 0.25, 0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 0.5 ,
#         1.  , 1.  , 1.  , 0.5 , 0.5 , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  ,
#         1.  , 1.  , 1.  , 1.25, 1.25, 1.25, 1.25, 1.25, 1.5 , 1.5 , 1.5 ,
#         1.5 , 1.5 , 1.5 , 1.5 , 1.5 , 1.5 , 1.5 , 2.  , 2.  , 2.  , 2.  ,
#         2.  , 2.  , 2.  , 2.  , 2.  , 2.  , 2.  , 2.  , 2.5 , 2.5 , 2.5 ,
#         2.5 , 2.5 , 2.5 , 2.5 , 2.5 , 2.5 , 2.5 , 3.  , 3.  , 3.  , 3.  ,
#         3.  , 3.  , 3.  , 3.  , 3.  , 3.  , 3.5 , 3.5 , 3.5 , 3.5 , 3.5 ,
#         3.5 , 3.5 , 3.5 , 4.  , 4.  , 4.  , 4.  , 4.  , 4.  , 4.  , 4.  ,
#         4.5 , 4.5 , 4.5 , 4.5 , 4.5 , 4.5 , 4.5 , 7.  , 7.  , 7.  , 7.  ,
#         7.  , 7.  , 7.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0. ]

# coil_winding_hc = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 2., 2., 2., 2., 2.]


  
# MOT_distance = (len(coil_winding) * parameters.wire_width 
#                 + parameters.length_to_MOT_from_ZS)

# z = np.linspace(0, MOT_distance + .1, 10000)
# y = ideal.get_ideal_B_field(ideal.ideal_B_field, z)

# total_field_total = coil.calculate_B_field_coil(coil_winding_total, current_for_coils, z)
# total_field_lc = coil.calculate_B_field_coil(coil_winding_lc, current_for_coils, z)
# total_field_hc = coil.calculate_B_field_coil(coil_winding_hc, current_for_coils, z)

# zprime_total = np.gradient(total_field_total)
# zprime_lc = np.gradient(total_field_lc)
# zprime_hc = np.gradient(total_field_hc)


# # Plot of total B field
# fig, ax = plt.subplots()

# ax.plot(z, y, label="ideal B field", color="tab:orange")
# ax.plot(z, total_field_total, label="calculated B field total",  color="royalblue")
# ax.plot(z, total_field_lc, label="calculated B field lc", color="cornflowerblue")
# ax.plot(z, total_field_hc, label="calculated B field hc", color="lightsteelblue")

# ax.axvline(x=MOT_distance, linestyle="--", color="k", label="MOT location = {}".format(MOT_distance))
# ax.set_title("Zeeman slower magnetic field (no compensation)")
# ax.set_xlabel("Position (m)")
# ax.set_ylabel("B field (Gauss)")
# ax.legend()

# fig.set_size_inches(12, 8)
# fig.savefig(os.path.join(file_path, "total_field_no_comp.pdf"), bbox_inches="tight")


# # Zoomed in plot of B field and gradient at MOT
# fig1, ax1 = plt.subplots()

# ax1.set_xlabel("Position (m)")
# ax1.set_ylabel("B field (Gauss)", color="tab:red")
# ax1.plot(z, total_field_total, label="B field total", color="indianred")
# ax1.plot(z, total_field_lc, label="B field lc", color="lightcoral")
# ax1.plot(z, total_field_hc, label="B field hc", color="rosybrown")
# ax1.plot(z, y, linestyle="--", color="k")
# ax1.axvline(x=MOT_distance, linestyle="--", color="k", label="MOT location = {}".format(MOT_distance))
# ax1.set_xlim(MOT_distance-0.02, MOT_distance+0.02)
# ax1.set_ylim(-2, 8)
# ax1.tick_params(axis="y", labelcolor="tab:red")

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# ax2.set_ylabel("Gradient total (Gauss/cm)", color="tab:blue")  # we already handled the x-label with ax1
# ax2.plot(z, zprime_total*100, label="total gradient", color="royalblue")
# ax2.plot(z, zprime_lc*100, label="lc gradient", color="cornflowerblue")
# ax2.plot(z, zprime_hc*100, label="hc gradient", color="lightsteelblue")
# ax2.set_ylim(-2.5, 1)
# ax2.tick_params(axis="y", labelcolor="tab:blue")

# ax1.legend(loc="lower right")
# ax2.legend(loc="lower left")
# ax1.set_title("Magnetic field and gradient at MOT position (no compensation)")
# fig1.set_size_inches(12, 8)
# fig1.tight_layout()
# fig1.savefig(os.path.join(file_path, "gradient_no_comp.pdf"), bbox_inches="tight")



# plt.show()









