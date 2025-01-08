# Jasmine Kalia
# January 8th, 2025
# Simulation code
# After making the Zeeman slower, I verified in multiple ways that the 
# Zeeman slower would work properly. 1) I simulated the motion of an atom
# through the slower.  



###############################################################################
# Unpickle run data

# file = os.path.join("C:\\", "Users", "Lithium", "Documents", "zeeman_slower", 
#                     "3.5mm", "optimization_plots",
#                     "19sections_6hclength_2hcmaxdensity_0overlap", 
#                     "data.pickle")

# file = os.path.join("C:\\", "Users", "Lithium", "Documents", "zeeman_slower", 
#                     "3.5mm", "optimization_plots",
#                     "19sections_6hclength_2hcmaxdensity_0overlap", 
#                     "data.pickle")
# (fixed_densities, densities, fixed_lengths, fixed_overlap, guess,
#             final, flag) = retrieve_run_data(file)
# print("fixed_densities: ", fixed_densities)
# print("densities: ", densities)
# print("fixed_lengths: ", fixed_lengths)
# print("fixed_overlap: ", fixed_overlap)
# print("guess: ", guess)
# print("final: ", final)
# print("flag: ", flag)


###############################################################################
# Best optimized result for 3.5mm
# Trying to see if this winding works for some different eta value
# It does! eta_er = 0.486


# folder_location = os.path.join("C:\\", "Users", "Lithium", "Documents", 
#                                "zeeman_slower", "eta", 
#                                "optimization_plots")
# iterations = 50000
# counter = 0

# # Arrays which define the solenoid configuration for the low current section. 
# densities = [7, 6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1.25, 1, 0.5, 
#              1, 0.5, 0.25, 0]

# # Arrays which define the solenoid configuration for the high current section. 
# fixed_densities = [2]
# fixed_lengths = [6]
# fixed_overlap = 0


# guess = [-7.12653878e+00, -3.73971016e-07, -6.34518412e-07, -8.82164728e-07,
#           7.01947561e-07,  6.91609592e+00,  8.16322065e+00,  7.57713685e+00,
#           9.52046922e+00,  1.04963877e+01, -1.19580619e+01, -1.04047639e+01,
#          -5.36808583e+00, -8.86173341e+00,  2.46843583e+00,  2.52389398e+00,
#          -9.16285867e+00,  7.20514955e+00,  1.10000000e+02,  2.99625224e+01,
#           1.28534803e+02]
# coils = guess[0:-2]
# current_guess = guess[-2::]

# z = np.linspace(0, ideal.slower_length_val, 10000)
# y_data = ideal.get_ideal_B_field(ideal.ideal_B_field, z)

# # Data points to collect
# eta_max = 100

# # Arrays to store data
# eta_arr = np.zeros(eta_max)
# av_li_deviation_arr = np.zeros(eta_max)
# li_deviation_arr = np.zeros(eta_max)

# for x in range(0, eta_max):
#     eta = x * 0.001 + 0.41
    
#     # Have to recalculate these
#     slower_length_val, ideal_B_field = \
#         ideal.get_slower_parameters(ideal.k_er, ideal.linewidth_er, ideal.m_er, 
#                                     eta, ideal.initial_velocity_er, 
#                                     ideal.mu0_er, ideal.laser_detuning_er)
#     z = np.linspace(0, slower_length_val, 10000)
#     y_data = ideal.get_ideal_B_field(ideal_B_field, z)
    
#     rmse, li_deviation, av_li_deviation, flag, final = \
#         run_optimization_current(fixed_densities, densities, fixed_lengths, 
#                                   fixed_overlap, coils, z, y_data, 
#                                   current_guess, iterations, eta, 
#                                   folder_location, counter)
#     eta_arr[x] = eta
#     av_li_deviation_arr[x] = av_li_deviation
#     li_deviation_arr[x] = li_deviation


# fig, ax = plt.subplots()
# fig1, ax1 = plt.subplots()
# fig2, ax2 = plt.subplots()

# ax.plot(eta_arr, av_li_deviation_arr, label="average li deviation")
# ax.plot(eta_arr, li_deviation_arr, label="max li deviation")

# ax1.plot(eta_arr, av_li_deviation_arr, label="average li deviation")

# ax2.plot(eta_arr, li_deviation_arr, label="max li deviation")

# ax.set_xlabel("eta")
# ax.legend()
# ax1.set_xlabel("eta")
# ax1.legend()
# ax2.set_xlabel("eta")
# ax2.legend()


# file_path = os.path.join(folder_location, "eta_plot.pdf")
# fig.savefig(file_path, bbox_inches="tight")
# file_path = os.path.join(folder_location, "av_plot.pdf")
# fig1.savefig(file_path, bbox_inches="tight")
# file_path = os.path.join(folder_location, "dev_plot.pdf")
# fig2.savefig(file_path, bbox_inches="tight")


################################################################################
# Post optimization

# file_path = os.path.join("C:\\", "Users", "Lithium", "Documents", 
#                          "zeeman_slower", "figs")

# z = np.linspace(0, ideal.slower_length_val, 100000)
# y_data = ideal.get_ideal_B_field(ideal.ideal_B_field, z)

# Current best solution, for 3.5mm wire_width and wire_height 
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


# rmse, li_deviation = post_optimization(fixed_densities, densities, 
#                                        fixed_lengths, fixed_overlap, 
#                                        z, y_data, guess, final, flag,
#                                        folder_location)


# discretized_slower_adjusted, ideal_B_field_adjusted, z_long, num_coils = \
#         discretize(fixed_lengths, fixed_overlap)

# z = np.linspace(0, ideal.slower_length_val, 10000)
# y = ideal.get_ideal_B_field(ideal.ideal_B_field, z)

# coil_winding, current_for_coils = \
#   coil.give_coil_winding_and_current(num_coils, fixed_densities, densities, 
#                                      fixed_lengths, np.round(final[0:-2]), 
#                                      final[-2], final[-1])

# total_length = coil.calculate_total_length(coil_winding)
# high_current_length = coil.calculate_high_current_section_length(coil_winding, 
#     current_for_coils)
# low_current_length = coil.calculate_low_current_section_length(coil_winding, 
#     current_for_coils)

# total_field = coil.calculate_B_field_coil(coil_winding, current_for_coils, z)


##############################################################################
# Final coil winding! We have entered in half gaps to reflect accuracy of
# how the physical winding occurs and then done manual post-processing to 
# get rid of slight deviations in B field caused by said gaps. See winding.py
# for this. 
# eta = 0.486 if wire_width = wire_height = 0.0036

# coil_winding_edited = [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.25, 0.25, 0.25, 0.25,
#         0.25, 0.25, 0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 0.5 ,
#         1.  , 1.  , 1.  , 0.5 , 0.5 , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  ,
#         1.  , 1.  , 1.  , 1.25, 1.25, 1.25, 1.25, 1.25, 1.5 , 1.5 , 1.5 ,
#         1.5 , 1.5 , 1.5 , 1.5 , 1.5 , 1.5 , 1.5 , 2.  , 2.  , 2.  , 2.  ,
#         2.  , 2.  , 2.  , 2.  , 2.  , 2.  , 2.  , 2.  , 3   , 3   , 2.5 ,
#         2.5 , 2.5 , 2.5 , 2.5 , 2.5 , 2.5 , 2.5 , 3.  , 3.  , 3.  , 3.  ,
#         3.  , 3.  , 3.  , 3.  , 3.  , 3.  , 3.5 , 3.5 , 3.5 , 3.5 , 3.5 ,
#         3.5 , 3.5 , 3.5 , 4.  , 4.  , 4.  , 4.  , 4.  , 4.  , 4.  , 4.  ,
#         6 , 6 , 6 , 4.5 , 4.5 , 4.5 , 4.5 , 7.  , 7.  , 7.  , 7.  ,
#         7.  , 7.  , 7.  , 2.  , 2.  , 2.  , 2.  , 2.  , 2.  ]

# current_for_coils_edited = [ 30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#         30.8086634 , 160, 160, 160, 160, 160, 160]

################################################################################
# # Plot motion of single atom through optimized slower winding

# # For lithium
# li_atom = atom.Atom("Li")
# s = 2
# v_i_li = ideal.initial_velocity_li
# laser_detuning_li = ideal.laser_detuning_li

# t_i, z_i, v_i, a_i = simulate.simulate_atom(li_atom, s, v_i_li, 
#                                             laser_detuning_li, 
#                                             optimized=False)
# t, z, v, a = simulate.simulate_atom(li_atom, s, v_i_li, laser_detuning_li,
#                                     coil_winding=coil_winding,
#                                     current_for_coils=current_for_coils)

# v_ideal = simulate.simulate_atom(li_atom, s, v_i_li, laser_detuning_li, 
#                                   optimized=False, full_output=False)
# v_final = simulate.simulate_atom(li_atom, s, v_i_li, laser_detuning_li,
#                                   coil_winding=coil_winding,
#                                   current_for_coils=current_for_coils, 
#                                   full_output=False)
# print("v_ideal: ", v_ideal)
# print("v_final: ", v_final)

# fig_li, ax_li = plt.subplots()

# ax_li.plot(z_i, v_i, "k--", label="ideal B field (v_initial = {:.0f})".format(
#             v_i_li))
# ax_li.plot(z, v, label="v_initial = {:.0f}".format(v_i_li))
            
# ax_li.set_xlabel("Position [m]")
# ax_li.set_ylabel("Velocity [m/s]")
# ax_li.set_title("Motion of Li atom in the Slower")
# ax_li.legend()

# file_path = os.path.join("C:\\", "Users","Erbium", "Documents", 
#                           "zeeman_slower", "figs", "debugging_li.pdf")
# fig_li.savefig(file_path, bbox_inches="tight")


# # For erbium
# er_atom = atom.Atom("Er")
# s = 2
# v_i_er = ideal.initial_velocity_er
# laser_detuning_er = ideal.laser_detuning_er

# t_i, z_i, v_i, a_i = simulate.simulate_atom(er_atom, s, v_i_er, 
#                                             laser_detuning_er, 
#                                             optimized=False)
# t, z, v, a = simulate.simulate_atom(er_atom, s, v_i_er, laser_detuning_er,
#                                     coil_winding=coil_winding,
#                                     current_for_coils=current_for_coils)

# v_ideal = simulate.simulate_atom(er_atom, s, v_i_er, laser_detuning_er, 
#                                   optimized=False, full_output=False)
# v_final = simulate.simulate_atom(er_atom, s, v_i_er, laser_detuning_er,
#                                   coil_winding=coil_winding,
#                                   current_for_coils=current_for_coils, 
#                                   full_output=False)
# print("v_ideal: ", v_ideal)
# print("v_final: ", v_final)

# fig_er, ax_er = plt.subplots()

# ax_er.plot(z_i, v_i, "k--", label="ideal B field (v_initial = {:.0f})".format(
#             v_i_er))
# ax_er.plot(z, v, label="v_initial = {:.0f}".format(v_i_er))
            
# ax_er.set_xlabel("Position [m]")
# ax_er.set_ylabel("Velocity [m/s]")
# ax_er.set_title("Motion of Er atom in the Slower")
# ax_er.legend()

# file_path = os.path.join("C:\\", "Users","Erbium", "Documents", 
#                           "zeeman_slower", "figs", "debugging_er.pdf")
# fig_er.savefig(file_path, bbox_inches="tight")


################################################################################
# Simulate motion of many atoms through ZS 

# # Plot simulations
# fig, ax = plt.subplots()

# # Make instances of each kind of atom
# li_atom = atom.Atom("Li")

# # Simulation of atom in ideal B field
# t_ideal, z_ideal, v_ideal, a_ideal = \
#   simulate.simulate_atom(li_atom, ideal.Isat_li_d2 * 2, ideal.initial_velocity_li, ideal.laser_detuning_li*1.04,
#                           optimized=False)
# ax.plot(z_ideal, v_ideal, "k--", 
#         label="ideal B field (v_initial = {:.0f})".format(
#           ideal.initial_velocity_li)
#         )

# # Simulation of atoms through calculated B field for different initial 
# # velocities
# for x in range(11, 10, -1):
#     t, z, v, a = simulate.simulate_atom(li_atom, ideal.Isat_li_d2 * 2, 
#                                         ideal.initial_velocity_li * (x/100 + .9), ideal.laser_detuning_li*1.04, 
#                                         coil_winding=coil_winding_edited, current_for_coils=current_for_coils_edited)
#     ax.plot(z, v, 
#             label="v_initial = {:.0f}".format(
#                 ideal.initial_velocity_li * (x/100 + .9)))


# ax.set_xlabel("Position [m]")
# ax.set_ylabel("Velocity [m/s]")
# ax.set_title("Motion of Li atom in the Slower")
# ax.legend()

# # file_path = os.path.join("C:\\", "Users","Erbium", "Documents", 
# #                          "zeeman_slower", "figs", "debugging1.pdf")
# # fig.savefig(file_path, bbox_inches="tight")

# plt.show()



# Choose atom to simulate
# if atom=="Er":
#     laser_detuning = ideal.laser_detuning_er
#     v_initial = ideal.initial_velocity_er
#     v_final = ideal.final_velocity_er
# else:
#     laser_detuning = ideal.laser_detuning_li
#     v_initial = ideal.initial_velocity_li
#     v_final = ideal.final_velocity_li


# Make instances of each kind of atom
# li_atom = atom.Atom("Li")

# s = 2
# v_i_li = ideal.initial_velocity_li
# laser_detuning_li = ideal.laser_detuning_li

# er_atom = atom.Atom("Er")

# v_i_er = ideal.initial_velocity_er
# laser_detuning_er = ideal.laser_detuning_er

# s = 1

# # Plot simulations
# for x in range(-50, 51, 1):

#     fig, ax = plt.subplots()

#     laser_detuning_li = (ideal.laser_detuning_li 
#                          + ideal.laser_detuning_li * x/100)
#     print("laser_detuning_li: ", laser_detuning_li)

#     for y in range(-1, 2, 1):

#         v_i_li = (ideal.initial_velocity_li + y * 10)
#         print("v_i_li: ", v_i_li)

#         # Simulation of atom in ideal B field
#         t_ideal, z_ideal, v_ideal, a_ideal = \
#             simulate.simulate_atom(li_atom, s, v_i_li, laser_detuning_li, 
#                                    optimized=False)

#         # Simulation of atom through calculated B field 
#         t, z, v, a = simulate.simulate_atom(li_atom, s, v_i_li, 
#                                             laser_detuning_li, coil_winding, 
#                                             current_for_coils) 

#         ax.plot(z_ideal, v_ideal, "--", 
#                 label="ideal B field (v_i = {:.0f}, v_f = {:.0f})".format(
#                                                                     v_i_li, 
#                                                                     v[-1]))
#         ax.plot(z, v, label="v_i = {:.0f}, v_f = {:.0f}".format(v_i_li, v[-1]))

#     ax.set_xlabel("Position [m]")
#     ax.set_ylabel("Velocity [m/s]")
#     ax.set_title("Motion of Li atom in the Slower, \n" 
#                  "detuning = {}".format(laser_detuning_li))
#     ax.legend()

#     file_name = "detuning_{:.2e}.pdf".format(laser_detuning_li)

#     file_path = os.path.join("C:\\", "Users","Erbium", "Documents", 
#                              "zeeman_slower", "detuning", file_name)
#     fig.savefig(file_path, bbox_inches="tight")





# # v_i_li = ideal.initial_velocity_li 
# # laser_detuning_li = ideal.laser_detuning_li * 1.01


# er_atom = atom.Atom("Er")

# v_i_er = ideal.initial_velocity_er
# laser_detuning_er = ideal.laser_detuning_er

# s = 1

# # Plot simulations
# fig, ax = plt.subplots()
    
# # Simulation of atom in ideal B field
# t_ideal, z_ideal, v_ideal, a_ideal = \
#     simulate.simulate_atom(er_atom, s, v_i_er, laser_detuning_er, 
#                                optimized=False)

# # Simulation of atom through calculated B field 
# t, z, v, a = simulate.simulate_atom(er_atom, s, v_i_er, laser_detuning_er, 
#                                         coil_winding, current_for_coils) 

# ax.plot(z_ideal, v_ideal, "k--", 
#             label="ideal B field (v_i = {:.0f})".format(v_i_er))
# ax.plot(z, v, label="v_i = {:.0f}".format(v_i_er))



# ax.set_xlabel("Position [m]")
# ax.set_ylabel("Velocity [m/s]")
# ax.set_title("Motion of Er atom in the Slower \n detuning = {}, s = {}".format(laser_detuning_er, s))
# ax.legend()

# file_path = os.path.join("C:\\", "Users","Erbium", "Documents", 
#                          "zeeman_slower", "figs", "er2.pdf")
# fig.savefig(file_path, bbox_inches="tight")


# ################################################################################
# # Make heatmaps of detuning versus saturation for the final velocity of atoms
# # using the observed ZS and compensation coil data 

# # Value from compensation.py
# MOT_distance = 0.5348 

# # Import data from 10/5/21 measurements
# # ZS real data
# file_location = os.path.join("C:\\", "Users", "Lithium", "Documents", 
#                               "zeeman_slower")
# position_full, background_ZS, lc, hc = \
#     np.genfromtxt(os.path.join(file_location, "data_10.5.21", "10.5.21_ZS_testing_data.csv"), 
#                   dtype=float, delimiter=",", skip_header=1, unpack=True)

# l_current = 30.81
# h_current = 195
# position_full = ((position_full * .01) - 0.2516)
# data_ZS = (-1 * ((lc - background_ZS) * l_current / 2 
#               + (hc - background_ZS) * h_current / 2))


# # Use simulation data for comp coils (did not take enough of the real data)
# # Values from the Mathematica notebook
# # ZS comp coil simulated data
# B_field_comp = coil.B_total_rect_coil(4*95, 115*10**(-3), 125*10**(-3), MOT_distance - 0.055, position_full) \
#     + coil.B_total_rect_coil(-4*47, 115*10**(-3), 125*10**(-3), MOT_distance + 0.055, position_full)

# # Total field
# B_field_total = (data_ZS - B_field_comp)

# plt.plot(position_full, B_field_total, label="measured")
# plt.plot(position_full, ideal.get_ideal_B_field(ideal.ideal_B_field, position_full))
# plt.legend()
# plt.show()


# #Interpolated total field
# zs = np.linspace(0, 1, 1000)
# B_field_interp = np.interp(zs, position_full, B_field_total)

# plt.plot(position_full, B_field_total, label="measured")
# # plt.plot(zs, B_field_interp, label="interpolated field")
# # plt.plot(position_full, ideal.get_ideal_B_field(ideal.ideal_B_field, position_full), label="ideal, eta=0.486")
# plt.plot(zs, ideal.get_ideal_B_field(ideal.get_slower_parameters(ideal.k_er, ideal.linewidth_er, ideal.m_er, .469, 
#                                            ideal.initial_velocity_er, 5, ideal.mu0_er, 
#                                            ideal.laser_detuning_er)[1], zs), label="v_f=5 m/s, eta=0.47")
# plt.plot(zs, ideal.get_ideal_B_field(ideal.get_slower_parameters(ideal.k_li, ideal.linewidth_li, ideal.m_li, .343, 
#                                            ideal.initial_velocity_li, 20, ideal.mu0_li, 
#                                            ideal.laser_detuning_li)[1], zs), label="Li, v_f=20 m/s, eta=0.357")
# plt.xlim(-0.1, 0.6)
# # plt.xlim(0.3, 0.5)
# # plt.ylim(500, 800)
# plt.legend()
# plt.show()

# # For lithium
# li_atom = atom.Atom("Li")
# s = 2
# v_i_li = ideal.initial_velocity_li
# laser_detuning_li = ideal.laser_detuning_li

# # For erbium
# er_atom = atom.Atom("Er")
# s = 2
# v_i_er = ideal.initial_velocity_er
# laser_detuning_er = ideal.laser_detuning_er

# xs = np.linspace(0, 1, 1000)

# data_li = ideal.get_ideal_B_field(ideal.get_slower_parameters(ideal.k_li, ideal.linewidth_li, ideal.m_li, .357, 
#                                             ideal.initial_velocity_li, 0, ideal.mu0_li, 
#                                             ideal.laser_detuning_li)[1], xs)
# data_er = ideal.get_ideal_B_field(ideal.get_slower_parameters(ideal.k_er, ideal.linewidth_er, ideal.m_er, .47, 
#                                             ideal.initial_velocity_er, 0, ideal.mu0_er, 
#                                             ideal.laser_detuning_er)[1], xs)

# t0_li, z0_li, v0_li, a0_li = simulate.simulate_atom(li_atom, s, v_i_li, laser_detuning_li, 
#                                     positions=xs, data=data_li*10**(-4),
#                                     optimized=False, observed=True, 
#                                     full_output=True)

# t_li, z_li, v_li, a_li = simulate.simulate_atom(li_atom, s*1.3, v_i_li, laser_detuning_li*1.01, 
#                                     positions=position_full, data=B_field_total*10**(-4),
#                                     optimized=False, observed=True, 
#                                     full_output=True)

# t0_er, z0_er, v0_er, a0_er = simulate.simulate_atom(er_atom, s, v_i_er, laser_detuning_er, 
#                                     positions=xs, data=data_er*10**(-4),
#                                     optimized=False, observed=True, 
#                                     full_output=True)

# t_er, z_er, v_er, a_er = simulate.simulate_atom(er_atom, s, v_i_er, laser_detuning_er, 
#                                     positions=position_full, data=B_field_total*10**(-4),
#                                     optimized=False, observed=True, 
#                                     full_output=True)

# plt.plot(z0_li, v0_li, label="ideal Li", color="r")
# plt.plot(z0_li, z0_li*0+20, color="k", linestyle="dashed")
# plt.plot(z_li, v_li, label="optimized Li", color="darkred")

# plt.plot(z0_er, v0_er, label="ideal Er", color="b")
# plt.plot(z0_er, z0_er*0+5, color="k", linestyle="dashed")
# plt.plot(z_er, v_er, label="optimized Er", color="darkblue")

# plt.legend()
# plt.show()

##############################################################################
# Make heatmaps of detuning versus saturation, for fixed HC and LC
# using the observed ZS and compensation coil data 

# Value from compensation.py
MOT_distance = 0.5348 

# Import data from 10/5/21 measurements
# ZS real data
file_location = os.path.join("C:\\", "Users", "Lithium", "Documents", 
                              "zeeman_slower")
position_full, background_ZS, lc, hc = \
    np.genfromtxt(os.path.join(file_location, "data_10.5.21", "10.5.21_ZS_testing_data.csv"), 
                  dtype=float, delimiter=",", skip_header=1, unpack=True)

l_current = 35
h_current = 65.5
position_full = ((position_full * .01) - 0.2516)
data_ZS = (-1 * ((lc - background_ZS) * l_current / 2 
              + (hc - background_ZS) * h_current / 2))


# Use simulation data for comp coils (did not take enough of the real data)
# Values from the Mathematica notebook
# ZS comp coil simulated data
lc_current = 31
hc_current = 67.5
B_field_comp = coil.B_total_rect_coil(4*hc_current, 115*10**(-3), 125*10**(-3), MOT_distance - 0.055, position_full) \
    + coil.B_total_rect_coil(-4*lc_current, 115*10**(-3), 125*10**(-3), MOT_distance + 0.055, position_full)

# Total field
B_field_total = (data_ZS - B_field_comp)

plt.plot(position_full, B_field_total)
plt.plot(position_full, 
          ideal.get_ideal_B_field(ideal.get_slower_parameters(ideal.k_er, 
                                                              ideal.linewidth_er, ideal.m_er, 0.486, 
                            ideal.initial_velocity_er, 5, ideal.mu0_er, 
                            ideal.laser_detuning_er)[1], position_full), color="r", linestyle="-")
plt.show()



# # Lithium
# shift = 50 * 10**6
# saturations = np.arange(1, 5.2, 0.2)

# li_atom = atom.Atom("Li")
# li_detunings = np.linspace(ideal.laser_detuning_li - shift, 
#                             ideal.laser_detuning_li + shift, 51)

# # Initialize array for storing data
# li_final_velocities = np.zeros((len(li_detunings), len(saturations)))

# for d, detuning in np.ndenumerate(li_detunings): 
#     for s, saturation in np.ndenumerate(saturations): 
#         v = simulate.simulate_atom(li_atom, saturation, 
#                                     ideal.initial_velocity_li, detuning, 
#                                     positions=position_full, data=B_field_total*10**(-4),
#                                     optimized=False, observed=True, 
#                                     full_output=False)
#         li_final_velocities[d][s] = v
#         print("li_final_velocities: ", li_final_velocities)
        
# print("li_final_velocities: ", li_final_velocities)
# li_file_name = ("li_final_velocities_hc=" + str(h_current) + "A_lc=" + str(l_current) \
#                + "A.pickle")
# save_data(li_final_velocities, os.path.join(file_location, li_file_name))

# Erbium
shift = 100 * 10**6
saturations = np.arange(1, 5.2, 0.2)

er_atom = atom.Atom("Er")
er_detunings = np.linspace(ideal.laser_detuning_er - .1*shift, 
                          ideal.laser_detuning_er + .9*shift, 101)

# Initialize array for storing data
er_final_velocities = np.zeros((len(er_detunings), len(saturations)))

for d, detuning in np.ndenumerate(er_detunings):
    for s, saturation in np.ndenumerate(saturations):
        v = simulate.simulate_atom(er_atom, saturation, 
                                    ideal.initial_velocity_er, detuning, 
                                    positions=position_full, data=B_field_total*10**(-4), 
                                    optimized=False, observed=True, 
                                    full_output=False)
        er_final_velocities[d][s] = v 
        print("er_final_velocities: ", er_final_velocities)


print("er_final_velocities: ", er_final_velocities)
er_file_name = ("er_final_velocities_hc=" + str(h_current) + "A_lc=" + str(l_current) \
                + "A.pickle")
save_data(er_final_velocities, os.path.join(file_location, er_file_name))


##############################################################################
# # HC vs LC heatmap, for fixed detuning and saturation

# # Value from compensation.py
# MOT_distance = 0.5348 

# # Import data from 10/5/21 measurements
# # ZS real data
# file_location = os.path.join("C:\\", "Users", "Lithium", "Documents", 
#                               "zeeman_slower")
# position_full, background_ZS, lc, hc = \
#     np.genfromtxt(os.path.join(file_location, "data_10.5.21", "10.5.21_ZS_testing_data.csv"), 
#                   dtype=float, delimiter=",", skip_header=1, unpack=True)
# position_full = ((position_full * .01) - 0.2516)

# # Use simulation data for comp coils (did not take enough of the real data)
# # Values from the Mathematica notebook
# # ZS comp coil simulated data
# B_field_comp = coil.B_total_rect_coil(4*95, 115*10**(-3), 125*10**(-3), MOT_distance - 0.055, position_full) \
#     + coil.B_total_rect_coil(-4*47, 115*10**(-3), 125*10**(-3), MOT_distance + 0.055, position_full)

# # Parameters for heatmaps
# s_init = 1
# li_detuning = ideal.laser_detuning_li*1.01
# high_currents = np.linspace(50, 120, 141)
# low_currents = np.linspace(25, 40, 31)

# # Lithium
# li_atom = atom.Atom("Li")

# # Initialize array for storing data
# li_final_velocities = np.zeros((len(high_currents), len(low_currents)))

# for h, h_current in np.ndenumerate(high_currents): 
#     for l, l_current in np.ndenumerate(low_currents): 
#         data_ZS = (-1 * ((lc - background_ZS) * l_current / 2 
#               + (hc - background_ZS) * h_current / 2))
#         B_field_total = data_ZS - B_field_comp
#         v = simulate.simulate_atom(li_atom, s_init, 
#                                     ideal.initial_velocity_li, li_detuning, 
#                                     positions=position_full, data=B_field_total*10**(-4),
#                                     optimized=False, observed=True, 
#                                     full_output=False)
#         li_final_velocities[h][l] = v
#         print("li_final_velocities: ", li_final_velocities)
        
# print("li_final_velocities: ", li_final_velocities)
# li_file_name = ("li_final_velocities_hc=" + str(h_current) + "A_lc=" + str(l_current) \
#               + "A_s_init=" + str(s_init) + "_detuning=" + str(round(li_detuning*10**(-6))) + "MHz.pickle")
# save_data(li_final_velocities, os.path.join(file_location, li_file_name))

# # Erbium
# er_atom = atom.Atom("Er")
# er_detuning = ideal.laser_detuning_er*1.0

# # Initialize array for storing data
# er_final_velocities = np.zeros((len(high_currents), len(low_currents)))

# for h, h_current in np.ndenumerate(high_currents):
#     for l, l_current in np.ndenumerate(low_currents):
#         data_ZS = (-1 * ((lc - background_ZS) * l_current / 2 
#               + (hc - background_ZS) * h_current / 2))
#         B_field_total = data_ZS - B_field_comp
#         v = simulate.simulate_atom(er_atom, s_init, 
#                                     ideal.initial_velocity_er, er_detuning, 
#                                     positions=position_full, data=B_field_total*10**(-4), 
#                                     optimized=False, observed=True, 
#                                     full_output=False)
#         er_final_velocities[h][l] = v 
#         print("er_final_velocities: ", er_final_velocities)


# print("er_final_velocities: ", er_final_velocities)
# er_file_name = ("er_final_velocities_hc=" + str(h_current) + "A_lc=" + str(l_current) \
#              + "A_s_init=" + str(s_init) + "_detuning=" + str(round(er_detuning*10**(-6))) + "MHz.pickle")
# save_data(er_final_velocities, os.path.join(file_location, er_file_name))


################################################################################
# # Plot motion of single atom through observed ZS and comp coil data

# # Value from compensation.py
# MOT_distance = 0.5348 

# # Import data from 10/5/21 measurements
# # ZS real data
# file_location = os.path.join("C:\\", "Users", "Lithium", "Documents", 
#                               "zeeman_slower")
# position_full, background_ZS, lc, hc = \
#     np.genfromtxt(os.path.join(file_location, "data_10.5.21", "10.5.21_ZS_testing_data.csv"), 
#                   dtype=float, delimiter=",", skip_header=1, unpack=True)

# l_current = 30.81
# h_current = 195
# position_full = ((position_full * .01) - 0.2516)
# data_ZS = (-1 * ((lc - background_ZS) * l_current / 2 
#               + (hc - background_ZS) * h_current / 2))


# # Use simulation data for comp coils (did not take enough of the real data)
# # Values from the Mathematica notebook
# # ZS comp coil simulated data
# B_field_comp = coil.B_total_rect_coil(4*95, 115*10**(-3), 125*10**(-3), MOT_distance - 0.055, position_full) \
#     + coil.B_total_rect_coil(-4*47, 115*10**(-3), 125*10**(-3), MOT_distance + 0.055, position_full)

# # Total field
# B_field_total = data_ZS + B_field_comp

# # For lithium
# li_atom = atom.Atom("Li")
# s_init = 0.55
# v_i_li = ideal.initial_velocity_li
# laser_detuning_li = ideal.laser_detuning_li

# t, z, v, a = simulate.simulate_atom(li_atom, s_init, v_i_li, laser_detuning_li, 
#                                     positions=position_full, data=B_field_total*10**(-4),
#                                     optimized=False, observed=True, 
#                                     full_output=True)

# fig_li, ax_li = plt.subplots()

# ax_li.plot(z, v)
# ax_li.set_xlabel("Position [m]")
# ax_li.set_ylabel("Velocity [m/s]")
# ax_li.set_title("Motion of Li atom in the Slower")
# ax_li.legend()

# plt.show()

# file_path = os.path.join("C:\\", "Users","Erbium", "Documents", 
#                           "zeeman_slower", "figs", "debugging_li.pdf")
# fig_li.savefig(file_path, bbox_inches="tight")


# # For erbium
# er_atom = atom.Atom("Er")
# s_init = .575
# v_i_er = ideal.initial_velocity_er
# laser_detuning_er = ideal.laser_detuning_er

# t_i, z_i, v_i, a_i = simulate.simulate_atom(er_atom, s_init, v_i_er, 
#                                             laser_detuning_er, 
#                                             optimized=False)

# t, z, v, a = simulate.simulate_atom(er_atom, s, v_i_er, laser_detuning_er,
#                                     coil_winding=coil_winding,
#                                     current_for_coils=current_for_coils)

# t, z, v, a = simulate.simulate_atom(er_atom, s_init, v_i_er, laser_detuning_er, 
#                                     positions=position_full, data=B_field_total*10**(-4),
#                                     optimized=False, observed=True, 
#                                     full_output=True)

# v_ideal = simulate.simulate_atom(er_atom, s, v_i_er, laser_detuning_er, 
#                                   optimized=False, full_output=False)
# v_final = simulate.simulate_atom(er_atom, s, v_i_er, laser_detuning_er,
#                                   coil_winding=coil_winding,
#                                   current_for_coils=current_for_coils, 
#                                   full_output=False)
# print("v_ideal: ", v_ideal)
# print("v_final: ", v_final)

# fig_er, ax_er = plt.subplots()

# ax_er.plot(z_i, v_i, "k--", label="ideal B field (v_initial = {:.0f})".format(
#             v_i_er))
# ax_er.plot(z, v, label="v_initial = {:.0f}".format(v_i_er))
            
# ax_er.set_xlabel("Position [m]")
# ax_er.set_ylabel("Velocity [m/s]")
# ax_er.set_title("Motion of Er atom in the Slower")
# ax_er.legend()

# file_path = os.path.join("C:\\", "Users","Erbium", "Documents", 
#                           "zeeman_slower", "figs", "debugging_er.pdf")
# fig_er.savefig(file_path, bbox_inches="tight")


################################################################################
# Simulate motion of many atoms through ZS 

# # Plot simulations
# fig, ax = plt.subplots()

# # Make instances of each kind of atom
# li_atom = atom.Atom("Li")

# # Simulation of atom in ideal B field
# t_ideal, z_ideal, v_ideal, a_ideal = \
#   simulate.simulate_atom(li_atom, ideal.Isat_li_d2 * 2, ideal.initial_velocity_li, ideal.laser_detuning_li*1.04,
#                           optimized=False)
# ax.plot(z_ideal, v_ideal, "k--", 
#         label="ideal B field (v_initial = {:.0f})".format(
#           ideal.initial_velocity_li)
#         )

# # Simulation of atoms through calculated B field for different initial 
# # velocities
# for x in range(11, 10, -1):
#     t, z, v, a = simulate.simulate_atom(li_atom, ideal.Isat_li_d2 * 2, 
#                                         ideal.initial_velocity_li * (x/100 + .9), ideal.laser_detuning_li*1.04, 
#                                         coil_winding=coil_winding_edited, current_for_coils=current_for_coils_edited)
#     ax.plot(z, v, 
#             label="v_initial = {:.0f}".format(
#                 ideal.initial_velocity_li * (x/100 + .9)))


# ax.set_xlabel("Position [m]")
# ax.set_ylabel("Velocity [m/s]")
# ax.set_title("Motion of Li atom in the Slower")
# ax.legend()

# # file_path = os.path.join("C:\\", "Users","Erbium", "Documents", 
# #                          "zeeman_slower", "figs", "debugging1.pdf")
# # fig.savefig(file_path, bbox_inches="tight")

# plt.show()



# Choose atom to simulate
# if atom=="Er":
#     laser_detuning = ideal.laser_detuning_er
#     v_initial = ideal.initial_velocity_er
#     v_final = ideal.final_velocity_er
# else:
#     laser_detuning = ideal.laser_detuning_li
#     v_initial = ideal.initial_velocity_li
#     v_final = ideal.final_velocity_li


# Make instances of each kind of atom
# li_atom = atom.Atom("Li")

# s = 2
# v_i_li = ideal.initial_velocity_li
# laser_detuning_li = ideal.laser_detuning_li

# er_atom = atom.Atom("Er")

# v_i_er = ideal.initial_velocity_er
# laser_detuning_er = ideal.laser_detuning_er

# s = 1

# # Plot simulations
# for x in range(-50, 51, 1):

#     fig, ax = plt.subplots()

#     laser_detuning_li = (ideal.laser_detuning_li 
#                          + ideal.laser_detuning_li * x/100)
#     print("laser_detuning_li: ", laser_detuning_li)

#     for y in range(-1, 2, 1):

#         v_i_li = (ideal.initial_velocity_li + y * 10)
#         print("v_i_li: ", v_i_li)

#         # Simulation of atom in ideal B field
#         t_ideal, z_ideal, v_ideal, a_ideal = \
#             simulate.simulate_atom(li_atom, s, v_i_li, laser_detuning_li, 
#                                    optimized=False)

#         # Simulation of atom through calculated B field 
#         t, z, v, a = simulate.simulate_atom(li_atom, s, v_i_li, 
#                                             laser_detuning_li, coil_winding, 
#                                             current_for_coils) 

#         ax.plot(z_ideal, v_ideal, "--", 
#                 label="ideal B field (v_i = {:.0f}, v_f = {:.0f})".format(
#                                                                     v_i_li, 
#                                                                     v[-1]))
#         ax.plot(z, v, label="v_i = {:.0f}, v_f = {:.0f}".format(v_i_li, v[-1]))

#     ax.set_xlabel("Position [m]")
#     ax.set_ylabel("Velocity [m/s]")
#     ax.set_title("Motion of Li atom in the Slower, \n" 
#                  "detuning = {}".format(laser_detuning_li))
#     ax.legend()

#     file_name = "detuning_{:.2e}.pdf".format(laser_detuning_li)

#     file_path = os.path.join("C:\\", "Users","Erbium", "Documents", 
#                              "zeeman_slower", "detuning", file_name)
#     fig.savefig(file_path, bbox_inches="tight")





# # v_i_li = ideal.initial_velocity_li 
# # laser_detuning_li = ideal.laser_detuning_li * 1.01


# er_atom = atom.Atom("Er")

# v_i_er = ideal.initial_velocity_er
# laser_detuning_er = ideal.laser_detuning_er

# s = 1

# # Plot simulations
# fig, ax = plt.subplots()
    
# # Simulation of atom in ideal B field
# t_ideal, z_ideal, v_ideal, a_ideal = \
#     simulate.simulate_atom(er_atom, s, v_i_er, laser_detuning_er, 
#                                optimized=False)

# # Simulation of atom through calculated B field 
# t, z, v, a = simulate.simulate_atom(er_atom, s, v_i_er, laser_detuning_er, 
#                                         coil_winding, current_for_coils) 

# ax.plot(z_ideal, v_ideal, "k--", 
#             label="ideal B field (v_i = {:.0f})".format(v_i_er))
# ax.plot(z, v, label="v_i = {:.0f}".format(v_i_er))



# ax.set_xlabel("Position [m]")
# ax.set_ylabel("Velocity [m/s]")
# ax.set_title("Motion of Er atom in the Slower \n detuning = {}, s = {}".format(laser_detuning_er, s))
# ax.legend()

# file_path = os.path.join("C:\\", "Users","Erbium", "Documents", 
#                          "zeeman_slower", "figs", "er2.pdf")
# fig.savefig(file_path, bbox_inches="tight")

