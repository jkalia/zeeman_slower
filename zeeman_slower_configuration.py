# Jasmine Kalia
# May 10th, 2021
# Zeeman slower coil configuration code 
# In this script, we attempt to optimize our coil configuration to the ideal 
# magnetic field of our Zeeman slower. We do so by optimizing the position and 
# length of solenoids with fixed density. The density corresponds to the 
# number of coils, which is either given by an integer or a fraction. 
# A fraction winding corresponds to an integer winding with the corresponding
# fraction of the current. We first treat this problem as continuous, 
# by letting an optimizer vary the positions of the finite solenoids, then 
# discretize the optimized solution. We also limit the length of the slower 
# that the optimizer can produce by fixing the high current section of the 
# slower.         

import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
from scipy import optimize
from scipy import misc
import pickle 
import os

import ideal_field as ideal
import coil_configuration as coil 
import solenoid_configuration as solenoid 
import parameters
import plotting
import heatmap_script as heatmap
import simulate
import atom 



matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


# Adds coils to the end of the slower, gives the correct discretized ideal 
# B field profile and slower discretization, and gives discretization to use
# for plotting
def discretize(fixed_lengths, fixed_overlap):

    new_coils = np.sum(fixed_lengths) - fixed_overlap

    discretized_slower = \
        np.linspace(0, 
            (ideal.slower_length_val 
                - (ideal.slower_length_val % parameters.wire_width)), 
            np.ceil(ideal.slower_length_val / parameters.wire_width).astype(int)
        )
    discretized_slower_temp = \
        (discretized_slower + (parameters.wire_width / 2))[:-1].copy()
    discretized_slower_adjusted = \
        np.append(discretized_slower_temp, 
                  coil.add_coils(new_coils, len(discretized_slower_temp)))
    
    num_coils = len(discretized_slower_adjusted)

    ideal_B_field_adjusted = \
        ideal.get_ideal_B_field(ideal.ideal_B_field, 
                                discretized_slower_adjusted)

    z_long = \
        np.linspace(0, (ideal.slower_length_val 
                        + new_coils * parameters.wire_width), 
                    100000)

    return (discretized_slower_adjusted, ideal_B_field_adjusted, z_long, 
            num_coils)


def calculate_RMSE(ideal, calculated):
    return np.sqrt(((ideal - calculated)**2).mean())


def max_deviation(total_field_final, ideal_field, B_field_range):
    return np.amax(np.abs(
        ((total_field_final[0:B_field_range] - ideal_field[0:B_field_range]) 
         * 10**(-4) * ideal.mu0_li / ideal.hbar / ideal.linewidth_li)))


def residuals(guess, y, z, num_coils, fixed_densities, densities, 
              fixed_lengths):
    return y - solenoid.calculate_B_field_solenoid(z, num_coils, 
                                                   fixed_densities, 
                                                   densities, fixed_lengths, 
                                                   guess[0:-2], guess[-2], 
                                                   guess[-1])


def optimizer(residuals, guess, x, y, iterations, num_coils, fixed_densities, 
              densities, fixed_lengths, fixed_overlap):

    print("optimizing!")
    print("fixed_lengths = {}, fixed_overlap = {}".format(fixed_lengths, 
                                                          fixed_overlap))
    final, flag = optimize.leastsq(residuals, guess, 
                                   args=(y, x, num_coils, fixed_densities, 
                                         densities, fixed_lengths), 
                                   maxfev=iterations)
    print("optimizing complete")
    return final, flag


def get_configurations(x_long, num_coils, fixed_densities, densities, 
                       fixed_lengths, lengths, low_current, high_current, 
                       discretization, ideal_field, B_field_range):

    solenoid_field = \
        solenoid.calculate_B_field_solenoid(x_long, num_coils, fixed_densities, 
                                            densities, fixed_lengths, lengths, 
                                            low_current, high_current)
        

    coil_winding_round, current_for_coils_round = \
            coil.give_coil_winding_and_current(num_coils, fixed_densities, 
                                               densities, fixed_lengths, 
                                               np.round(lengths), low_current, 
                                               high_current)
    total_field_round = coil.calculate_B_field_coil(coil_winding_round, 
                                                    current_for_coils_round,
                                                    discretization)

    coil_winding_floor, current_for_coils_floor = \
            coil.give_coil_winding_and_current(num_coils, fixed_densities, 
                                               densities, fixed_lengths, 
                                               np.floor(lengths), low_current, 
                                               high_current)
    total_field_floor = coil.calculate_B_field_coil(coil_winding_floor, 
                                                    current_for_coils_floor,
                                                    discretization)

    coil_winding_ceil, current_for_coils_ceil = \
            coil.give_coil_winding_and_current(num_coils, fixed_densities, 
                                               densities, fixed_lengths, 
                                               np.ceil(lengths), low_current, 
                                               high_current)
    total_field_ceil = coil.calculate_B_field_coil(coil_winding_ceil, 
                                                   current_for_coils_ceil,
                                                   discretization)

    rmse_array = [0, 0, 0]
    rmse_array[0] = calculate_RMSE(ideal_field[0:B_field_range], 
                                   total_field_round[0:B_field_range])
    rmse_array[1] = calculate_RMSE(ideal_field[0:B_field_range], 
                                   total_field_floor[0:B_field_range])
    rmse_array[2] = calculate_RMSE(ideal_field[0:B_field_range], 
                                   total_field_ceil[0:B_field_range])
    min_rmse = np.amin(rmse_array)

    if rmse_array[0] == min_rmse:
        coil_winding = coil_winding_round
        current_for_coils = current_for_coils_round
        total_field = total_field_round
        label = "round"
    elif rmse_array[1] == min_rmse:
        coil_winding = coil_winding_floor
        current_for_coils = current_for_coils_floor
        total_field = total_field_floor
        label = "floor"
    else:
        coil_winding = coil_winding_ceil
        current_for_coils = current_for_coils_ceil
        total_field = total_field_ceil
        label = "ceil"

    return solenoid_field, coil_winding, current_for_coils, total_field, label


def save_data(data, file_path):
    file = open(file_path, "wb")
    pickle.dump(data, file)
    file.close()
    return 


def retrieve_run_data(file_path):
    retrieved_data = pickle.load(open(file_path, "rb"))

    fixed_densities = retrieved_data[0]
    densities = retrieved_data[1]
    fixed_lengths = retrieved_data[2]
    fixed_overlap = retrieved_data[3]
    guess = retrieved_data[4]
    final = retrieved_data[5]
    flag = retrieved_data[6]

    return (fixed_densities, densities, fixed_lengths, fixed_overlap, guess, 
            final, flag)


# Wrapper for optimization
def run_optimization(fixed_densities, densities, fixed_lengths, fixed_overlap, 
                     z, y, guess, iterations, folder_location, counter):
    
    discretized_slower_adjusted, ideal_B_field_adjusted, z_long, num_coils = \
        discretize(fixed_lengths, fixed_overlap)

    # Data used for calculations of measures of goodness 
    B_field_range = (len(discretized_slower_adjusted) 
                     - (np.sum(fixed_lengths) - fixed_overlap) + 1)

    final, flag = optimizer(residuals, guess, z, y, iterations, num_coils, 
                            fixed_densities, densities, fixed_lengths, 
                            fixed_overlap)

    (solenoid_field_init, coil_winding_init, 
        current_for_coils_init, total_field_init) = \
        get_configurations(z_long, num_coils, fixed_densities, densities, 
                           fixed_lengths, guess[0:-2], guess[-2], guess[-1], 
                           discretized_slower_adjusted, ideal_B_field_adjusted,
                           B_field_range)[0:4]

    (solenoid_field_final, coil_winding_final, 
        current_for_coils_final, total_field_final, rmse_label) = \
        get_configurations(z_long, num_coils, fixed_densities, densities, 
                           fixed_lengths, final[0:-2], final[-2], final[-1], 
                           discretized_slower_adjusted, ideal_B_field_adjusted,
                           B_field_range)

    print("coil_winding: ", coil_winding_final)
    print("current_for_coils: ", current_for_coils_final)

    # Measures of goodness
    rmse = calculate_RMSE(ideal_B_field_adjusted[0:B_field_range], 
                          total_field_final[0:B_field_range])
    li_deviation = max_deviation(total_field_final, ideal_B_field_adjusted, 
                                 B_field_range)

    # Plot title 
    title = ("lc = {}, hc = {}, \n "
        "fixed overlap = {}, RMSE = {} ({}), max Li deviation = {}, \n "
        "optimizer flag = {}".format(final[-2], final[-1], fixed_overlap, 
        rmse, rmse_label, li_deviation, flag))

    # Name folder 
    directory = \
        "{}sections_{}hclength_{}hcmaxdensity_{}overlap_{}counter".format(
        len(densities), np.sum(fixed_lengths), np.amax(fixed_densities), 
        fixed_overlap, counter)

    file_path = os.path.join(folder_location, directory)
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    plotting.make_plots(z, z_long, y, discretized_slower_adjusted, 
                        ideal_B_field_adjusted, solenoid_field_init, 
                        coil_winding_init, current_for_coils_init, 
                        total_field_init, solenoid_field_final, 
                        coil_winding_final, current_for_coils_final, 
                        total_field_final, fixed_overlap, B_field_range, 
                        title, file_path)

    data = (fixed_densities, densities, fixed_lengths, fixed_overlap, guess, 
            final, flag)

    file_name = "data.pickle"
    f = os.path.join(file_path, file_name)

    save_data(data, f)

    return rmse, li_deviation, flag, final


# Wrapper for plotting and generating values post-optimization
def post_optimization(fixed_densities, densities, fixed_lengths, fixed_overlap, 
                      z, y, guess, final, flag, folder_location):
    
    discretized_slower_adjusted, ideal_B_field_adjusted, z_long, num_coils = \
        discretize(fixed_lengths, fixed_overlap)

    # Data used for calculations of measures of goodness 
    B_field_range = (len(discretized_slower_adjusted) 
                     - (np.sum(fixed_lengths) - fixed_overlap) + 1)

    (solenoid_field_init, coil_winding_init, 
        current_for_coils_init, total_field_init) = \
        get_configurations(z_long, num_coils, fixed_densities, densities, 
                           fixed_lengths, guess[0:-2], guess[-2], guess[-1], 
                           discretized_slower_adjusted, ideal_B_field_adjusted,
                           B_field_range)[0:4]

    (solenoid_field_final, coil_winding_final, 
        current_for_coils_final, total_field_final, rmse_label) = \
        get_configurations(z_long, num_coils, fixed_densities, densities, 
                           fixed_lengths, final[0:-2], final[-2], final[-1], 
                           discretized_slower_adjusted, ideal_B_field_adjusted,
                           B_field_range)

    print("coil_winding: ", coil_winding_final)
    print("current_for_coils: ", current_for_coils_final)
    print("number of coils: ", len(coil_winding_final))
    print("number of coils: ", num_coils)

    # Measures of goodness
    rmse = calculate_RMSE(ideal_B_field_adjusted[0:B_field_range], 
                          total_field_final[0:B_field_range])
    li_deviation = max_deviation(total_field_final, ideal_B_field_adjusted, 
                              B_field_range)

    # Plot title 
    title = ("lc = {}, hc = {}, \n "
        "fixed overlap = {}, RMSE = {} ({}), max Li deviation = {}, \n "
        "optimizer flag = {}".format(final[-2], final[-1], fixed_overlap, 
        rmse, rmse_label, li_deviation, flag))

    # Name folder 
    directory = "{}sections_{}hclength_{}hcmaxdensity_{}overlap".format(
        len(densities), np.sum(fixed_lengths), np.amax(fixed_densities), 
        fixed_overlap)

    file_path = os.path.join(folder_location, directory)
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    plotting.make_plots(z, z_long, y, discretized_slower_adjusted, 
                        ideal_B_field_adjusted, solenoid_field_init, 
                        coil_winding_init, current_for_coils_init, 
                        total_field_init, solenoid_field_final, 
                        coil_winding_final, current_for_coils_final, 
                        total_field_final, fixed_overlap, B_field_range, 
                        title, file_path)

    data = (fixed_densities, densities, fixed_lengths, fixed_overlap, guess, 
            final, flag)

    file_name = "data.pickle"
    f = os.path.join(file_path, file_name)

    save_data(data, f)

    return rmse, li_deviation


################################################################################

# Run optimizer

# Location to save data
# folder_location = \
#     "/Users/jkalia/Documents/research/fletcher_lab/zeeman_slower/optimization_plots/"

# # Iterations for optimizer
# iterations = 20000

# Arrays which define the solenoid configuration for the low current section. 
# densities = [7, 6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1.25, 1, 0.5, 1, 
#              0.5, 0.25, 0]

# Arrays which define the solenoid configuration for the high current section.
# fixed_densities = [2]
# fixed_lengths = [6]
# fixed_overlap = 0

# z = np.linspace(0, ideal.slower_length_val, 100000)
# y_data = ideal.get_ideal_B_field(ideal.ideal_B_field, z)
# guess = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 110, 35, 120]

# rmse, li_deviation = run_optimization(fixed_densities, densities, 
#                                       fixed_lengths, fixed_overlap, 
#                                       z, y_data, guess, iterations, 
#                                       folder_location)


################################################################################

# Unpickle data

# folder_location = \
#     "/Users/jkalia/Documents/research/fletcher_lab/zeeman_slower/optimization_plots/"
# file = os.path.join(folder_location, 
#                     "19sections_6hclength_2hcmaxdensity_0overlap_new", 
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


################################################################################

# Post optimization

folder_location = \
    "/Users/jkalia/Documents/research/fletcher_lab/zeeman_slower/optimization_plots_post/"

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


# rmse, li_deviation = post_optimization(fixed_densities, densities, 
#                                        fixed_lengths, fixed_overlap, 
#                                        z, y_data, guess, final, flag,
#                                        folder_location)


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


################################################################################
# TODO: today, get all the code working to generate these heatmaps


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


# fig_li, ax_li = plt.subplots()

# ax_li.plot(z_i, v_i, "k--", label="ideal B field (v_initial = {:.0f})".format(
#            v_i_li))
# ax_li.plot(z, v, label="v_initial = {:.0f}".format(v_i_li))
            
# ax_li.set_xlabel("Position [m]")
# ax_li.set_ylabel("Velocity [m/s]")
# ax_li.set_title("Motion of Li atom in the Slower")
# ax_li.legend()


# file_path = os.path.join("C:\\", "Users","Erbium", "Documents", 
#                          "zeeman_slower", "figs", "debugging1.pdf")
# fig.savefig(file_path, bbox_inches="tight")


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


# fig_er, ax_er = plt.subplots()

# ax_er.plot(z_i, v_i, "k--", label="ideal B field (v_initial = {:.0f})".format(
#            v_i_er))
# ax_er.plot(z, v, label="v_initial = {:.0f}".format(v_i_er))
            
# ax_er.set_xlabel("Position [m]")
# ax_er.set_ylabel("Velocity [m/s]")
# ax_er.set_title("Motion of Er atom in the Slower")
# ax_er.legend()



# plt.show()






shift = 60 * 10**6
saturations = np.linspace(1, 2, 20)

# Lithium
li_atom = atom.Atom("Li")
detunings = np.linspace(ideal.laser_detuning_li - shift, 
                        ideal.laser_detuning_li + shift, 24)
v_cutoff = 20

# Initialize array for storing data
final_velocities = np.zeros((len(detunings), len(saturations)))

for d in range(len(detunings)):
    for s in range(len(saturations)):
        v = simulate.simulate_atom(li_atom, s, ideal.initial_velocity_li, d, 
                                   coil_winding=coil_winding, 
                                   current_for_coils=current_for_coils, 
                                   optimized=True, full_output=False)
        if v < v_cutoff:
            final_velocities[d][s] = v
        else:
            final_velocities[d][s] = 100

print("final_velocities: ", final_velocities)






















################################################################################
# # Plot simulations
# fig, ax = plt.subplots()

# # Simulation of atom in ideal B field
# t_ideal, z_ideal, v_ideal, a_ideal = \
#   simulate.simulate_atom("Li", ideal.Isat_li_d2 * 2, ideal.initial_velocity_li, 
#                          optimized=False)
# ax.plot(z_ideal, v_ideal, "k--", 
#         label="ideal B field (v_initial = {:.0f})".format(
#           ideal.initial_velocity_li)
#         )

# # Simulation of atoms through calculated B field for different initial 
# # velocities
# for x in range(11, 9, -1):
#     t, z, v, a = simulate.simulate_atom("Li", ideal.Isat_li_d2 * 2, 
#                                         ideal.initial_velocity_li * (x/100 + .9), 
#                                         coil_winding, current_for_coils)
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





# coils_used = np.count_nonzero(coil_winding)

# MOT_distance = (len(coil_winding) * parameters.wire_width 
#                 + parameters.length_to_MOT_from_ZS)

# z = np.linspace(0, MOT_distance + .1, 10000)
# y = ideal.get_ideal_B_field(ideal.ideal_B_field, z)

# total_field = coil.calculate_B_field_coil(coil_winding, current_for_coils, z)
# zprime = np.gradient(total_field)


# fig, ax = plt.subplots()
# fig1, ax1 = plt.subplots()


# ax.plot(z, total_field, label="calculated B field")
# ax.plot(z, y, label="ideal B field")
# ax.axvline(x=MOT_distance, linestyle="--", color="k", label="MOT location")
# ax.legend()

# plt.show()


# ax1.set_xlabel("Position (m)")
# ax1.set_ylabel("B field (Gauss)", color="tab:red")
# ax1.plot(z, total_field, color="tab:red")
# ax1.plot(z, y, linestyle="--", color="tab:red")
# ax1.axvline(x=MOT_distance, linestyle="--", color="k", label="MOT location")
# ax1.set_xlim(MOT_distance-0.05, MOT_distance+0.05)
# ax1.set_ylim(-10, 50)
# ax1.tick_params(axis="y", labelcolor="tab:red")


# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# ax2.set_ylabel("Gradient (Gauss/cm)", color="tab:blue")  # we already handled the x-label with ax1
# ax2.plot(z, zprime*100, color="tab:blue")
# ax2.set_ylim(-20, 5)
# ax2.tick_params(axis="y", labelcolor="tab:blue")

# ax1.legend()
# fig1.tight_layout()

# fig1.savefig("/Users/jkalia/Documents/research/fletcher_lab/zeeman_slower/figs/gradient.pdf")

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

# # plt.show()






