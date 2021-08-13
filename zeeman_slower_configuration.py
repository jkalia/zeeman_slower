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
import pickle 
import os

import ideal_field as ideal
import coil_configuration as coil 
import solenoid_configuration as solenoid 
import parameters
import plotting
# import heatmap_script as heatmap
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


def retrieve_heatmap_data(file_path):
    return pickle.load(open(file_path, "rb"))


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

# folder_location = \
#     "/Users/jkalia/Documents/research/fletcher_lab/zeeman_slower/optimization_plots_post/"

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

# print(coil_winding)

##############################################################################
# Rerun ZS optimization using 3.6mm for wire thickness
# Run optimizer

# Location to save data
folder_location = os.path.join("C:\\", "Users", "Lithium", "Documents", 
                               "zeeman_slower", "3.6mm_optimization_plots")

# Iterations for optimizer
iterations = 10000

# Arrays which define the solenoid configuration for the low current section. 
densities = [7, 6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1.25, 1, 0.5, 1, 
              0.5, 0.25, 0]

# Arrays which define the solenoid configuration for the high current section.
fixed_densities = [2]
fixed_lengths = [6]
fixed_overlap = 0

z = np.linspace(0, ideal.slower_length_val, 10000)
y_data = ideal.get_ideal_B_field(ideal.ideal_B_field, z)
guess = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 110, 35, 120]

# Iterate fixed_lengths from 4 to 10 
min_length = 4
max_length = 10

# Initialize array for storing data
rmse_array = np.zeros(((max_length - min_length + 1), 
                      np.ceil(max_length / 2).astype(int) + 1))
deviation_array = np.zeros(((max_length - min_length + 1), 
                           np.ceil(max_length / 2).astype(int) + 1))

# Iterate over fixed lengths
for i in range(min_length, (max_length + 1), 1):
    fixed_lengths[0] = i
    
    # Set max overlap
    max_overlap = np.ceil(fixed_lengths[0] / 2).astype(int)
    if max_overlap > 2:
        max_overlap = 2

    # Iterate over fixed_overlap
    for j in range(max_overlap + 1):

        flag = 0
        flag_2 = 0
        counter = 0
        fixed_overlap = j

        while (flag != 1) and (flag != 3):

            print("fixed_lengths: ", fixed_lengths)
            print("fixed_overlap: ", fixed_overlap)
            print("counter: ", counter)

            # Run optimization and collect data
            rmse, li_deviation, flag, final = \
                run_optimization(fixed_densities, densities, fixed_lengths, 
                                 fixed_overlap, z, y_data, guess, iterations,
                                 folder_location, counter)

            print("rmse: ", rmse)   
            print("li_deviation: ", li_deviation)
            guess = final
            counter += 1

            if flag == 2:
                flag_2 += 1
            if flag_2 > 200:
                break

        rmse_array[(fixed_lengths[0] - min_length)][fixed_overlap] = rmse 
        deviation_array[(fixed_lengths[0] - min_length)][fixed_overlap] = \
            li_deviation

        print("rmse_array: ", rmse_array)
        print("deviation_array: ", deviation_array)


print("rmse_array: ", rmse_array)
print("deviation_array: ", deviation_array)

data = (rmse_array, deviation_array)
save_data(data, os.path.join(folder_location, "heatmap.pickle"))



##############################################################################
# TODO: Put half gaps in coil winding
# TODO: Post-processing to get rid of slight deivations in B field caused by
# these gaps 


##############################################################################
# TODO: Calculate resistance, temperature change per minute, length, of each
# section




# Total B field
# fig, ax = plt.subplots()

# ax.plot(z, y, label="ideal B field", color="tab:orange")
# ax.plot(z, total_field, label="calculated B field",  color="royalblue")

# ax.legend()

# fig.set_size_inches(12, 8)
# fig.savefig(os.path.join(file_path, "total_field_no_comp.pdf"), bbox_inches="tight")

# plt.show()


################################################################################
# Plot motion of atom through ZS


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
#                                  optimized=False, full_output=False)
# v_final = simulate.simulate_atom(li_atom, s, v_i_li, laser_detuning_li,
#                                  coil_winding=coil_winding,
#                                  current_for_coils=current_for_coils, 
#                                  full_output=False)
# print("v_ideal: ", v_ideal)
# print("v_final: ", v_final)



# fig_li, ax_li = plt.subplots()

# ax_li.plot(z_i, v_i, "k--", label="ideal B field (v_initial = {:.0f})".format(
#            v_i_li))
# ax_li.plot(z, v, label="v_initial = {:.0f}".format(v_i_li))
            
# ax_li.set_xlabel("Position [m]")
# ax_li.set_ylabel("Velocity [m/s]")
# ax_li.set_title("Motion of Li atom in the Slower")
# ax_li.legend()


# file_path = os.path.join("C:\\", "Users","Erbium", "Documents", 
#                          "zeeman_slower", "figs", "debugging_li.pdf")
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
#                                  optimized=False, full_output=False)
# v_final = simulate.simulate_atom(er_atom, s, v_i_er, laser_detuning_er,
#                                  coil_winding=coil_winding,
#                                  current_for_coils=current_for_coils, 
#                                  full_output=False)
# print("v_ideal: ", v_ideal)
# print("v_final: ", v_final)


# fig_er, ax_er = plt.subplots()

# ax_er.plot(z_i, v_i, "k--", label="ideal B field (v_initial = {:.0f})".format(
#            v_i_er))
# ax_er.plot(z, v, label="v_initial = {:.0f}".format(v_i_er))
            
# ax_er.set_xlabel("Position [m]")
# ax_er.set_ylabel("Velocity [m/s]")
# ax_er.set_title("Motion of Er atom in the Slower")
# ax_er.legend()

# file_path = os.path.join("C:\\", "Users","Erbium", "Documents", 
#                          "zeeman_slower", "figs", "debugging_er.pdf")
# fig_er.savefig(file_path, bbox_inches="tight")

# plt.show()


################################################################################
# Make heatmaps of detuning versus saturation for the final velocity of atoms


# shift = 20 * 10**6
# saturations = np.arange(1, 5.2, 0.2)

# # Lithium
# li_atom = atom.Atom("Li")
# li_detunings = np.linspace(ideal.laser_detuning_li - shift, 
#                             ideal.laser_detuning_li + shift, 20)

# # Initialize array for storing data
# li_final_velocities = np.zeros((len(li_detunings), len(saturations)))

# for d, detuning in np.ndenumerate(li_detunings): ##DUMB
#     for s, saturation in np.ndenumerate(saturations): ##DUMB
#         v = simulate.simulate_atom(li_atom, saturation, 
#                                     ideal.initial_velocity_li, detuning, 
#                                     coil_winding=coil_winding, 
#                                     current_for_coils=current_for_coils, 
#                                     optimized=True, full_output=False)
#         li_final_velocities[d][s] = v
#         print("li_final_velocities: ", li_final_velocities)
        
# print("li_final_velocities: ", li_final_velocities)
# heatmap.make_heatmap(li_final_velocities, len(li_detunings), len(saturations), 
#                       "Final velocity of Lithium in ZS", "saturation",
#                       "detuning", file_path, "li_final_velocities.pdf")
# save_data(li_final_velocities, 
#           os.path.join(file_path, "li_final_velocities.pickle"))


# # Erbium
# shift = 100 * 10**6
# er_atom = atom.Atom("Er")
# er_detunings = np.arange(ideal.laser_detuning_er - shift, 
#                          ideal.laser_detuning_er, 1 * 10**6)

# # Initialize array for storing data
# er_final_velocities = np.zeros((len(er_detunings), len(saturations)))

# for d, detuning in np.ndenumerate(er_detunings):
#     for s, saturation in np.ndenumerate(saturations):
#         v = simulate.simulate_atom(er_atom, saturation, 
#                                     ideal.initial_velocity_er, detuning, 
#                                     coil_winding=coil_winding, 
#                                     current_for_coils=current_for_coils, 
#                                     optimized=True, full_output=False)
#         er_final_velocities[d][s] = v 
#         print("er_final_velocities: ", er_final_velocities)


# print("er_final_velocities: ", er_final_velocities)
# # heatmap.make_heatmap(er_final_velocities, len(er_detunings), len(saturations), 
# #                       "Final velocity of Erbium in ZS", "saturation",
# #                       "detuning", file_path, "er_final_velocities.pdf")
# save_data(er_final_velocities, 
#           os.path.join(file_path, "er_final_velocities_high_isat.pickle"))


################################################################################
# Unpickle heatmap data

# folder_location = os.path.join("C:\\", "Users","Lithium", "Documents", 
#                           "zeeman_slower", "figs")

# li_file = os.path.join(folder_location, "li_final_velocities.pickle")
# li_heatmap = retrieve_heatmap_data(li_file)
# print("li_final_velocities: ", li_heatmap)

# er_file = os.path.join(folder_location, "er_final_velocities.pickle")
# er_heatmap = retrieve_heatmap_data(er_file)
# print("er_final_velocities: ", er_heatmap)

# er_file_high_isat = os.path.join(folder_location, "er_final_velocities_high_isat.pickle")
# er_high_isat = retrieve_heatmap_data(er_file_high_isat)
# print("high isat: ", er_high_isat)


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


# # fig, ax = plt.subplots()
# fig1, ax1 = plt.subplots()


# # ax.plot(z, total_field, label="calculated B field")
# # ax.plot(z, y, label="ideal B field")
# # ax.axvline(x=MOT_distance, linestyle="--", color="k", label="MOT location")
# # ax.legend()

# # plt.show()


# ax1.set_xlabel("Position (m)")
# ax1.set_ylabel("B field (Gauss)", color="tab:red")
# ax1.plot(z, total_field, color="tab:red")
# ax1.plot(z, y, linestyle="--", color="tab:red")
# ax1.axvline(x=MOT_distance, linestyle="--", color="k", label="MOT location")
# ax1.set_xlim(MOT_distance-0.02, MOT_distance+0.02)
# ax1.set_ylim(0, 10)
# ax1.tick_params(axis="y", labelcolor="tab:red")


# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# ax2.set_ylabel("Gradient (Gauss/cm)", color="tab:blue")  # we already handled the x-label with ax1
# ax2.plot(z, zprime*100, color="tab:blue")
# ax2.set_ylim(-5, 0)
# ax2.tick_params(axis="y", labelcolor="tab:blue")

# ax1.legend()
# fig1.tight_layout()

# path_name = os.path.join("C:\\", "Users", "Lithium", "Documents", 
#                          "zeeman_slower", "figs", "gradient.pdf")
# plt.show()

# fig1.savefig(path_name)





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






