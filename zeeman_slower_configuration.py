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
from scipy import special
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
def discretize(fixed_lengths, fixed_overlap, eta):

    new_coils = np.sum(fixed_lengths) - fixed_overlap
    
    slower_length_val, ideal_B_field = \
        ideal.get_slower_parameters(ideal.k_er, ideal.linewidth_er, ideal.m_er, 
                                    eta, ideal.initial_velocity_er, 
                                    ideal.mu0_er, ideal.laser_detuning_er)

    discretized_slower = \
        np.linspace(0, 
            (slower_length_val 
                - (slower_length_val % parameters.wire_width)), 
            np.ceil(slower_length_val / parameters.wire_width).astype(int)
        )
    discretized_slower_temp = \
        (discretized_slower + (parameters.wire_width / 2))[:-1].copy()
    discretized_slower_adjusted = \
        np.append(discretized_slower_temp, 
                  coil.add_coils(new_coils, len(discretized_slower_temp)))
    
    num_coils = len(discretized_slower_adjusted)

    ideal_B_field_adjusted = \
        ideal.get_ideal_B_field(ideal_B_field, 
                                discretized_slower_adjusted)

    z_long = \
        np.linspace(0, (slower_length_val 
                        + new_coils * parameters.wire_width), 
                    10000)

    return (discretized_slower_adjusted, ideal_B_field_adjusted, z_long, 
            num_coils)


def calculate_RMSE(ideal, calculated):
    return np.sqrt(((ideal - calculated)**2).mean())


# A one Li linewidth deivation is equivalent to approximately 4 G
def max_deviation(total_field_final, ideal_field, B_field_range):
    return np.amax(np.abs(
        ((total_field_final[0:B_field_range] - ideal_field[0:B_field_range]) 
         * 10**(-4) * ideal.mu0_li / ideal.hbar / ideal.linewidth_li)))
    

def average_deviation(total_field_final, ideal_field, B_field_range):
    return np.average(np.abs(
        ((total_field_final[0:B_field_range] - ideal_field[0:B_field_range]) 
         * 10**(-4) * ideal.mu0_li / ideal.hbar / ideal.linewidth_li)))


def error_function(z, threshold):
    return threshold * special.erf(z / threshold)


# Residuals have units of Gauss
def residuals(guess, y, z, num_coils, fixed_densities, densities, 
              fixed_lengths):
    z = y - solenoid.calculate_B_field_solenoid(z, num_coils, fixed_densities, 
                                                densities, fixed_lengths, 
                                                guess[0:-2], guess[-2], 
                                                guess[-1])
    return error_function(z, 14)
     

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
        

    coil_winding, current_for_coils = \
            coil.give_coil_winding_and_current(num_coils, fixed_densities, 
                                               densities, fixed_lengths, 
                                               np.round(lengths), low_current, 
                                               high_current)
    total_field = coil.calculate_B_field_coil(coil_winding, current_for_coils,
                                              discretization)

    return (solenoid_field, coil_winding, current_for_coils, total_field, 
            "round")


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


def retrieve_data(file_path):
    return pickle.load(open(file_path, "rb"))


# Wrapper for optimization
def run_optimization(fixed_densities, densities, fixed_lengths, fixed_overlap, 
                     z, y, guess, iterations, eta, folder_location, counter):
    
    discretized_slower_adjusted, ideal_B_field_adjusted, z_long, num_coils = \
        discretize(fixed_lengths, fixed_overlap, eta)

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
    av_li_deviation = average_deviation(total_field_final, 
                                        ideal_B_field_adjusted, B_field_range)

    # Plot title 
    title = ("lc = {:.3f}, hc = {:.3f}, \n "
        "fixed overlap = {}, coil winding len = {}, RMSE = {:.3f} ({}), "
        "max Li deviation = {:.3}, ave. li deviation = {:.3} \n "
        "optimizer flag = {}".format(final[-2], final[-1], fixed_overlap, 
                                     len(coil_winding_final), rmse, 
                                     rmse_label, li_deviation, 
                                     av_li_deviation, flag))

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

    return rmse, li_deviation, av_li_deviation, flag, final


##############################################################################
# Optimize without changing the coil winding, change the current only

def residuals_current(guess, y, z, num_coils, fixed_densities, densities, 
                      fixed_lengths, coils):
    return y - solenoid.calculate_B_field_solenoid(z, num_coils, 
                                                   fixed_densities, 
                                                   densities, fixed_lengths, 
                                                   coils, guess[-2], 
                                                   guess[-1])
     

def optimizer_current(residuals_current, guess, x, y, iterations, num_coils, 
                      fixed_densities, densities, fixed_lengths, 
                      fixed_overlap, coils):

    print("optimizing!")
    print("fixed_lengths = {}, fixed_overlap = {}".format(fixed_lengths, 
                                                          fixed_overlap))
    final, flag = optimize.leastsq(residuals_current, guess, 
                                   args=(y, x, num_coils, fixed_densities, 
                                         densities, fixed_lengths, coils), 
                                   maxfev=iterations)
    print("optimizing complete")
    return final, flag


def run_optimization_current(fixed_densities, densities, fixed_lengths, 
                             fixed_overlap, coils, z, y, guess, iterations, 
                             eta, folder_location, counter):
    
    discretized_slower_adjusted, ideal_B_field_adjusted, z_long, num_coils = \
        discretize(fixed_lengths, fixed_overlap, eta)

    # Data used for calculations of measures of goodness 
    B_field_range = (len(discretized_slower_adjusted) 
                     - (np.sum(fixed_lengths) - fixed_overlap) + 1)

    final, flag = optimizer_current(residuals_current, guess, z, y, 
                                    iterations, num_coils, 
                                    fixed_densities, densities, fixed_lengths, 
                                    fixed_overlap, coils)

    (solenoid_field_init, coil_winding_init, 
        current_for_coils_init, total_field_init) = \
        get_configurations(z_long, num_coils, fixed_densities, densities, 
                           fixed_lengths, coils, guess[-2], guess[-1], 
                           discretized_slower_adjusted, ideal_B_field_adjusted,
                           B_field_range)[0:4]

    (solenoid_field_final, coil_winding_final, 
        current_for_coils_final, total_field_final, rmse_label) = \
        get_configurations(z_long, num_coils, fixed_densities, densities, 
                           fixed_lengths, coils, final[-2], final[-1], 
                           discretized_slower_adjusted, ideal_B_field_adjusted,
                           B_field_range)

    print("coil_winding: ", coil_winding_final)
    print("current_for_coils: ", current_for_coils_final)

    # Measures of goodness
    rmse = calculate_RMSE(ideal_B_field_adjusted[0:B_field_range], 
                          total_field_final[0:B_field_range])
    li_deviation = max_deviation(total_field_final, ideal_B_field_adjusted, 
                                 B_field_range)
    av_li_deviation = average_deviation(total_field_final, 
                                        ideal_B_field_adjusted, B_field_range)


    title = ("lc = {:.3f}, hc = {:.3f}, eta = {:.4f},  \n "
        "fixed overlap = {}, coil winding len = {}, RMSE = {:.3f} ({}), "
        "max Li deviation = {:.3}, ave. li deviation = {:.3} \n "
        "optimizer flag = {}".format(final[-2], final[-1], eta, fixed_overlap, 
                                     len(coil_winding_final), rmse, 
                                     rmse_label, li_deviation, 
                                     av_li_deviation, flag))

    # Name folder 
    directory = \
        "{}sections_{}hclength_{}hcmaxdensity_{}overlap_{:.4f}eta_{}counter"\
        "_current_only".format(
        len(densities), np.sum(fixed_lengths), np.amax(fixed_densities), 
        fixed_overlap, eta, counter)

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
            final, flag, eta)

    file_name = "data.pickle"
    f = os.path.join(file_path, file_name)

    save_data(data, f)

    return rmse, li_deviation, av_li_deviation, flag, final


##############################################################################
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


# folder_location = os.path.join("C:\\", "Users", "Lithium", "Documents", 
#                                "zeeman_slower", "3.5mm", 
#                                "optimization_plots")
# iterations = 100000
# counter = 0

# # Arrays which define the solenoid configuration for the low current section. 
# densities = [7, 6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1.25, 1, 0.5, 1, 
#              0.5, 0.25, 0]

# # Arrays which define the solenoid configuration for the high current section.
# fixed_densities = [2]
# fixed_lengths = [6]
# fixed_overlap = 0

# z = np.linspace(0, ideal.slower_length_val, 10000)
# y_data = ideal.get_ideal_B_field(ideal.ideal_B_field, z)


# guess = [-7.12653878e+00, -3.73971016e-07, -6.34518412e-07, -8.82164728e-07,
#           7.01947561e-07,  6.91609592e+00,  8.16322065e+00,  7.57713685e+00,
#           9.52046922e+00,  1.04963877e+01, -1.19580619e+01, -1.04047639e+01,
#          -5.36808583e+00, -8.86173341e+00,  2.46843583e+00,  2.52389398e+00,
#          -9.16285867e+00,  7.20514955e+00,  1.10000000e+02,  30.8086634 , 
#           130.84645074]
# coils = guess[0:-2]
# current_guess = guess[-2::]

    
# rmse, li_deviation, av_li_deviation, flag, final = \
#         run_optimization_current(fixed_densities, densities, fixed_lengths, 
#                                  fixed_overlap, coils, z, y_data, 
#                                  current_guess, iterations, ideal.eta_er, 
#                                  folder_location, counter)



# rmse, li_deviation, av_li_deviation, flag, final = \
#     run_optimization(fixed_densities, densities, fixed_lengths, 
#                       fixed_overlap, z, y_data, guess, iterations, ideal.eta_er, 
#                       folder_location, counter)


###############################################################################
# # Iterate fixed_lengths to find best solution

# min_length = 4
# max_length = 10

# # Initialize array for storing data
# rmse_array = np.zeros(((max_length - min_length + 1), 
#                       np.ceil(max_length / 2).astype(int) + 1))
# deviation_array = np.zeros(((max_length - min_length + 1), 
#                             np.ceil(max_length / 2).astype(int) + 1))
# average_array = np.zeros(((max_length - min_length + 1), 
#                             np.ceil(max_length / 2).astype(int) + 1))

# # Iterate over fixed lengths
# for i in range(min_length, (max_length + 1), 1):
#     fixed_lengths[0] = i
    
#     # Set max overlap
#     max_overlap = np.ceil(fixed_lengths[0] / 2).astype(int)
#     if max_overlap > 1:
#         max_overlap = 1

#     # Iterate over fixed_overlap
#     for j in range(max_overlap + 1):

#         flag = 0
#         flag_2 = 0
#         counter = 0
#         fixed_overlap = j

#         while (flag != 1) and (flag != 3):

#             print("fixed_lengths: ", fixed_lengths)
#             print("fixed_overlap: ", fixed_overlap)
#             print("counter: ", counter)

#             # Run optimization and collect data
#             rmse, li_deviation, av_li_deviation, flag, final = \
#                 run_optimization(fixed_densities, densities, fixed_lengths, 
#                                   fixed_overlap, z, y_data, guess, iterations, eta,
#                                   folder_location, counter)

#             print("rmse: ", rmse)   
#             print("li_deviation: ", li_deviation)
#             print("average li_deviation: ", av_li_deviation)
#             guess = final
#             counter += 1

#             if flag == 2:
#                 flag_2 += 1
#             if flag_2 > 200:
#                 break

#         rmse_array[(fixed_lengths[0] - min_length)][fixed_overlap] = rmse 
#         deviation_array[(fixed_lengths[0] - min_length)][fixed_overlap] = \
#             li_deviation
#         average_array[(fixed_lengths[0] - min_length)][fixed_overlap] = \
#             av_li_deviation

#         print("rmse_array: ", rmse_array)
#         print("deviation_array: ", deviation_array)
#         print("average_array: ", average_array)


# print("rmse_array: ", rmse_array)
# print("deviation_array: ", deviation_array)
# print("average_array: ", average_array)

# data = (rmse_array, deviation_array, average_array)
# save_data(data, os.path.join(folder_location, "heatmap.pickle"))


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
# eta_max = 100s

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

coil_winding_edited = [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.25, 0.25, 0.25, 0.25,
       0.25, 0.25, 0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 0.5 ,
       1.  , 1.  , 1.  , 0.5 , 0.5 , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  ,
       1.  , 1.  , 1.  , 1.25, 1.25, 1.25, 1.25, 1.25, 1.5 , 1.5 , 1.5 ,
       1.5 , 1.5 , 1.5 , 1.5 , 1.5 , 1.5 , 1.5 , 2.  , 2.  , 2.  , 2.  ,
       2.  , 2.  , 2.  , 2.  , 2.  , 2.  , 2.  , 2.  , 3   , 3   , 2.5 ,
       2.5 , 2.5 , 2.5 , 2.5 , 2.5 , 2.5 , 2.5 , 3.  , 3.  , 3.  , 3.  ,
       3.  , 3.  , 3.  , 3.  , 3.  , 3.  , 3.5 , 3.5 , 3.5 , 3.5 , 3.5 ,
       3.5 , 3.5 , 3.5 , 4.  , 4.  , 4.  , 4.  , 4.  , 4.  , 4.  , 4.  ,
       6 , 6 , 6 , 4.5 , 4.5 , 4.5 , 4.5 , 7.  , 7.  , 7.  , 7.  ,
       7.  , 7.  , 7.  , 2.  , 2.  , 2.  , 2.  , 2.  , 2.  ]

current_for_coils_edited = [ 30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
        30.8086634 , 160, 160, 160, 160, 160, 160]


################################################################################
# Plot motion of single atom through optimized slower winding


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


################################################################################
# Simulate motion of many atoms through ZS 

# Plot simulations
fig, ax = plt.subplots()

# Make instances of each kind of atom
li_atom = atom.Atom("Li")

# Simulation of atom in ideal B field
t_ideal, z_ideal, v_ideal, a_ideal = \
  simulate.simulate_atom(li_atom, ideal.Isat_li_d2 * 2, ideal.initial_velocity_li, ideal.laser_detuning_li*1.04,
                         optimized=False)
ax.plot(z_ideal, v_ideal, "k--", 
        label="ideal B field (v_initial = {:.0f})".format(
          ideal.initial_velocity_li)
        )

# Simulation of atoms through calculated B field for different initial 
# velocities
for x in range(11, 10, -1):
    t, z, v, a = simulate.simulate_atom(li_atom, ideal.Isat_li_d2 * 2, 
                                        ideal.initial_velocity_li * (x/100 + .9), ideal.laser_detuning_li*1.04, 
                                        coil_winding=coil_winding_edited, current_for_coils=current_for_coils_edited)
    ax.plot(z, v, 
            label="v_initial = {:.0f}".format(
                ideal.initial_velocity_li * (x/100 + .9)))


ax.set_xlabel("Position [m]")
ax.set_ylabel("Velocity [m/s]")
ax.set_title("Motion of Li atom in the Slower")
ax.legend()

# file_path = os.path.join("C:\\", "Users","Erbium", "Documents", 
#                          "zeeman_slower", "figs", "debugging1.pdf")
# fig.savefig(file_path, bbox_inches="tight")

plt.show()



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


################################################################################
# Make heatmaps of detuning versus saturation for the final velocity of atoms
# for the optimized slower

# Import data from 10/5/21 measurements
# file_location = os.path.join("C:\\", "Users", "Lithium", "Documents", 
#                              "zeeman_slower", "data_10.5.21")
# position_full, background, lc, hc = \
#     np.genfromtxt(os.path.join(file_location, "10.5.21_ZS_testing_data.csv"), 
#                   dtype=float, delimiter=",", skip_header=1, unpack=True)

# l_current = 30.81
# h_current = 195
# position_full = ((position_full * .01) - 0.2516)
# data = (-1 * ((lc - background) * l_current / 2 
#               + (hc - background) * h_current / 2))


# # Lithium
# shift = 20 * 10**6
# saturations = np.arange(1, 5.2, 0.2)

# li_atom = atom.Atom("Li")
# li_detunings = np.linspace(ideal.laser_detuning_li - shift, 
#                             ideal.laser_detuning_li + shift, 20)

# # Initialize array for storing data
# li_final_velocities = np.zeros((len(li_detunings), len(saturations)))

# for d, detuning in np.ndenumerate(li_detunings): 
#     for s, saturation in np.ndenumerate(saturations): 
#         v = simulate.simulate_atom(li_atom, saturation, 
#                                    ideal.initial_velocity_li, detuning, 
#                                    positions=position_full, data=data,
#                                    optimized=False, observed=True, 
#                                    full_output=False)
#         li_final_velocities[d][s] = v
#         print("li_final_velocities: ", li_final_velocities)
        
# print("li_final_velocities: ", li_final_velocities)
# save_data(li_final_velocities, 
#           os.path.join(file_location, "li_final_velocities.pickle"))


# # Erbium
# shift = 100 * 10**6
# saturations = np.arange(1, 5.2, 0.2)

# er_atom = atom.Atom("Er")
# er_detunings = np.linspace(ideal.laser_detuning_er - shift, 
#                           ideal.laser_detuning_er + shift / 4, 126)

# # Initialize array for storing data
# er_final_velocities = np.zeros((len(er_detunings), len(saturations)))

# for d, detuning in np.ndenumerate(er_detunings):
#     for s, saturation in np.ndenumerate(saturations):
#         v = simulate.simulate_atom(er_atom, saturation, 
#                                     ideal.initial_velocity_er, detuning, 
#                                     positions=position_full, data=data, 
#                                     optimized=False, observed=True, 
#                                     full_output=False)
#         er_final_velocities[d][s] = v 
#         print("er_final_velocities: ", er_final_velocities)

# print("er_final_velocities: ", er_final_velocities)
# save_data(er_final_velocities, 
#           os.path.join(file_location, "er_final_velocities_high_isat.pickle"))


################################################################################
# Unpickle heatmap data

# folder_location = os.path.join("C:\\", "Users","Lithium", "Documents", 
#                           "zeeman_slower", "figs")

# li_file = os.path.join(folder_location, "li_final_velocities.pickle")
# li_heatmap = retrieve_data(li_file)
# print("li_final_velocities: ", li_heatmap)

# er_file = os.path.join(folder_location, "er_final_velocities.pickle")
# er_heatmap = retrieve_data(er_file)
# print("er_final_velocities: ", er_heatmap)

# er_file_high_isat = os.path.join(folder_location, "er_final_velocities_high_isat.pickle")
# er_high_isat = retrieve_data(er_file_high_isat)
# print("high isat: ", er_high_isat)





###############################################################################
# Fixing eta, hc, and lc for the real data


def residuals_data(params, y, x):
    return y - model_data(x, params)


def model_data(x, params):
    slower_parameters = ideal.get_slower_parameters(ideal.k_er, 
                                                    ideal.linewidth_er, 
                                                    ideal.m_er, params[0], 
                                                    ideal.initial_velocity_er, 
                                                    ideal.mu0_er, 
                                                    ideal.laser_detuning_er)
    ideal_B_field = slower_parameters[1]
    return ideal.get_ideal_B_field(ideal_B_field, x)


# # Import data from 10/5/21 measurements
# file_location = os.path.join("C:\\", "Users", "Lithium", "Documents", 
#                              "zeeman_slower")
# position_full, background, lc, hc = \
#     np.genfromtxt(os.path.join(file_location, "10.5.21_ZS_testing_data.csv"), 
#                   dtype=float, delimiter=",", skip_header=1, unpack=True)


# position_full = ((position_full * .01) - 0.2516)
# position = position_full[0:93]

# low_currents = np.linspace(25, 28, 21)
# high_currents = np.linspace(90, 128, 61)

# reduced_chi_squareds = np.zeros((len(low_currents), len(high_currents)))
# etas = np.zeros((len(low_currents), len(high_currents)))


# # params[0] = eta
# init_params = [0.49]

# for i, l_current in np.ndenumerate(low_currents):
#     for j, h_current  in np.ndenumerate(high_currents):
#         print("low current = {}, high current = {}".format(l_current, 
#                                                             h_current))
#         data = (-1 * ((lc - background) * l_current / 2 
#                       + (hc - background) * h_current / 2))[0:93]
#         result = optimize.leastsq(residuals_data, init_params, 
#                                     args=(data, position), full_output=True)
#         chi_squared = (result[2]['fvec']**2).sum() / (len(result[2]['fvec']) 
#                                                       - len(result[0]))
#         reduced_chi_squareds[i][j] = chi_squared
#         etas[i][j] = result[0][0]
        
 
        
        

# # Save array
# data = (reduced_chi_squareds, etas)
# save_data(data, os.path.join(file_location, "reduced_chi_sqaured.pickle"))


# # Unpickle
# reduced_chi_squareds, etas = \
#     retrieve_data(os.path.join(file_location, 
#                                         "reduced_chi_sqaured.pickle"))


# fig, ax = plt.subplots()
# im = ax.imshow(reduced_chi_squareds)

# cbar = ax.figure.colorbar(im, ax=ax)

# ax.set_yticks(np.arange(len(low_currents)))
# ax.set_xticks(np.arange(len(high_currents)))
# ax.set_yticklabels(list(map(str, np.round(low_currents, 2))))
# ax.set_xticklabels(list(map(str, np.round(high_currents).astype(int))))

# fig.set_size_inches(12, 8)
# fig.savefig(os.path.join(os.path.join(file_location), "eta.pdf"), 
#             bbox_inches="tight")

# fig, ax = plt.subplots()

# ax.plot(z_result, y_result, label="ideal B field", color="k", linestyle="--")
# ax.plot(z_result, total_field_edited, label="expected total B field", 
#         color="green")
# ax.plot(position, data, marker=".", 
#         color="k", label="observed total B field")

# ax.set_xlabel("Position (m)")
# ax.set_ylabel("B field (G)")
# ax.legend()

# fig.set_size_inches(12, 8)
# fig.savefig(os.path.join(os.path.join(file_location), "data.pdf"), 
#             bbox_inches="tight")

# position = ((position * .01) - 0.2516)

# indices = np.where(reduced_chi_squareds == np.amin(reduced_chi_squareds))
# best_eta = etas[indices[0][0]][indices[1][0]]

# data = (-1 * ((lc - background) * low_currents[indices[0][0]] / 2 
#                       + (hc - background) * high_currents[indices[1][0]] / 2))

# fig, ax = plt.subplots()

# ax.plot(position_full, data, marker=".", color="k", 
#         label="observed total B field")
# ax.plot(position_full, model_data(position_full, [best_eta]), label="fit")

# ax.set_xlabel("Position (m)")
# ax.set_ylabel("B field (G)")
# ax.legend()

# fig.set_size_inches(12, 8)
# fig.savefig(os.path.join(os.path.join(file_location), "eta_fit.pdf"), 
#             bbox_inches="tight")
