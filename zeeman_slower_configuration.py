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
                                    eta, ideal.initial_velocity_er, 0, 
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


# One Li linewidth deivation is equivalent to approximately 4 G
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

folder_location = os.path.join("/Users", "jkalia", "Documents", "research", 
                               "fletcher_lab", "zeeman_slower_cleaned", 
                               "zeeman_slower", "optimization_plots")
iterations = 100000
counter = 0

# Arrays which define the solenoid configuration for the low current section. 
densities = [7, 6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1.25, 1, 0.5, 1, 
             0.5, 0.25, 0]

# Arrays which define the solenoid configuration for the high current section.
fixed_densities = [2]
fixed_lengths = [6]
fixed_overlap = 0

z = np.linspace(0, ideal.slower_length_val, 10000)
y_data = ideal.get_ideal_B_field(ideal.ideal_B_field, z)


guess = [-7.12653878e+00, -3.73971016e-07, -6.34518412e-07, -8.82164728e-07,
          7.01947561e-07,  6.91609592e+00,  8.16322065e+00,  7.57713685e+00,
          9.52046922e+00,  1.04963877e+01, -1.19580619e+01, -1.04047639e+01,
         -5.36808583e+00, -8.86173341e+00,  2.46843583e+00,  2.52389398e+00,
         -9.16285867e+00,  7.20514955e+00,  1.10000000e+02,  30.8086634 , 
          130.84645074]
coils = guess[0:-2]
current_guess = guess[-2::]

# Run optimizer for coils and currents
rmse, li_deviation, av_li_deviation, flag, final = \
    run_optimization(fixed_densities, densities, fixed_lengths, 
                      fixed_overlap, z, y_data, guess, iterations, ideal.eta_er, 
                      folder_location, counter)


# Run optimizer for currents only
rmse, li_deviation, av_li_deviation, flag, final = \
        run_optimization_current(fixed_densities, densities, fixed_lengths, 
                                 fixed_overlap, coils, z, y_data, 
                                 current_guess, iterations, ideal.eta_er, 
                                 folder_location, counter)


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


##############################################################################
# Final coil winding! We have entered in half gaps to reflect accuracy of
# how the physical winding occurs and then done manual post-processing to 
# get rid of slight deviations in B field caused by said gaps. See winding.py
# for this. 
# eta = 0.486 if wire_width = wire_height = 0.0036

# coil_winding_edited = \
#        [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.25, 0.25, 0.25, 0.25,
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

# current_for_coils_edited = \
#       [ 30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
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
