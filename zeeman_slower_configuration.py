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
# import simulate


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

    return (fixed_densities, densities, fixed_lengths, fixed_overlap, guess, 
            final)


# Wrapper for optimization
def run_optimization(fixed_densities, densities, fixed_lengths, fixed_overlap, 
                     z, y, guess, iterations, folder_location):
    
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
            final)

    file_name = "data.pickle"
    f = os.path.join(file_path, file_name)

    save_data(data, f)

    return rmse, li_deviation


# Give the optimized field with the correct discretization
def get_B_field_data(fixed_densities, densities, fixed_lengths, fixed_overlap, 
                     discretization, final):
    
    discretized_slower_adjusted, ideal_B_field_adjusted, z_long, num_coils = \
        discretize(fixed_lengths, fixed_overlap)

    # Data used for calculations of measures of goodness 
    B_field_range = (len(discretized_slower_adjusted) 
                     - (np.sum(fixed_lengths) - fixed_overlap) + 1)

    total_field_final, rmse_label = \
      get_configurations(z_long, num_coils, fixed_densities, 
                                           densities, fixed_lengths, 
                                           final[0:-2], final[-2], final[-1], 
                                           discretization, 
                                           ideal_B_field_adjusted, 
                                           B_field_range)[3:5]
    print("rmse_label: ", rmse_label) ##### TODO: this is returning ceil instead of round because of 
                                      ##### the fact that B field range is wrong  
                                      ##### works fine if you don't bullshit B_field       

    return total_field_final


# Wrapper for atom propagation
def propagation(fixed_densities, densities, fixed_lengths, fixed_overlap, 
                discretization, final):

    total_field = get_B_field_data(fixed_densities, densities, fixed_lengths,
                                   fixed_overlap, discretization, final)



    pass



# Wrapper for plotting and generating values post-optimization
def post_optimization(fixed_densities, densities, fixed_lengths, fixed_overlap, 
                      z, y, guess, final, folder_location):
    
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
        "o".format(final[-2], final[-1], fixed_overlap, 
        rmse, rmse_label, li_deviation))

    # Name folder 
    directory = "{}sections_{}hclength_{}hcmaxdensity_{}overlap_post3".format(
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

    return rmse, li_deviation

################################################################################


# Location to save data
folder_location = \
    "/Users/jkalia/Documents/research/fletcher_lab/zeeman_slower/post/"

# # Iterations for optimizer
# iterations = 20000

# Arrays which defines the solenoid configuration for the low current section. 
densities = [7, 6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1.25, 1, 0.5, 1, 
             0.5, 0.25, 0]

# Arrays which define the solenoid configuration for the high current section.
fixed_densities = [2]
fixed_lengths = [6]
fixed_overlap = 0

z = np.linspace(0, ideal.slower_length_val, 100000)
y_data = ideal.get_ideal_B_field(ideal.ideal_B_field, z)
# guess = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 110, 35, 120]

# rmse, li_deviation = run_optimization(fixed_densities, densities, 
#                                       fixed_lengths, fixed_overlap, 
#                                       z, y_data, guess, iterations, 
#                                       folder_location)


guess = [-7.44649506, 0.000217686255, 0.000289963625, 5.69787469e-07,
         -6.03636938e-07, 7.39882542, 7.99586292, 7.66019569, 9.46768959,
         10.4478657, -11.8695287, -10.3297987, -5.29031708, 8.77704977,
         -2.44430421, 2.50143956, 9.06183171, -7.09477639, 12.0, 29.5916806,
         129.141887]
final = [-7.18428594e+00, -2.85549832e-06, -9.70206319e-07,  5.69787469e-07,
         -6.03636938e-07,  6.99444598e+00,  8.14020624e+00,  7.59039476e+00,
          9.51184227e+00,  1.04877096e+01, -1.19425779e+01, -1.03911870e+01,
         -5.35495084e+00,  8.84634912e+00, -2.46417535e+00,  2.51996538e+00,
          9.14485245e+00, -7.18603523e+00,  1.20000000e+01,  2.98967317e+01,
          1.28602604e+02]



# rmse, li_deviation = post_optimization(fixed_densities, densities, 
#                                        fixed_lengths, fixed_overlap, 
#                                        z, y_data, guess, final, 
#                                        folder_location)



discretized_slower_adjusted, ideal_B_field_adjusted, z_long, num_coils = \
        discretize(fixed_lengths, fixed_overlap)

z_result = np.linspace(0, ideal.slower_length_val, 10000)
y_result = ideal.get_ideal_B_field(ideal.ideal_B_field, z_result)

coil_winding, current_for_coils = \
  coil.give_coil_winding_and_current(num_coils, fixed_densities, densities, 
                                     fixed_lengths, np.round(final[0:-2]), 
                                     final[-2], final[-1])
total_field = coil.calculate_B_field_coil(coil_winding, current_for_coils, 
                                          z_result)

# fig, ax = plt.subplots()

# ax.plot(z_result, y_result)

# plt.show()


# t, z, v, a = \
#   simulate.simulate_atom("Li", ideal.Isat_li_d2 * 2, coil_winding, 
#                          current_for_coils)
# print("final velocity: ", v[-1])

# t_ideal, z_ideal, v_ideal, a_ideal = \
#   simulate.simulate_atom("Li", ideal.Isat_li_d2 * 2, optimized=False)

# t_ideal, z_ideal, v_ideal, a_ideal = \
#   simulate.simulate_atom("Li", ideal.Isat_li_d2 * 1000, optimized=False)

# fig, ax = plt.subplots()
# ax.plot(z, v, label="propagation through optimized B field")
# ax.plot(z_ideal, v_ideal, 'k--', label='propagation through ideal B field')
# ax.set_xlabel("Position [m]")
# ax.set_ylabel("Velocity [m/s]")
# ax.plot(v_ideal, t_ideal)
# ax.set_xlim(0, ideal.slower_length_val)
# ax.set_title("Motion of Li atom in the Slower")
# ax.legend()

# plt.show()




# fig1, ax1 = plt.subplots()
# ax1.plot(z_result, y_result)
# ax1.plot(z_result, total_field)

# fig2, ax2, = plt.subplots()
# ax2.plot(z_result, (total_field - y_result) * 10**(-4) 
#               * ideal.mu0_li / ideal.hbar / ideal.linewidth_li, label="Li")
# ax2.axvline(x=ideal.slower_length_val, color="m")

# plt.show()







# Unpickle
# folder_location = \
#     "/Users/jkalia/Documents/research/fletcher_lab/zeeman_slower/plots/"
# file = os.path.join(folder_location, "run1", "data.pickle")
# (fixed_densities, densities, fixed_lengths, fixed_overlap, guess,
#             final) = retrieve_run_data(file)
# print("fixed_densities: ", fixed_densities)
# print("densities: ", densities)
# print("fixed_lengths: ", fixed_lengths)
# print("fixed_overlap: ", fixed_overlap)
# print("guess: ", guess)
# print("final: ", final)



# # Iterate fixed_lengths from 4 to 10 
# min_length = 4
# max_length = 10

# # Initialize array for storing data
# rmse_array = np.zeros(((max_length - min_length + 1), 
#                       np.ceil(max_length / 2).astype(int) + 1))
# deviation_array = np.zeros(((max_length - min_length + 1), 
#                            np.ceil(max_length / 2).astype(int) + 1))

# # Iterate over fixed lengths
# for i in range(min_length, (max_length + 1), 1):
#   fixed_lengths[0] = i 

#   # Set max overlap
#   max_overlap = np.ceil(fixed_lengths[0] / 2).astype(int)

#   for j in range(max_overlap + 1):
#       fixed_overlap = j

#       # Run optimization and collect data
#       rmse, li_deviation = run_optimization(fixed_densities, densities, 
#                                             fixed_lengths, fixed_overlap, 
#                                             z, y_data, guess, iterations)
#       print("rmse: ", rmse)
#       print("li_deviation: ", li_deviation)

#       rmse_array[(fixed_lengths[0] - min_length)][fixed_overlap] = rmse 
#       deviation_array[(fixed_lengths[0] - min_length)][fixed_overlap] = \
#           li_deviation

#       print("rmse_array: ", rmse_array)
#       print("deviation_array: ", deviation_array)


# print("rmse_array: ", rmse_array)
# print("deviation_array: ", deviation_array)


# data = (rmse_array, deviation_array)
# save_data(data, "heatmap.pickle")












