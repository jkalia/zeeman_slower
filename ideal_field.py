# Jasmine Kalia
# March 29th, 2021
# ideal_field.py
# Module for zeeman_slower_configuration
# Contains all functions and variables relevant to the ideal B field.


import numpy as np 
import scipy.constants 


# Physical constants
amu = scipy.constants.physical_constants["atomic mass constant"][0] # [kg]
hbar = scipy.constants.hbar                                         # [J s]
uB = scipy.constants.physical_constants["Bohr magneton"][0]         # [J T^-1]
c = scipy.constants.c                                               # [m s^-1]


# Er constants
m_er = 167.259 * amu                                                # [kg]
linewidth_er = 2 * np.pi * 29.7 * 10**6                             # [Hz]
lambda_er = 401 * 10**(-9)                                          # [m]
k_er = 2 * np.pi / lambda_er                                        # [m^-1]
omega_er = k_er * c                                                 # [s^-1]
Isat_er = 602                                                       # [W/m^2]
recoilvel_er = hbar * k_er / m_er


# Li constants
m_li = 6.941 * amu                                                  # [kg]
linewidth_li = 2 * np.pi * 5.872 * 10**6                            # [Hz]
lambda_li = 671 * 10**(-9)                                          # [m]
k_li = 2 * np.pi / lambda_li                                        # [m^-1]
omega_li = k_li * c                                                 # [s^-1]
Isat_li_d1 = 75.9                                                   # [W/m^2]
Isat_li_d2 = 25.4                                                   # [W/m^2]
recoilvel_li = hbar * k_li / m_li


# Laser detuning from atomic resonance frequency for increasing-field ZS
def get_laser_detuning(capture_velocity, k):
    return -k * capture_velocity / (2 * np.pi)


# Er parameters for ZS
initial_velocity_er = 470                                           # [m/s]
final_velocity_er = 5                                               # [m/s]
eta_er = 0.5
mu0_er = 1.13719 * uB
laser_detuning_er = round(get_laser_detuning(initial_velocity_er, k_er), -6)


# Li parameters for ZS
eta_li = 0.38
mu0_li = 1 * uB


# Determine ideal B field for slower
def max_acceleration(k, linewidth, m):
    return hbar * k * linewidth / (2 * m)

def slower_acceleration(max_acceleration, eta):
    return eta * max_acceleration

def slower_length(capture_velocity, slower_acceleration):
    return capture_velocity**2 / (2 * slower_acceleration)

def B0(capture_velocity, k, mu0):
    return hbar * k * capture_velocity / mu0

def Bbias(mu0, laser_detuning):
    return hbar * laser_detuning * 2 * np.pi/ mu0

def B_field(B0, Bbias, slower_length):
    return lambda z : Bbias + B0 * np.sqrt(1 - z / slower_length)

def get_slower_parameters(k, linewidth, m, eta, capture_velocity, mu0, 
                          laser_detuning):
    max_acceleration_val = max_acceleration(k, linewidth, m)
    slower_acceleration_val = slower_acceleration(max_acceleration_val, eta)
    slower_length_val = slower_length(capture_velocity, slower_acceleration_val)
    B0_val = B0(capture_velocity, k, mu0)
    Bbias_val = Bbias(mu0, laser_detuning)

    print("max acceleration: ", max_acceleration_val)
    print("slower_acceleration_val:", slower_acceleration_val)
    print("slower_length_val: ", slower_length_val)
    print("B0_val: ", B0_val)
    print("Bbias_val: ", Bbias_val)

    return slower_length_val, B_field(B0_val, Bbias_val, slower_length_val)


# Gives the ideal B field with the correct discretization
def get_ideal_B_field(ideal_B_field, discretization):
    B_field = ideal_B_field(discretization) * -1 * 10**4
    return np.nan_to_num(B_field)


# Obtain B field for increasing-field ZS 
slower_parameters = get_slower_parameters(k_er, linewidth_er, m_er, eta_er, 
                                          initial_velocity_er, mu0_er, 
                                          laser_detuning_er)
slower_length_val = slower_parameters[0]
ideal_B_field = slower_parameters[1] # function of z


