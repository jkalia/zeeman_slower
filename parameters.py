# Jasmine Kalia
# June 17th, 2021
# parameters.py
# Module for zeeman_slower_configuration
# Contains all functions and variables relevant to the physical parameters of 
# the Zeeman slower.


# Physical parameters
slower_diameter = 0.0127 * 2                                # [m]
wire_width = 0.0036                                         # [m]
wire_height = 0.0036                                        # [m]
n = 1 / wire_width                                          # [turns per m]
length_to_MOT_from_ZS = 101 / 1000                          # [m]


resistivity_Cu = 1.68 * 10**(-8)                            # [Ohm m]
cross_section = ((1/8 * 2.54 * 10**(-2))**2 
                  - (1/16 * 2.54 * 10**(-2))**2)            # [m^2]
density_Cu = 8960                                           # [kg m^-3]
specific_heat_capacity_Cu = 385                             # [J kg^-1 K^-1]
volumetric_heat_capacity = (density_Cu 
                            * specific_heat_capacity_Cu)    # [J m^-3 K^-1]


def resistance(length):
    return resistivity_Cu * length / cross_section


def power(current, length):
    return current**2 * resistance(length)


def voltage(current, length):
    return current * resistance(length)


def temp_increase_per_minute(current, length):
    heat_capacity_per_section = (length * cross_section 
                                 * volumetric_heat_capacity)
    return (power(current, length) / heat_capacity_per_section * 60)





       