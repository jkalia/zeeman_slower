# Jasmine Kalia
# June 17th, 2021
# parameters.py
# Module for zeeman_slower_configuration
# Contains all functions and variables relevant to the physical parameters of 
# the Zeeman slower.


# Physical parameters
slower_diameter = 0.0127 * 2                                # [m]
wire_width = 0.0035                                         # [m]
wire_height = 0.0035                                        # [m]
n = 1 / wire_width                                          # [turns per m]
length_to_MOT_from_ZS = 100.6 / 1000                        # [m]


resistivity_Cu = 1.68 * 10**(-8)                            # [Ohm m]
cross_section = ((1/8 * 2.54 * 10**(-2))**2 
                  - (1/16 * 2.54 * 10**(-2))**2)            # [m^2]
resistance_per_length = resistivity_Cu/cross_section





