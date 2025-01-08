# Jasmine Kalia
# July 9th, 2021
# atom.py
# Creates an Atom class which stores the variables for an erbium atom or for a 
# lithium atom.


import ideal_field as ideal


class Atom:
    def __init__(self, atom_type):
        if atom_type == "Er":
            self.m = ideal.m_er
            self.linewidth = ideal.linewidth_er
            self.k = ideal.k_er
            self.mu0 = ideal.mu0_er
            self.Isat = ideal.Isat_er
        elif atom_type == "Li":
            self.m = ideal.m_li
            self.linewidth = ideal.linewidth_li
            self.k = ideal.k_li
            self.mu0 = ideal.mu0_li
            self.Isat = ideal.Isat_li_d2

