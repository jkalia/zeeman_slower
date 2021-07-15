# Jasmine Kalia
# June 18th, 2021
# plotting.py
# Module for zeeman_slower_configuration
# Contains all functions and variables relevant to creating figures.

# I think what I am supposed to do is to init the figures and axes in the main 
# file and then only write the functions which act on the axes here

import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 

import ideal_field as ideal
import parameters

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def give_high_current_section(current_for_coils):
    low_current = current_for_coils[0]
    for i, current in np.ndenumerate(current_for_coils):
        if current != low_current:
            return i[0] * parameters.wire_width + parameters.wire_width / 2
    return 0


def add_vertical_line(axis, position, **plt_kwargs):
    axis.axvline(x=position, **plt_kwargs)
    return axis


def denote_regions(axis, discretization, coil_winding, write=True):
    for i in range(len(coil_winding)):
        if i < len(coil_winding) - 1:
            if coil_winding[i] != coil_winding[i+1]:
                add_vertical_line(axis, 
                                  discretization[i] + parameters.wire_width/2)
                if write:
                    if coil_winding[i] == 1/3:
                        axis.text(x=(discretization[i] 
                                     - parameters.wire_width*2), 
                                  y=(600-50*coil_winding[i]), 
                                  s=("{:.2f}".format(coil_winding[i])))
                    else:
                        axis.text(x=(discretization[i] 
                                     - parameters.wire_width*3), 
                                  y=(700-50*coil_winding[i]), 
                                  s=("{:.2f}".format(coil_winding[i])))
    return 


def plot_initial(axis, x, x_long, y, discretization, ideal_field, 
                 solenoid_field_initial, current_for_coils_initial, 
                 total_field_initial, fixed_overlap):
    axis.plot(x, y, color="m", linestyle="-", 
              label="ideal B field to optimize to")
    axis.plot(x_long, ideal.get_ideal_B_field(ideal.ideal_B_field, x_long), 
         color="r", linestyle="-", label="ideal B field")
    axis.plot(discretization, ideal_field, color="b", marker=".", 
              label="discretized adjusted ideal B field")
    axis.plot(x_long, solenoid_field_initial, color="g", linestyle="-", 
              label="initial guess solenoid")
    axis.plot(discretization, total_field_initial, color="g", marker=".", 
              linestyle="None", label="initial guess coil winding")
    add_vertical_line(axis, 
                      give_high_current_section(current_for_coils_initial))
    axis.set_xlabel("Position (m)")
    axis.set_ylabel("B field (G)")
    axis.set_title(label="fixed overlap = {}".format(fixed_overlap))
    axis.legend()
    return 


def plot_lines(axis, x, x_long, y, discretization, solenoid_field_initial, 
               total_field_initial, solenoid_field_final, coil_winding_final, 
               current_for_coils_final, total_field_final, title):
    axis.plot(x, y , color="m", linestyle="--", 
         label="ideal B field to optimize to")
    axis.plot(x_long, solenoid_field_initial, color="g", linestyle="-", 
         label="initial guess solenoid")
    axis.plot(discretization, total_field_initial, color="g",
         marker=".", label="initial guess coil winding")
    axis.plot(x_long, solenoid_field_final, color="k", linestyle="-", 
         label="solenoid optimization result")
    axis.plot(discretization, total_field_final, color="k",
         marker=".", linestyle="None", 
         label="np.round() optimized coil winding")
    add_vertical_line(axis, give_high_current_section(current_for_coils_final))
    add_vertical_line(axis, ideal.slower_length_val, color="m")
    denote_regions(axis, discretization, coil_winding_final)
    axis.set_xlabel("Position (m)")
    axis.set_ylabel("B field (G)")
    axis.set_title(label=title)
    return 


def plot_slower_top(axis, x, x_long, y, discretization, solenoid_field_final, 
                    coil_winding_final, current_for_coils_final, 
                    total_field_final):
    axis.plot(x, y, color="m", linestyle="--", 
              label="ideal B field to optimize to")
    axis.plot(x_long, solenoid_field_final, color="k", linestyle="-", 
              label="solenoid optimization result")
    axis.plot(discretization, total_field_final, color="k", marker=".", 
              linestyle="None", label="np.round() optimized coil winding")
    add_vertical_line(axis, give_high_current_section(current_for_coils_final), 
                      color="k")
    add_vertical_line(axis, ideal.slower_length_val, color="m")
    denote_regions(axis, discretization, coil_winding_final, write=False)
    axis.set_ylabel("B field (G)")
    return 


def plot_slower_bottom(axis, discretization, coil_winding_final, 
                       current_for_coils_final):
    axis.bar(discretization, coil_winding_final, parameters.wire_width, 
           edgecolor='black', color='none')
    add_vertical_line(axis, give_high_current_section(current_for_coils_final), 
                      color="k")
    add_vertical_line(axis, ideal.slower_length_val, color="m")
    denote_regions(axis, discretization, coil_winding_final, write=False)
    axis.set_xlabel("Position (m)")
    axis.set_ylabel("Coil winding")
    axis.set_yticks(np.arange(0, 7.5, step=0.5))
    axis.grid(b=True, axis="y")
    return 


def plot_slower(axis_top, axis_bottom, x, x_long, y, discretization, 
                solenoid_field_final, coil_winding_final, 
                current_for_coils_final, total_field_final):
    plot_slower_top(axis_top, x, x_long, y, discretization, 
                    solenoid_field_final, coil_winding_final, 
                    current_for_coils_final, total_field_final)
    plot_slower_bottom(axis_bottom, discretization, coil_winding_final, 
                       current_for_coils_final)
    return


def plot_diff_top(axis, x, x_long, y, discretization, solenoid_field_final, 
                  coil_winding_final, current_for_coils_final, 
                  total_field_final):
    axis.plot(x, y, color="m", linestyle="--", 
              label="ideal B field to optimize to")
    axis.plot(x_long, solenoid_field_final, color="k", linestyle="-", 
              label="solenoid optimization result")
    axis.plot(discretization, total_field_final, color="k", marker=".", 
              linestyle="None", label="np.round() optimized coil winding")
    add_vertical_line(axis, give_high_current_section(current_for_coils_final), 
                      color="k")
    add_vertical_line(axis, ideal.slower_length_val, color="m")
    denote_regions(axis, discretization, coil_winding_final)
    axis.set_ylabel("B field (G)")
    return 


def plot_diff_bottom(axis, discretization, ideal_field, coil_winding_final, 
                     current_for_coils_final, total_field_final, 
                     B_field_range):
    axis.plot(discretization[0:B_field_range], 
              (total_field_final[0:B_field_range] 
               - ideal_field[0:B_field_range]) * 10**(-4) 
               * ideal.mu0_er / ideal.hbar / ideal.linewidth_er, label="Er")
    axis.plot(discretization[0:B_field_range], 
              (total_field_final[0:B_field_range] 
              - ideal_field[0:B_field_range]) * 10**(-4) 
              * ideal.mu0_li / ideal.hbar / ideal.linewidth_li, label="Li")
    add_vertical_line(axis, give_high_current_section(current_for_coils_final), 
                      color="k")
    add_vertical_line(axis, ideal.slower_length_val, color="m")
    denote_regions(axis, discretization, coil_winding_final, write=False)
    axis.set_xlabel("Position (m)")
    axis.set_ylabel("frequency shift / linewidth")
    axis.legend()
    return 


def plot_diff(axis_top, axis_bottom, x, x_long, y, discretization, ideal_field, 
              solenoid_field_final, coil_winding_final, 
              current_for_coils_final, total_field_final, B_field_range):
    plot_diff_top(axis_top, x, x_long, y, discretization, solenoid_field_final, 
                  coil_winding_final, current_for_coils_final, 
                  total_field_final)
    plot_diff_bottom(axis_bottom, discretization, ideal_field, coil_winding_final, 
                     current_for_coils_final, total_field_final, 
                     B_field_range)
    return


def plot_simple(axis, x, y, discretization, total_field_final, title):
    axis.plot(x, y , color="m", linestyle="--", 
         label="ideal B field to optimize to")
    axis.plot(discretization, total_field_final, color="k",
         marker=".", label="optimized B field")
    axis.set_xlabel("Position (m)")
    axis.set_ylabel("B field (G)")
    axis.set_title(label=title)
    return 


def make_plots(x, x_long, y, discretization, ideal_field, 
               solenoid_field_initial, coil_winding_initial, 
               current_for_coils_initial, total_field_initial, 
               solenoid_field_final, coil_winding_final, 
               current_for_coils_final, total_field_final, fixed_overlap, 
               B_field_range, title, file_path):


    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    fig4, ax4 = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    fig5, ax5 = plt.subplots()

    plot_initial(ax1, x, x_long, y, discretization, ideal_field, 
                 solenoid_field_initial, current_for_coils_initial, 
                 total_field_initial, fixed_overlap)
    plot_lines(ax2, x, x_long, y, discretization, solenoid_field_initial, 
               total_field_initial, solenoid_field_final, coil_winding_final, 
               current_for_coils_final, total_field_final, title)
    plot_slower(ax3[0], ax3[1], x, x_long, y, discretization, 
                solenoid_field_final, coil_winding_final, 
                current_for_coils_final, total_field_final)
    plot_diff(ax4[0], ax4[1], x, x_long, y, discretization, ideal_field, 
              solenoid_field_final, coil_winding_final, current_for_coils_final, 
              total_field_final, B_field_range)
    plot_simple(ax5, x, y, discretization, total_field_final, title)


    fig1.set_size_inches(12, 8)
    fig2.set_size_inches(12, 8)
    fig3.suptitle(t=title)
    fig3.set_size_inches(12, 8)
    fig4.suptitle(t=title)
    fig4.set_size_inches(12, 8)
    fig5.set_size_inches(12, 8)

    fig1.savefig(file_path + "/initial.pdf", bbox_inches="tight")
    fig2.savefig(file_path + "/lines.pdf", bbox_inches="tight")
    fig3.savefig(file_path + "/slower.pdf", bbox_inches="tight")
    fig4.savefig(file_path + "/diff.pdf", bbox_inches="tight")
    fig5.savefig(file_path + "/simple.pdf", bbox_inches="tight")

    return 



