# Jasmine Kalia
# August 2nd, 2021
# Zeeman slower compensation code     

import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
import os

import ideal_field as ideal
import coil_configuration as coil
import parameters
import winding

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


file_path = os.path.join("C:\\", "Users", "Lithium", "Documents", 
                          "zeeman_slower", "3.5mm", "optimization_plots", 
                          "19sections_6hclength_2hcmaxdensity_0overlap"
                          "_0.4860eta_current_only", 
                          "figs")

coil_winding_total = [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.25, 0.25, 
                      0.25, 0.25, 0.25, 0.25, 0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 
                      0.5 , 0.5 , 0.5 , 0.5 , 1.  , 1.  , 1.  , 0.5 , 0.5 , 
                      1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 
                      1.25, 1.25, 1.25, 1.25, 1.25, 1.5 , 1.5 , 1.5 , 1.5 , 
                      1.5 , 1.5 , 1.5 , 1.5 , 1.5 , 1.5 , 2.  , 2.  , 2.  , 
                      2.  , 2.  , 2.  , 2.  , 2.  , 2.  , 2.  , 2.  , 2.  , 
                      3   , 3   , 2.5 , 2.5 , 2.5 , 2.5 , 2.5 , 2.5 , 2.5 , 
                      2.5 , 3.  , 3.  , 3.  , 3.  , 3.  , 3.  , 3.  , 3.  , 
                      3.  , 3.  , 3.5 , 3.5 , 3.5 , 3.5 , 3.5 , 3.5 , 3.5 , 
                      3.5 , 4.  , 4.  , 4.  , 4.  , 4.  , 4.  , 4.  , 4.  ,
                      6   , 6   , 6   , 4.5 , 4.5 , 4.5 , 4.5 , 7.  , 7.  ,          
                      7.  , 7.  , 7.  , 7.  , 7.  , 2.  , 2.  , 2.  , 2.  , 
                      2.  , 2.  ]
coil_winding_lc = [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.25, 0.25, 0.25, 
                   0.25, 0.25, 0.25, 0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 
                   0.5 , 0.5 , 1.  , 1.  , 1.  , 0.5 , 0.5 , 1.  , 1.  , 1.  , 
                   1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.25, 1.25, 1.25, 1.25, 
                   1.25, 1.5 , 1.5 , 1.5 , 1.5 , 1.5 , 1.5 , 1.5 , 1.5 , 1.5 , 
                   1.5 , 2.  , 2.  , 2.  , 2.  , 2.  , 2.  , 2.  , 2.  , 2.  , 
                   2.  , 2.  , 2.  ,  3  , 3   , 2.5 , 2.5 , 2.5 , 2.5 , 2.5 , 
                   2.5 , 2.5 , 2.5 , 3.  , 3.  , 3.  , 3.  , 3.  , 3.  , 3.  , 
                   3.  , 3.  , 3.  , 3.5 , 3.5 , 3.5 , 3.5 , 3.5 , 3.5 , 3.5 , 
                   3.5 , 4.  , 4.  , 4.  , 4.  , 4.  , 4.  , 4.  , 4.  , 6   , 
                   6   , 6   , 4.5 , 4.5 , 4.5 , 4.5 , 7.  , 7.  , 7.  , 7.  , 
                   7.  , 7.  , 7.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]
coil_winding_hc = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
                   0., 0., 0., 0., 0., 0., 0., 0., 2., 2., 2., 2., 2., 2.]
sections = [62, 98, 112]

current_for_coils = [ 30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
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
                     30.8086634 ,  160        ,  160        ,  160        , 
                     160        ,  160        ,  160]

low_current = -1 * current_for_coils[0]
high_current = -1 * current_for_coils[-1]

slower_length = (len(coil_winding_total) * parameters.wire_width 
                 + parameters.wire_width * 1.5)
MOT_distance = slower_length + parameters.length_to_MOT_from_ZS

z = np.linspace(0, MOT_distance + .1, 10000)
y = ideal.get_ideal_B_field(ideal.ideal_B_field, z)

total_field_total = winding.calculate_B_field_coil_gap(coil_winding_total, 
                                                       current_for_coils, z, 
                                                       sections)
total_field_lc = winding.calculate_B_field_coil_gap(coil_winding_lc, 
                                                    current_for_coils, z, 
                                                    sections)
total_field_hc = winding.calculate_B_field_coil_gap(coil_winding_hc, 
                                                    current_for_coils, z, 
                                                    sections)

zprime_total = np.gradient(total_field_total)
zprime_lc = np.gradient(total_field_lc)
zprime_hc = np.gradient(total_field_hc)


# Plot of total B field
fig, ax = plt.subplots()

ax.plot(z, y, label="ideal B field", color="tab:orange")
ax.plot(z, total_field_total, label="calculated B field total", 
        color="royalblue")
ax.plot(z, total_field_lc, label="calculated B field lc", 
        color="cornflowerblue")
ax.plot(z, total_field_hc, label="calculated B field hc", 
        color="lightsteelblue")

ax.axvline(x=MOT_distance, linestyle="--", color="k", 
           label="MOT location = {}".format(MOT_distance))
ax.set_title("Zeeman slower magnetic field (no compensation)")
ax.set_xlabel("Position (m)")
ax.set_ylabel("B field (Gauss)")
ax.legend()

fig.set_size_inches(12, 8)
fig.savefig(os.path.join(file_path, "total_field_no_comp.pdf"), 
            bbox_inches="tight")


# Zoomed in plot of B field and gradient at MOT
fig1, ax1 = plt.subplots()

ax1.set_xlabel("Position (m)")
ax1.set_ylabel("B field (Gauss)", color="tab:red")
ax1.plot(z, total_field_total, label="B field total", color="indianred")
ax1.plot(z, total_field_lc, label="B field lc", color="lightcoral")
ax1.plot(z, total_field_hc, label="B field hc", color="rosybrown")
ax1.plot(z, y, linestyle="--", color="k")
ax1.axvline(x=MOT_distance, linestyle="--", color="k", 
            label="MOT location = {}".format(MOT_distance))
ax1.set_xlim(MOT_distance-0.02, MOT_distance+0.02)
ax1.set_ylim(-2, 8)
ax1.tick_params(axis="y", labelcolor="tab:red")

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel("Gradient total (Gauss/cm)", color="tab:blue")  # we already handled the x-label with ax1
ax2.plot(z, zprime_total*100, label="total gradient", color="royalblue")
ax2.plot(z, zprime_lc*100, label="lc gradient", color="cornflowerblue")
ax2.plot(z, zprime_hc*100, label="hc gradient", color="lightsteelblue")
ax2.set_ylim(-2.5, 1)
ax2.tick_params(axis="y", labelcolor="tab:blue")

ax1.legend(loc="lower left")
ax2.legend(loc="lower right")
ax1.set_title("Magnetic field and gradient at MOT position (no compensation)")
fig1.set_size_inches(12, 8)
fig1.tight_layout()
fig1.savefig(os.path.join(file_path, "gradient_no_comp.pdf"), 
             bbox_inches="tight")



# Now, we start adding in compensation coils. These coils will be rectangular
# and will be used to zero the residual B field and gradient at the MOT.
# The locations of these coils are constrained by the physical limits of the 
# experiment design. 
# We will reference the compensation coils off of the MOT location. 
# Absolute minimum clearance in the vertical axis is 170mm

coil1_hc = coil.B_total_rect_coil(high_current, 135 / 1000, 
                                  (500 / 2) / 1000, 
                                  MOT_distance - (65 / 1000), z)
coil2_hc = coil.B_total_single_coil(-1 * high_current, 
                                    65 / 1000 + parameters.wire_width / 2, 
                                    MOT_distance + (110 / 1000) + parameters.wire_width / 2, z)

# TODO: try 2 on pump tower side and still only 1 on ZS side 



position1 = 0
position2 = position1 + parameters.wire_height

coil1_lc = coil.B_total_rect_coil(low_current * 0, 160 / 1000, 
                                  (450 / 2) / 1000, 
                                  MOT_distance - (65 / 1000), z)
coil2_lc = coil.B_total_single_coil(-1 * low_current * 0, 
                                    60 / 1000 + parameters.wire_width / 2, 
                                    MOT_distance + (113 / 1000) + parameters.wire_width / 2, z)


comp_lc = coil2_lc + coil1_lc
B_field = total_field_total + comp_lc
B_field_lc = total_field_lc + comp_lc


comp_hc = coil2_hc + coil1_hc
B_field = total_field_total + comp_hc
B_field_hc = total_field_hc + comp_hc

zprime_comp = np.gradient(B_field)
zprime_comp_lc = np.gradient(B_field_lc)
zprime_comp_hc = np.gradient(B_field_hc)



# Plot of total B field
fig2, ax2 = plt.subplots()

ax2.plot(z, y, label="ideal B field", color="tab:orange")
ax2.plot(z, total_field_total, label="calculated B field (no comp)", 
        color="royalblue")
ax2.plot(z, total_field_lc, label="calculated B field lc (no comp)", 
        color="cornflowerblue")
ax2.plot(z, total_field_hc, label="calculated B field hc (no comp)", 
          color="lightsteelblue")

ax2.plot(z, B_field, label="calculated B field", color="brown")
ax2.plot(z, coil1_hc, label="hc comp coil 1", color="green")
ax2.plot(z, coil2_hc, label="hc comp coil 2", color="red")

ax2.axvline(x=MOT_distance, linestyle="--", color="k", 
            label="MOT location = {}".format(MOT_distance))
ax2.set_title("Zeeman slower magnetic field (no compensation)")
ax2.set_xlabel("Position (m)")
ax2.set_ylabel("B field (Gauss)")
ax2.legend()

fig2.set_size_inches(12, 8)
fig2.savefig(os.path.join(file_path, "total_field_comp.pdf"), 
              bbox_inches="tight")



# Zoomed in plot of B field and gradient at MOT
fig3, ax3 = plt.subplots()

ax3.set_xlabel("Position (m)")
ax3.set_ylabel("B field (Gauss)", color="tab:red")
ax3.plot(z, total_field_total, label="B field total (no comp)", color="indianred")
ax3.plot(z, total_field_lc, label="B field lc (no comp)", color="lightcoral")
ax3.plot(z, total_field_hc, label="B field hc (no comp)", color="rosybrown")

ax3.plot(z, B_field, label="calculated B field", color="brown")
ax3.plot(z, B_field_hc, label="B field hc", color="rosybrown")
ax3.plot(z, comp_hc, label="hc comp coils", color="orange")
# ax3.plot(z, coil2_hc, label="hc comp coil 2", color="orange")

ax3.plot(z, y, linestyle="--", color="k")
ax3.axvline(x=MOT_distance, linestyle="--", color="k", 
            label="MOT location = {}".format(MOT_distance))
ax3.set_xlim(MOT_distance-0.02, MOT_distance+0.02)
ax3.set_ylim(-10, 10)
ax3.tick_params(axis="y", labelcolor="tab:red")

ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis
ax4.set_ylabel("Gradient total (Gauss/cm)", color="tab:blue")  # we already handled the x-label with ax1
ax4.plot(z, zprime_total*100, label="total gradient (no comp)", color="royalblue")
ax4.plot(z, zprime_lc*100, label="lc gradient (no comp)", color="cornflowerblue")
ax4.plot(z, zprime_hc*100, label="hc gradient (no comp)", color="lightsteelblue")

ax4.plot(z, zprime_comp*100, label="total gradient", color="purple")
# ax4.plot(z, zprime_comp_lc*100, label="lc gradient", color="cornflowerblue")
ax4.plot(z, zprime_comp_hc*100, label="hc gradient", color="lightsteelblue")

ax4.set_ylim(-5, 5)
ax4.tick_params(axis="y", labelcolor="tab:blue")

ax3.legend(loc="lower left")
ax4.legend(loc="lower right")
ax3.set_title("Magnetic field and gradient at MOT position (no compensation)")
fig3.set_size_inches(12, 8)
fig3.tight_layout()
fig3.savefig(os.path.join(file_path, "gradient_comp.pdf"), 
             bbox_inches="tight")



# Zoomed in plot of B field and gradient at MOT
# Cleaner plot 
fig5, ax5 = plt.subplots()

ax5.set_xlabel("Position (m)")
ax5.set_ylabel("B field (Gauss)", color="tab:red")
ax4 = ax5.twinx()
ax4.set_ylabel("Gradient total (Gauss/cm)", color="tab:blue")
ax5.plot(z, y, linestyle="--", color="k")
ax5.axvline(x=MOT_distance, linestyle="--", color="k", 
            label="MOT location = {}".format(MOT_distance))


# hc set
# ax5.plot(z, total_field_hc, label="B field hc (no comp)", color="rosybrown")
# ax4.plot(z, zprime_hc*100, label="hc gradient (no comp)", color="lightsteelblue")
ax5.plot(z, B_field_hc , label="B field hc", color="purple")
ax5.plot(z, coil1_hc , label="coil 1 hc", color="salmon")
ax5.plot(z, coil2_hc , label="coil 2 hc", color="red")
ax4.plot(z, zprime_comp_hc*100, label="hc gradient", color="blue")


# # lc set
# # ax5.plot(z, total_field_lc, label="B field lc (no comp)", color="rosybrown")
# # ax4.plot(z, zprime_lc*100, label="lc gradient (no comp)", color="cornflowerblue")
# ax5.plot(z, B_field_lc , label="B field lc", color="purple")
# ax5.plot(z, coil1_lc , label="coil 1 lc", color="salmon")
# ax5.plot(z, coil2_lc , label="coil 2 lc", color="red")
# ax4.plot(z, zprime_comp_lc*100, label="lc gradient", color="cornflowerblue")


# # total set
# ax5.plot(z, total_field_total, label="B field total (no comp)", color="indianred")
# ax4.plot(z, zprime_total*100, label="total gradient (no comp)", color="royalblue")




ax5.set_xlim(MOT_distance-0.02, MOT_distance+0.02)
ax5.set_ylim(-10, 10)
ax5.tick_params(axis="y", labelcolor="tab:red")
ax4.set_ylim(-5, 5)
ax4.tick_params(axis="y", labelcolor="tab:blue")

ax5.legend(loc="lower left")
ax4.legend(loc="lower right")
ax5.set_title("Magnetic field and gradient at MOT position (no compensation)")
fig5.set_size_inches(12, 8)
fig5.tight_layout()
fig5.savefig(os.path.join(file_path, "gradient_comp_clean.pdf"), 
              bbox_inches="tight")
