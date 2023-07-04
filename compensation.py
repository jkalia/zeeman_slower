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
                         "zeeman_slower", "data_10.5.21")

###############
# Expected data
###############
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
# sections = [62, 98, 112]
sections = [62, 98, 112, 112]
# current_for_coils = [ 30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  30.8086634 ,  30.8086634 ,  30.8086634 ,
#                      30.8086634 ,  160        ,  160        ,  160        , 
#                      160        ,  160        ,  160]

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
                     30.8086634 ,  195        ,  195        ,  195        , 
                     195        ,  195        ,  195]

low_current_expected = -1 * current_for_coils[0]
high_current_expected = -1 * current_for_coils[-1]

slower_length = (len(coil_winding_total) * parameters.wire_width 
                 + parameters.wire_width * 1.5)
MOT_distance = slower_length + parameters.length_to_MOT_from_ZS
z = np.linspace(0, MOT_distance + .1, 10000)

total_field_total = winding.calculate_B_field_coil_gap(coil_winding_total, 
                                                       current_for_coils, z, 
                                                       sections)
print(winding.calculate_B_field_coil_gap(coil_winding_total, 
                                                       current_for_coils, np.array([MOT_distance]), 
                                                       sections))
total_field_lc = winding.calculate_B_field_coil_gap(coil_winding_lc, 
                                                    current_for_coils, z, 
                                                    sections)
total_field_hc = winding.calculate_B_field_coil_gap(coil_winding_hc, 
                                                    current_for_coils, z, 
                                                    sections)

zprime_total = np.gradient(total_field_total, z) * 0.01
zprime_lc = np.gradient(total_field_lc, z) * 0.01
zprime_hc = np.gradient(total_field_hc, z) * 0.01


###############
# Observed data for ZS
###############
position, background, lc, hc = \
    np.genfromtxt(os.path.join(file_path, "10.5.21_ZS_testing_data.csv"), 
                  dtype=float, delimiter=",", skip_header=1, unpack=True)
print(position)
position = (position*.01)-0.2516
obs_lc_B_field = -1*(lc-background)*30.81/2
obs_hc_B_field = -1*(hc-background)*195/2
obs_total_B_field = (-1*(lc-background)*30.81/2-1*(hc-background)*195/2)

obs_zprime_total = np.gradient(obs_total_B_field, position) * 0.01
obs_zprime_lc = np.gradient(obs_lc_B_field, position) * 0.01
obs_zprime_hc = np.gradient(obs_hc_B_field, position) * 0.01

low_current_observed = -1*30.81
high_current_observed = -1*195

###############
# Ideal data
###############
y = ideal.get_ideal_B_field(ideal.ideal_B_field, z)


# Plot of total B field
fig, ax = plt.subplots()

ax.plot(z, y, label="ideal B field", color="tab:orange")
ax.plot(z, total_field_total, label="calculated B field total", 
        color="royalblue")
ax.plot(z, total_field_lc, label="calculated B field lc", 
        color="cornflowerblue")
ax.plot(z, total_field_hc, label="calculated B field hc", 
        color="lightsteelblue")
ax.plot(position, obs_total_B_field, marker=".", color="indigo", linestyle="none",
        label="observed total B field")
ax.plot(position, obs_lc_B_field, marker=".", color="darkviolet", linestyle="none",
        label="observed lc B field")
ax.plot(position, obs_hc_B_field, marker=".", color="plum", linestyle="none",
        label="observed hc B field")



ax.axvline(x=MOT_distance, linestyle="--", color="k", 
           label="MOT location = {}".format(MOT_distance))
ax.set_title("Zeeman slower magnetic field (no compensation)")
ax.set_xlabel("Position (m)")
ax.set_ylabel("B field (Gauss)")

ax.legend()

fig.set_size_inches(12, 8)
# fig.savefig(os.path.join(file_path, "total_field_no_comp_v2.pdf"), 
#             bbox_inches="tight")


# Zoomed in plot of B field and gradient at MOT
fig1, ax1 = plt.subplots()

ax1.set_xlabel("Position (m)")
ax1.set_ylabel("B field (Gauss)", color="tab:red")


ax1.plot(position, obs_total_B_field, label="B field total", color="indianred")
ax1.plot(position, obs_lc_B_field, label="B field lc", color="lightcoral")
ax1.plot(position, obs_hc_B_field, label="B field hc", color="rosybrown")


ax1.plot(z, y, linestyle="--", color="tab:red")
ax1.axvline(x=MOT_distance, linestyle="--", color="k", 
            label="MOT location = {}".format(MOT_distance))
ax1.set_xlim(MOT_distance-0.02, MOT_distance+0.02)
ax1.set_ylim(-2, 10)
ax1.tick_params(axis="y", labelcolor="tab:red")

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel("Gradient total (Gauss/cm)", color="tab:blue")  # we already handled the x-label with ax1
ax2.plot(position, obs_zprime_total, label="total gradient", color="royalblue")
ax2.plot(position, obs_zprime_lc, label="lc gradient", color="cornflowerblue")
ax2.plot(position, obs_zprime_hc, label="hc gradient", color="lightsteelblue")
ax2.plot(z, np.gradient(total_field_total, z)*0.01, label="np.grad with z")

ax2.plot(z, y, linestyle="--", color="tab:blue")
ax2.set_ylim(-5, 1)
ax2.tick_params(axis="y", labelcolor="tab:blue")

ax1.legend(loc="lower left")
ax2.legend(loc="lower right")
ax1.set_title("Magnetic field and gradient at MOT position (no compensation)")
fig1.set_size_inches(12, 8)
fig1.tight_layout()
# fig1.savefig(os.path.join(file_path, "gradient_no_comp.pdf"), 
#               bbox_inches="tight")


# Plot the B field and gradient on the same figure
fig3, ax3 = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})

ax3[0].plot(z, total_field_total, label="calculated B field total", 
        color="royalblue")
ax3[0].plot(z, total_field_lc, label="calculated B field lc", 
        color="cornflowerblue")
ax3[0].plot(z, total_field_hc, label="calculated B field hc", 
        color="lightsteelblue")
ax3[1].plot(z, zprime_total, label="calculated total gradient", color="royalblue")
ax3[1].plot(z, zprime_lc, label="calculated lc gradient", color="cornflowerblue")
ax3[1].plot(z, zprime_hc, label="calculated hc gradient", color="lightsteelblue")

# ax3[0].plot(z, y, label="ideal B field", color="tab:orange")
# ax3[0].plot(position, obs_total_B_field, marker=".", color="indigo", 
#         label="observed total B field (no comp)")
# ax3[0].plot(position, obs_lc_B_field, marker=".", color="darkviolet", 
#         label="observed lc B field (no comp)")
# ax3[0].plot(position, obs_hc_B_field, marker=".", color="plum", 
#         label="observed hc B field (no comp)")
ax3[0].axvline(x=MOT_distance, linestyle="--", color="k", 
            label="MOT location = {}".format(MOT_distance))
ax3[0].set_ylabel("B field (Gauss)")
ax3[0].set_ylim(-10, 10)


ax3[1].plot(z, y, label="ideal B field", color="tab:orange")
# ax3[1].plot(position, obs_zprime_total, label="observed total gradient (no comp)", color="indigo")
# ax3[1].plot(position, obs_zprime_lc, label="observed lc gradient (no comp)", color="darkviolet")
# ax3[1].plot(position, obs_zprime_hc, label="observed hc gradient (no comp)", color="plum")
ax3[1].axvline(x=MOT_distance, linestyle="--", color="k", 
            label="MOT location = {}".format(MOT_distance))
ax3[1].set_ylabel("Gradient (Gauss/cm)")
ax3[1].set_xlabel("Position (m)")
ax3[1].set_xlim(MOT_distance-0.02, MOT_distance+0.02)
ax3[1].set_ylim(-5, 2)

ax3[0].plot(z,(7.14 - (1.7/0.01)*(z-MOT_distance)), color="r", 
            label="to compensate")
ax3[0].legend()
ax3[1].legend()
fig3.set_size_inches(24, 16)
# fig3.savefig(os.path.join(file_path, "field_and_gradient_compensation.pdf"), 
#               bbox_inches="tight")


###############
# Observed data for compensation coils
# Compensation coil dimensions are in Mathematica notebook
###############
file_path = os.path.join("C:\\", "Users", "Lithium", "Documents", 
                         "zeeman_slower", "data_02.03.22")

position1, position2, background1, coil1, background2, coil2 = \
    np.genfromtxt(os.path.join(file_path, "02.01.22_comp_coil_testing_data.csv"), 
                  dtype=float, delimiter=",", skip_header=1, unpack=True)
position1 = (position1*0.01)-0.061+MOT_distance-.055
position2 = (position2*0.01)-0.061+MOT_distance-.055
obs_coil_1 = -1*(coil1-background1)*95/5
obs_coil_2 = np.flip((coil2-background2)*47/5)

obs_total_comp = obs_coil_1 + -1*obs_coil_2

fig4, ax4 = plt.subplots()

ax4.plot(z,(7.14 - (1.7/0.01)*(z-MOT_distance)), color="r", 
         label="predicted compensation")
ax4.plot(position1, obs_total_comp, marker=".", linestyle="none",
         label="observed compensation")
ax4.axvline(x=MOT_distance, linestyle="--", color="k", 
            label="MOT location = {}".format(MOT_distance))
ax4.set_xlim(MOT_distance-0.02, MOT_distance+0.02)
ax4.set_ylim(-10, 10)
ax4.legend()
fig4.set_size_inches(12, 8)
# fig4.savefig(os.path.join(file_path, "compensation.pdf"), 
#               bbox_inches="tight")










