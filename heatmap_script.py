import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
import pickle
import os

import ideal_field as ideal

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


##############################################################################
# Analyzing detuning versus saturation

# Unpickle heatmap data
# Li
folder_location = os.path.join("C:\\", "Users","Lithium", "Documents", 
                                "zeeman_slower") 

li_file_name = "li_final_velocities_hc=77.5A_lc=35.5A.pickle"
li_file = os.path.join(folder_location, li_file_name)
li_final_velocities = pickle.load(open(li_file, "rb"))

plt.figure(figsize=(40, 40))

# Li
saturations = np.arange(1, 5.2, 0.2)
shift = 50 * 10**6
li_detunings = np.linspace(ideal.laser_detuning_li - shift, 
                            ideal.laser_detuning_li + shift, 51)

cutoff = 100
for i, vfinal in np.ndenumerate(li_final_velocities):
    if vfinal < 0:
        li_final_velocities[i[0]][i[1]] = -1000
    if vfinal < cutoff and vfinal > 0:
        li_final_velocities[i[0]][i[1]] = 0
    if vfinal > cutoff:
        li_final_velocities[i[0]][i[1]] = 1000

fig_li, ax_li = plt.subplots()
im_li = ax_li.imshow(li_final_velocities, vmin=-1000, vmax=1000)

ax_li.set_xticks(np.arange(len(saturations)))
ax_li.set_yticks(np.arange(len(li_detunings)))
ax_li.set_xticklabels(list(map(str, np.round(saturations, 2))), fontsize=6)
ax_li.set_yticklabels(list(map(str, np.round(li_detunings*10**(-6), 0))), fontsize=6)

# Rotate the tick labels and set their alignment.
plt.setp(ax_li.get_xticklabels(), rotation=45, ha="right",
          rotation_mode="anchor")

ax_li.set_title("Motion of Li atoms in ZS + comp coils (cutoff = 100 m/s, lc = 35.5 A, hc = 75.5 A)")
ax_li.set_ylabel("detuning (MHz)")
ax_li.set_xlabel("saturation")

fig_li.tight_layout()
fig_li.savefig(os.path.join(folder_location, 
                            "li_final_velocities_lc=35.5A_hc=77.5A.pdf"), 
                bbox_inches="tight")


# Unpickle heatmap data
# Er
folder_location = os.path.join("C:\\", "Users","Lithium", "Documents", 
                                "zeeman_slower") 

er_file_name = "er_final_velocities_hc=77.5A_lc=35.5A.pickle"
er_file = os.path.join(folder_location, er_file_name)
er_final_velocities_high_isat = pickle.load(open(er_file, "rb"))

plt.figure(figsize=(40, 40))

# Er
saturations = np.arange(1, 5.2, 0.2)
shift = 80 * 10**6
er_detunings = np.linspace(ideal.laser_detuning_er - shift, 
                          ideal.laser_detuning_er + shift, 81)

cutoff = 10
for i, vfinal in np.ndenumerate(er_final_velocities_high_isat):
    if vfinal < 0:
        er_final_velocities_high_isat[i[0]][i[1]] = -1000
    if vfinal < cutoff and vfinal > 0:
        er_final_velocities_high_isat[i[0]][i[1]] = 0
    if vfinal > cutoff:
        er_final_velocities_high_isat[i[0]][i[1]] = 1000

fig_er, ax_er = plt.subplots()
im_er = ax_er.imshow(er_final_velocities_high_isat, vmin=-1000, vmax=1000)


ax_er.set_xticks(np.arange(len(saturations)))
ax_er.set_yticks(np.arange(len(er_detunings)))
ax_er.set_xticklabels(list(map(str, np.round(saturations, 2))), fontsize=6)
ax_er.set_yticklabels(list(map(str, np.round(er_detunings*10**(-6), 0))), fontsize=6)

# Rotate the tick labels and set their alignment.
plt.setp(ax_er.get_xticklabels(), rotation=45, ha="right",
          rotation_mode="anchor")

ax_er.set_title("Motion of Er atoms in ZS + comp coils (cutoff = 10 m/s, lc = 35.5 A, hc = 77.5 A)")
ax_er.set_ylabel("detuning (MHz)")
ax_er.set_xlabel("saturation")

fig_er.tight_layout()
fig_er.savefig(os.path.join(folder_location, 
                            "er_final_velocities_lc=35.5A_hc=77.5A.pdf"), 
                bbox_inches="tight")


##############################################################################
# # Analyzing detuning versus saturation
# # Er zoomed in heatmaps

# folder_location = os.path.join("C:\\", "Users","Lithium", "Documents", 
#                                 "zeeman_slower")
# er_file_high_isat = os.path.join(folder_location, 
#                                   "er_final_velocities_high_isat.pickle")
# er_final_velocities_high_isat = pickle.load(open(er_file_high_isat, "rb"))


# shift = 80 * 10**6
# er_detunings = np.arange(ideal.laser_detuning_er - shift, 
#                           ideal.laser_detuning_er + shift, 81)
# saturations = np.arange(1, 5.2, 0.2)
# vcutoff = 5

# s_cutoff = len(saturations)
# d_cutoff_lower = 75
# d_cutoff_upper = 40
# er_high_isat_zoom = er_final_velocities_high_isat[d_cutoff_upper:d_cutoff_lower, 0:s_cutoff]
# binaries = np.zeros(np.shape(er_high_isat_zoom))

# vcutoff = 5
# for i, vfinal in np.ndenumerate(er_high_isat_zoom):
#     if vfinal < 0:
#         binaries[i[0]][i[1]] = -1000
#     if vfinal < vcutoff and vfinal > 0:
#         binaries[i[0]][i[1]] = 0
#     if vfinal > vcutoff:
#         binaries[i[0]][i[1]] = 1000


# plt.figure(figsize=(20, 40))
# fig, ax = plt.subplots()
# im = ax.imshow(binaries)


# ax.set_xticks(np.arange(len(saturations[0:s_cutoff])))
# ax.set_yticks(np.arange(len(er_detunings[d_cutoff_upper:d_cutoff_lower])))
# ax.set_xticklabels(list(map(str, np.round(saturations[0:s_cutoff], 2))), fontsize=6)
# ax.set_yticklabels(list(map(str, np.round(er_detunings[d_cutoff_upper:d_cutoff_lower]*10**(-6), 0))), fontsize=6)

# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#           rotation_mode="anchor")

# ax.set_title("Motion of Er atoms in ZS (ideal detuning = -1172 MHz, cutoff 5 m/s)")
# ax.set_ylabel("detuning (MHz)")
# ax.set_xlabel("saturation")


# fig.tight_layout()

# fig.savefig(os.path.join(folder_location, "er_final_velocities_binary_high_isat_zoom.pdf"), bbox_inches="tight")


##############################################################################
# # Analyzing low current versus high current

# # Unpickle heatmap data
# # Li
# folder_location = os.path.join("C:\\", "Users","Lithium", "Documents", 
#                                 "zeeman_slower") 
# li_file_name = "li_final_velocities_hc=190.0A_lc=35.0A_s_init=1_detuning=-1040MHz.pickle"
# li_file = os.path.join(folder_location, li_file_name)
# li_final_velocities = pickle.load(open(li_file, "rb"))

# # Parse file name to get values
# s = 1
# detuning = -1040

# plt.figure(figsize=(40, 40))

# high_currents = np.linspace(120, 190, 141)
# low_currents = np.linspace(25, 35, 21)

# cutoff = 100
# for i, vfinal in np.ndenumerate(li_final_velocities):
#     if vfinal < 0:
#         li_final_velocities[i[0]][i[1]] = -1000
#     if vfinal < cutoff and vfinal > 0:
#         li_final_velocities[i[0]][i[1]] = 0
#     if vfinal > cutoff:
#         li_final_velocities[i[0]][i[1]] = 1000

# fig_li, ax_li = plt.subplots()
# im_li = ax_li.imshow(li_final_velocities)

# ax_li.set_xticks(np.arange(len(low_currents)))
# ax_li.set_yticks(np.arange(len(high_currents)))
# ax_li.set_xticklabels(list(map(str, low_currents)), fontsize=6)
# ax_li.set_yticklabels(list(map(str, high_currents)), fontsize=6)

# # Rotate the tick labels and set their alignment.
# plt.setp(ax_li.get_xticklabels(), rotation=45, ha="right",
#           rotation_mode="anchor")

# ax_li.set_title("Motion of Li atoms in ZS + comp coils (cutoff = 100 m/s, detuning = -1040 MHz, s = 1)")
# ax_li.set_ylabel("high current (A)")
# ax_li.set_xlabel("low current (A)")

# fig_li.tight_layout()
# fig_li.savefig(os.path.join(folder_location, 
#                             "li_final_velocities.pdf"), 
#                 bbox_inches="tight")


# # Unpickle heatmap data
# # Er
# folder_location = os.path.join("C:\\", "Users","Lithium", "Documents", 
#                                 "zeeman_slower") 
# er_file_name = "er_final_velocities_hc=120.0A_lc=40.0A_s_init=1_detuning=-1172MHz.pickle"
# er_file = os.path.join(folder_location, er_file_name)
# er_final_velocities_high_isat = pickle.load(open(er_file, "rb"))

# plt.figure(figsize=(40, 40))

# # Er
# high_currents = np.linspace(50, 120, 141)
# low_currents = np.linspace(25, 40, 31)

# cutoff = 5
# for i, vfinal in np.ndenumerate(er_final_velocities_high_isat):
#     if vfinal < 0:
#         er_final_velocities_high_isat[i[0]][i[1]] = -1000
#     if vfinal < cutoff and vfinal > 0:
#         er_final_velocities_high_isat[i[0]][i[1]] = 0
#     if vfinal > cutoff:
#         er_final_velocities_high_isat[i[0]][i[1]] = 1000

# fig_er, ax_er = plt.subplots()
# im_er = ax_er.imshow(er_final_velocities_high_isat)

# ax_er.set_xticks(np.arange(len(low_currents)))
# ax_er.set_yticks(np.arange(len(high_currents)))
# ax_er.set_xticklabels(list(map(str, low_currents)), fontsize=6)
# ax_er.set_yticklabels(list(map(str, high_currents)), fontsize=6)

# # Rotate the tick labels and set their alignment.
# plt.setp(ax_er.get_xticklabels(), rotation=45, ha="right",
#           rotation_mode="anchor")

# ax_er.set_title("Motion of Er atoms in ZS + comp coils (cutoff = 5 m/s, detuning = -1172 MHz, s = 1)")
# ax_er.set_ylabel("high current (A)")
# ax_er.set_xlabel("low current (A)")

# fig_er.tight_layout()
# fig_er.savefig(os.path.join(folder_location, 
#                             "er_final_velocities_s_init=1_detuning=-1172MHz.pdf"), 
#                 bbox_inches="tight")
