import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
from scipy import optimize
import random


# Iterate fixed_lengths from 4 to 10 
min_length = 4
max_length = 10



# From run overnight on 6/21/21
# rmse_array = [[ 8.48746515,  9.97952066, 11.3569415,   0.,    0.,   0.        ],
#  [10.97901425,  9.16825321,  8.86411585, 13.79452628,  0.,          0.        ],
#  [ 8.17865624,  8.88781443,  7.85443844, 11.30155081,  0.,          0.        ],
#  [11.60547799,  8.96410279,  7.49592904,  9.80667831, 13.51222824,  0.        ],
#  [ 8.36018651,  9.14130748,  9.50929753,  8.8161949,  11.41697786,  0.        ],
#  [ 6.95144548,  8.17007378,  9.42269643,  8.85713327, 11.06761548, 13.55682791],
#  [ 8.56179801,  7.6615222,   9.33659978,  8.12664137,  9.83040328, 12.57638557]]


# # From run overnight on 6/23/21
# rmse_array = [
#  [ 6.83181508, 10.17353859, 10.05432199,  0.,          0.,          0.   ],
#  [ 8.11323506,  8.35562027,  8.82911397, 13.08085381,  0.,          0.   ],
#  [ 7.6966789,   6.54063395,  7.31258223, 10.74469601,  0.,          0.   ],
#  [ 7.38321057,  8.4697332,   7.22855785,  9.82670864, 14.24849468,  0.   ],
#  [ 7.73987144,  6.41848887,  8.06805693,  8.44028877, 11.20249809,  0.   ],
#  [ 7.77175933,  6.37434652,  6.44818484,  8.55515581, 10.53630675, 13.62142843],
#  [ 8.09062395,  6.34553508,  7.94144803, 10.67725029, 10.68908727, 12.52766468]
#              ]

# deviation_array = [
#  [ 4.41639875,  5.43204557, 17.60222496,  0.,          0.,          0.   ],
#  [ 4.99048158,  5.53090427, 12.25989711, 26.09838964,  0.,          0.   ],
#  [ 4.9747326,   4.68649303, 9.1437886,  18.04560498,  0.,          0.    ],
#  [ 5.01179301,  5.64475566,  7.66223134, 15.12308931, 25.51731128,  0.   ],
#  [ 4.98141638,  4.6851612,   6.93678641, 12.76334409, 20.12017164,  0.   ],
#  [ 4.97655129,  4.67912928,  5.95551045, 11.62787776, 18.03675851, 23.6968532 ],
#  [ 5.00390718,  4.65847359,  7.02942678, 11.50407231, 16.66917542, 21.44731769]
#                   ]


# fig_rmse, ax_rmse = plt.subplots()
# fig_dev, ax_dev = plt.subplots()

# im_rmse = ax_rmse.imshow(rmse_array)
# im_dev = ax_dev.imshow(deviation_array)


# # Loop over data dimensions and create text annotations
# for i in range(max_length + 1 - min_length):
#     for j in range(np.ceil(max_length / 2).astype(int) + 1):
#         text = ax_rmse.text(j, i, np.round(rmse_array[i][j], 2),
#                        ha="center", va="center", color="w")
#         text = ax_dev.text(j, i, np.round(deviation_array[i][j], 2),
#                        ha="center", va="center", color="w")

# y_labels = ["3", "4", "5", "6", "7", "8", "9", "10"]

# ax_rmse.set_yticklabels(y_labels)
# ax_dev.set_yticklabels(y_labels)

# ax_rmse.set_title("RMSE")
# ax_rmse.set_ylabel("fixed length")
# ax_rmse.set_xlabel("fixed overlap")
# fig_rmse.tight_layout()

# ax_dev.set_title("max Li deviation")
# ax_dev.set_ylabel("fixed length")
# ax_dev.set_xlabel("fixed overlap")
# fig_rmse.tight_layout()



# plt.show()



################################################################################
# # Trying to optimize solution more

# # Location to save data
# folder_location = os.path.join("C:\\", "Users","Erbium", "Documents", 
#                                "zeeman_slower", "heatmap1")

# # # Iterations for optimizer
# iterations = 10000

# # Arrays which defines the solenoid configuration for the low current section. 
# densities = [7, 6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1.25, 1, 0.5, 1, 
#              0.5, 0.25, 0]

# # Arrays which define the solenoid configuration for the high current section.
# fixed_densities = [2]
# fixed_lengths = [6]
# fixed_overlap = 0

# z = np.linspace(0, ideal.slower_length_val, 10000)
# y_data = ideal.get_ideal_B_field(ideal.ideal_B_field, z)
# guess = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 110, 35, 120]

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

#     fixed_lengths[0] = i

#     # Set max overlap
#     max_overlap = np.ceil(fixed_lengths[0] / 2).astype(int)
#     if max_overlap > 2:
#         max_overlap = 2

#     # Iterate over fixed_overlap
#     for j in range(max_overlap + 1):

#         flag = 0
#         flag_2 = 0
#         counter = 0
#         fixed_overlap = j

#         while (flag != 1) and (flag != 3):

#             print("fixed_lengths: ", fixed_lengths)
#             print("fixed_overlap: ", fixed_overlap)
#             print("counter: ", counter)

#             # Run optimization and collect data
#             rmse, li_deviation, flag, final = \
#                 run_optimization(fixed_densities, densities, fixed_lengths, 
#                                  fixed_overlap, z, y_data, guess, iterations,
#                                  folder_location, counter)

#             print("rmse: ", rmse)   
#             print("li_deviation: ", li_deviation)
#             guess = final
#             counter += 1

#             if flag == 2:
#                 flag_2 += 1
#             if flag_2 > 200:
#                 break

#         rmse_array[(fixed_lengths[0] - min_length)][fixed_overlap] = rmse 
#         deviation_array[(fixed_lengths[0] - min_length)][fixed_overlap] = \
#             li_deviation

#         print("rmse_array: ", rmse_array)
#         print("deviation_array: ", deviation_array)


# print("rmse_array: ", rmse_array)
# print("deviation_array: ", deviation_array)

# data = (rmse_array, deviation_array)
# save_data(data, "heatmap.pickle")


def make_heatmap(array, iter1, iter2, y_labels, title, ylabel, xlabel, 
                 file_path):

    fig, ax = plt.subplots()
    im = ax.imshow(array)

    # Loop over data dimensions and create text annotations
    for i in range(iter1):
        for j in range(iter2):
            print(i)
            print(j)
            text = ax.text(j, i, np.round(array[i][j], 2), ha="center", 
                           va="center", color="w")


    ax.set_yticklabels(y_labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.set_size_inches(12, 8)

    fig.savefig(file_path + "/heatmap.pdf", bbox_inches="tight")

    return











