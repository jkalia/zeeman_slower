import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
from scipy import optimize
import random


# Iterate fixed_lengths from 4 to 10 
min_length = 4
max_length = 10

# Initialize array for storing data
rmse_array = np.zeros(((max_length - min_length + 1), 
                        np.ceil(max_length / 2).astype(int) + 1))
deviation_array = np.zeros(((max_length - min_length + 1), 
                             np.ceil(max_length / 2).astype(int) + 1))

# Use random to generate values
random.seed()


# # Iterate over fixed lengths
# for i in range(min_length, (max_length + 1), 1):
#   fixed_lengths[0] = i 

#   # Set max overlap
#   max_overlap = np.ceil(fixed_lengths[0] / 2).astype(int)

#   for j in range(max_overlap + 1):
#       fixed_overlap = j

#       rmse = random.randrange(0, 10)
#       li_deviation = random.randrange(0, 4)

#       rmse_array[(fixed_lengths[0] - min_length)][fixed_overlap] = rmse 
#       deviation_array[(fixed_lengths[0] - min_length)][fixed_overlap] = \
#           li_deviation

#       print("rmse_array: ", rmse_array)
#       print("deviation_array: ", deviation_array)


rmse_array = [[ 8.48746515,  9.97952066, 11.3569415,   0.,    0.,   0.        ],
 [10.97901425,  9.16825321,  8.86411585, 13.79452628,  0.,          0.        ],
 [ 8.17865624,  8.88781443,  7.85443844, 11.30155081,  0.,          0.        ],
 [11.60547799,  8.96410279,  7.49592904,  9.80667831, 13.51222824,  0.        ],
 [ 8.36018651,  9.14130748,  9.50929753,  8.8161949,  11.41697786,  0.        ],
 [ 6.95144548,  8.17007378,  9.42269643,  8.85713327, 11.06761548, 13.55682791],
 [ 8.56179801,  7.6615222,   9.33659978,  8.12664137,  9.83040328, 12.57638557]]


# deviation_array = [[ 4.54597424,  4.75864282,  8.58131858,  0.,  0., 0.       ],
#  [ 3.96873077,  4.14070563,  6.43849042, 11.1371103,   0.,          0.        ],
#  [ 4.96584097,  3.94280168,  5.03926655,  9.06132087,  0.,          0.        ],
#  [ 3.9444856,   3.93689076,  4.55087186,  7.27504545, 11.37164212,  0.        ],
#  [ 4.92221234,  3.92720894,  3.1406602,   6.34469767, 10.10956176,  0.        ],
#  [ 4.25699949,  3.47827248,  3.03221774,  6.5989776,   9.39207263, 11.29648746],
#  [ 3.94987697,  3.49923862,  3.00652533,  5.83917244,  8.3443662,  10.60359943]]


deviation_array = ([[0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0.]])


fig_rmse, ax_rmse = plt.subplots()
fig_dev, ax_dev = plt.subplots()

im_rmse = ax_rmse.imshow(rmse_array)
im_dev = ax_dev.imshow(deviation_array)


# Loop over data dimensions and create text annotations
for i in range(max_length + 1 - min_length):
    for j in range(np.ceil(max_length / 2).astype(int) + 1):
        text = ax_rmse.text(j, i, np.round(rmse_array[i][j], 2),
                       ha="center", va="center", color="w")
        text = ax_dev.text(j, i, np.round(deviation_array[i][j], 2),
                       ha="center", va="center", color="w")

y_labels = ["3", "4", "5", "6", "7", "8", "9", "10"]

ax_rmse.set_yticklabels(y_labels)
ax_dev.set_yticklabels(y_labels)

ax_rmse.set_title("RMSE")
ax_rmse.set_ylabel("fixed length")
ax_rmse.set_xlabel("fixed overlap")
fig_rmse.tight_layout()

ax_dev.set_title("max Li deviation")
ax_dev.set_ylabel("fixed length")
ax_dev.set_xlabel("fixed overlap")
fig_rmse.tight_layout()



plt.show()














