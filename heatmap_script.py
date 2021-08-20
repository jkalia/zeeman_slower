import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
import os
# import zeeman_slower_configuration as zs

import ideal_field as ideal

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


# # Iterate fixed_lengths from 4 to 10 
# min_length = 4
# max_length = 10



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


rmse_array = [
 [4.261345,   6.45621162, 0.,         0.,         0.,         0.        ],
 [6.56767627, 5.02215333, 0.,         0.,         0.,         0.        ],
 [5.12067383, 6.67764858, 0.,         0.,         0.,         0.        ],
 [4.94217515, 5.9656794,  0.,         0.,         0.,         0.        ],
 [4.97210579, 4.99280766, 0.,         0.,         0.,         0.        ],
 [5.03272719, 4.71731003, 0.,         0.,         0.,         0.        ],
 [5.11554469, 4.67402702, 0.,         0.,         0.,         0.        ]]
deviation_array = [
 [ 5.85930167, 11.9326273,   0.,          0.,          0.,          0.      ],
 [ 4.52223978,  9.54256907,  0.,          0.,          0.,          0.      ],
 [ 4.53787603,  8.04629055,  0.,          0.,          0.,          0.      ],
 [ 3.98464181,  7.60117149,  0.,          0.,          0.,          0.      ],
 [ 3.67553848,  6.78164151,  0.,          0.,          0.,          0.      ],
 [ 3.50841342,  6.2725938,   0.,          0.,          0.,          0.      ],
 [ 3.4277186,   5.95421519,  0.,          0.,          0.,          0.      ]]
average_array = [
 [0.74683769, 0.92515479, 0.,         0.,         0.,         0.        ],
 [1.33414603, 0.69863461, 0.,         0.,         0.,         0.        ],
 [0.96934371, 1.24456255, 0.,         0.,         0.,         0.        ],
 [0.95763625, 1.13086844, 0.,         0.,         0.,         0.        ],
 [0.97493836, 0.89121708, 0.,         0.,         0.,         0.        ],
 [0.99262506, 0.84747737, 0.,         0.,         0.,         0.        ],
 [1.01295514, 0.85059986, 0.,         0.,         0.,         0.        ]]


fig_rmse, ax_rmse = plt.subplots()
fig_dev, ax_dev = plt.subplots()
fig_ave, ax_ave = plt.subplots()

im_rmse = ax_rmse.imshow(rmse_array)
im_dev = ax_dev.imshow(deviation_array)
im_ave = ax_ave.imshow(average_array)

min_length = 4
max_length = 10

# Loop over data dimensions and create text annotations
for i in range(max_length + 1 - min_length):
    for j in range(np.ceil(max_length / 2).astype(int) + 1):
        text = ax_rmse.text(j, i, np.round(rmse_array[i][j], 2),
                        ha="center", va="center", color="w")
        text = ax_dev.text(j, i, np.round(deviation_array[i][j], 2),
                        ha="center", va="center", color="w")
        text = ax_ave.text(j, i, np.round(average_array[i][j], 2),
                        ha="center", va="center", color="w")

y_labels = ["3", "4", "5", "6", "7", "8", "9", "10"]

ax_rmse.set_yticklabels(y_labels)
ax_dev.set_yticklabels(y_labels)
ax_ave.set_yticklabels(y_labels)

ax_rmse.set_title("RMSE")
ax_rmse.set_ylabel("fixed length")
ax_rmse.set_xlabel("fixed overlap")
fig_rmse.tight_layout()

ax_dev.set_title("max Li deviation")
ax_dev.set_ylabel("fixed length")
ax_dev.set_xlabel("fixed overlap")
fig_dev.tight_layout()

ax_ave.set_title("average Li deviation")
ax_ave.set_ylabel("fixed length")
ax_ave.set_xlabel("fixed overlap")
fig_ave.tight_layout()


file_path = os.path.join("C:\\", "Users", "Lithium", "Documents", 
                               "zeeman_slower", "3.6mm", "optimization_plots")
fig_rmse.savefig(os.path.join(file_path, "rmse.pdf"), bbox_inches="tight")
fig_dev.savefig(os.path.join(file_path, "dev.pdf"), bbox_inches="tight")
fig_ave.savefig(os.path.join(file_path, "ave.pdf"), bbox_inches="tight")



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


def make_heatmap(array, iter1, iter2, title, xlabel, ylabel, file_path, 
                 file_name):

    fig, ax = plt.subplots()
    im = ax.imshow(array)

    # Loop over data dimensions and create text annotations
    for i in range(iter1):
        for j in range(iter2):
            print(i)
            print(j)
            text = ax.text(j, i, np.round(array[i][j], 2), ha="center", 
                           va="center", color="w")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.set_size_inches(12, 8)

    file = os.path.join(file_path, file_name)

    fig.savefig(file, bbox_inches="tight")

    return



# li_final_velocities = [[630.56532683, 627.8709937 , 624.31368836, 504.26320876,
#          19.4332561 ,  18.75873297,  18.09811306,  17.44892083,
#          16.80897112,  16.17628987],
#        [629.08933975, 626.34015697, 622.6855701 , 498.96327583,
#          17.48501964,  16.76952318,  16.06556532,  15.37029331,
#          14.68108207,  13.99543588],
#        [627.56649876, 624.83973917, 621.09499238,  16.23381229,
#          15.45395402,  14.68519095,  13.92364404,  13.16562741,
#          12.40747691,  11.64537285],
#        [626.10188976, 623.27621924, 619.44404536,  14.15520017,
#          13.30868639,  12.46623656,  11.6222846 ,  10.77089531,
#           9.90530455,   9.01725485],
#        [624.58632855, 621.75982756, 617.76510819,  11.93734779,
#          10.99380016,  10.03872662,   9.06113807,   8.04668375,
#           6.97455529,   5.81052647],
#        [623.07938811, 620.24599131, 616.10574783,   9.4994068 ,
#           8.3933302 ,   7.23044549,   5.97086085,   4.53454715,
#           2.68593378, -14.31216112],
#        [621.60760171, 618.68462027, 614.44474765,   6.64301988,
#           5.16353823,   3.34545315,  -1.72411187, -14.02204638,
#         -14.23707439, -14.43512928],
#        [620.09840768, 617.13341715, 612.75766254,   2.38326142,
#         -13.07768906, -13.34771857, -13.56545587, -13.78225692,
#         -13.99969172, -14.21684943],
#        [618.57589867, 615.61129507, 611.03654182, -12.62372084,
#         -12.86185548, -13.10217357, -13.34273894, -13.58163537,
#         -13.81774062, -14.05047921],
#        [617.08344328, 614.0837781 , 609.28337012, -12.38484486,
#         -12.65161064, -12.91528288, -13.1744691 , -13.42849071,
#         -13.67713948, -13.92047056],
#        [615.60131835, 612.52763415, 607.49640148, -12.20703301,
#         -12.49308262, -12.77199091, -13.04358291, -13.3079552 ,
#         -13.56540709, -13.81634466],
#        [614.08851545, 610.95238667, 605.66541787, -12.07041118,
#         -12.36922303, -12.65858511, -12.93895893, -13.21085263,
#         -13.47484356, -13.73153941],
#        [612.5595996 , 609.39642509, 603.77787999, -11.96195997,
#         -12.26981914, -12.56681659, -12.85376716, -13.1314131 ,
#         -13.40049393, -13.66174066],
#        [611.08012246, 607.86870331, 601.81128304, -11.87395645,
#         -12.18860601, -12.49146943, -12.78357375, -13.0658039 ,
#         -13.33899957, -13.60397131],
#        [609.55245155, 606.27687718, 599.73213019, -11.8015129 ,
#         -12.12149531, -12.42905211, -12.72534628, -13.01135471,
#         -13.28798212, -13.5560927 ],
#        [608.06689524, 604.73741805, 597.47180312, -11.74134891,
#         -12.06567637, -12.37711555, -12.67692104, -12.96613182,
#         -13.24569663, -13.51651743],
#        [656.93739721, 655.5103186 , 653.40407591, -11.69116085,
#         -12.0191365 , -12.33387726, -12.63670121, -12.92869077,
#         -13.21082566, -13.48403547],
#        [656.46368281, 655.32736117, 654.24518101, 653.17442073,
#         652.09678802, 650.95623773, 649.53607395, -12.89793389,
#         -13.18235404, -13.45770576],
#        [655.71646613, 654.62698522, 653.60137608, 652.62687894,
#         651.69306092, 650.79075927, 649.91116264, 649.04474055,
#         648.16752821, 647.28564744],
#        [654.93407517, 653.8403827 , 652.81975345, 651.87101351,
#         650.95986754, 650.09036589, 649.26856993, 648.46405203,
#         647.68447391, 646.93835598]]

# # Actually Er
# li_final_velocities = [[382.58110133, 327.79305803,  53.5267461 ,  37.37541062,
#         -23.35732849, -23.31286365, -23.92316403, -24.54004936,
#         -25.14575802, -25.73793007],
#        [381.66456662, 326.95644591,  51.8515586 ,  34.06228105,
#         -22.94680336, -23.307205  , -23.92709569, -24.54719332,
#         -25.15490861, -25.74865058],
#        [380.70867814, 326.10412064,  50.093938  ,  30.17327811,
#         -22.80462949, -23.30480376, -23.93210151, -24.55506133,
#         -25.16466113, -25.75991031],
#        [379.70995847, 325.23477667,  48.23747797,  25.49913067,
#         -22.74026314, -23.30484049, -23.93806391, -24.56360799,
#         -25.17498821, -25.77168892],
#        [378.66226062, 324.34693257,  46.26128915,  18.9938032 ,
#         -22.70659101, -23.30678421, -23.94488973, -24.57279397,
#         -25.18586508, -25.78396763],
#        [377.56282558, 323.43892156,  44.13784643,   7.21661176,
#         -22.68790247, -23.31027461, -23.9525037 , -24.58258485,
#         -25.19726923, -25.79672903],
#        [376.40582934, 322.50792903,  41.82851918, -24.86077637,
#         -22.67767151, -23.31505745, -23.96084392, -24.59295025,
#         -25.20918006, -25.80995691],
#        [375.18226387, 321.55269613,  39.27864584, -22.84361784,
#         -22.67274277, -23.32094726, -23.96985872, -24.60386312,
#         -25.22157868, -25.82363618],
#        [373.88552933, 320.57070791,  36.40475275, -22.30119349,
#         -22.67141399, -23.32780474, -23.97950429, -24.6152992 ,
#         -25.23444768, -25.83775272],
#        [372.50305756, 319.55909635,  33.07226301, -22.14313056,
#         -22.67268766, -23.33552261, -23.9897431 , -24.6272366 ,
#         -25.24777095, -25.8522933 ],
#        [371.02156933, 318.51353249,  29.21787243, -22.07764323,
#         -22.67594021, -23.34401632, -24.00054257, -24.63965545,
#         -25.26153354, -25.8672455 ],
#        [369.42599578, 317.43100984,  24.43805214, -22.04558981,
#         -22.6807609 , -23.35321795, -24.01187416, -24.65253761,
#         -25.27572153, -25.88259764],
#        [367.69276848, 316.30683612,  17.61376968, -22.02909368,
#         -22.68686719, -23.36307189, -24.02371264, -24.66586646,
#         -25.29032192, -25.8983387 ],
#        [365.79220517, 315.13452848,   3.15768143, -22.02111738,
#         -22.69405752, -23.37353194, -24.03603552, -24.67962668,
#         -25.30532256, -25.91445827],
#        [363.69032424, 313.90826988, -24.35608011, -22.01836951,
#         -22.7021835 , -23.38455912, -24.04882263, -24.69380411,
#         -25.32071203, -25.93094652],
#        [361.34298305, 312.61900714, -22.01269803, -22.01911416,
#         -22.71113283, -23.39612013, -24.06205572, -24.70838562,
#         -25.33647961, -25.94779411],
#        [358.69840159, 311.25749747, -21.56241357, -22.02235085,
#         -22.7208185 , -23.40818618, -24.07571821, -24.72335896,
#         -25.35261517, -25.96499221],
#        [355.72139152, 309.80935342, -21.43528029, -22.02746216,
#         -22.73117159, -23.42073214, -24.08979495, -24.73871274,
#         -25.36910917, -25.98253239],
#        [352.42584402, 308.26015769, -21.38327381, -22.0340457 ,
#         -22.7421365 , -23.43373586, -24.10427204, -24.75443625,
#         -25.38595259, -26.00040665],
#        [348.93993997, 306.5879595 , -21.35886264, -22.04182724,
#         -22.75366758, -23.44717763, -24.11913663, -24.77051946,
#         -25.40313685, -26.01860736],
#        [345.50472449, 304.76386265, -21.34751758, -22.05061273,
#         -22.76572678, -23.46103981, -24.13437685, -24.78695292,
#         -25.42065385, -26.03712725],
#        [342.36175734, 302.74994919, -21.34342798, -22.06026039,
#         -22.77828193, -23.47530647, -24.14998166, -24.80372773,
#         -25.43849585, -26.05595935],
#        [339.62553842, 300.48943076, -21.34383858, -22.07066368,
#         -22.79130552, -23.48996317, -24.16594076, -24.82083548,
#         -25.4566555 , -26.075097  ],
#        [337.28702702, 297.89997642, -21.34727967, -22.08174064,
#         -22.80477371, -23.50499672, -24.18224453, -24.83826819,
#         -25.47512578, -26.09453383],
#        [335.28276909, 294.85140417, -21.35289651, -22.09342675,
#         -22.81866568, -23.52039501, -24.19888392, -24.8560183 ,
#         -25.49390001, -26.11426372],
#        [333.54026162, 291.12556696, -21.36015774, -22.10567029,
#         -22.83296303, -23.53614688, -24.21585045, -24.87407863,
#         -25.51297178, -26.1342808 ],
#        [331.99829521, 286.3394665 , -21.36871506, -22.11842908,
#         -22.84764941, -23.55224201, -24.23313611, -24.89244235,
#         -25.53233495, -26.15457941],
#        [330.61168014, 279.81943351, -21.37833004, -22.13166815,
#         -22.86271013, -23.5686708 , -24.25073333, -24.91110292,
#         -25.55198364, -26.17515412],
#        [329.34367504, 270.65332398, -21.38883339, -22.14535809,
#         -22.87813194, -23.58542428, -24.26863494, -24.93005412,
#         -25.57191221, -26.19599969],
#        [328.16853111, 258.20725694, -21.40010112, -22.15947389,
#         -22.89390281, -23.60249409, -24.28683413, -24.94928999,
#         -25.59211524, -26.21711106],
#        [327.06897418, 234.54355519, -21.41203995, -22.17399399,
#         -22.91001172, -23.61987235, -24.30532446, -24.96880483,
#         -25.61258749, -26.23848335],
#        [326.0272423 , -20.64827959, -21.424578  , -22.18889965,
#         -22.92644857, -23.63755166, -24.32409974, -24.98859316,
#         -25.63332394, -26.26011183],
#        [325.0340792 , -20.65436174, -21.43765873, -22.20417436,
#         -22.943204  , -23.65552504, -24.34315411, -25.00864971,
#         -25.65431972, -26.28199192],
#        [324.07970182, -20.66231557, -21.45123679, -22.21980353,
#         -22.96026938, -23.67378586, -24.36248194, -25.02896944,
#         -25.67557013, -26.3041192 ],
#        [323.15816221, -20.67167549, -21.46527524, -22.23577408,
#         -22.97763664, -23.69232786, -24.38207785, -25.04954747,
#         -25.69707065, -26.32648934],
#        [322.26197464, -20.68214908, -21.47974348, -22.25207426,
#         -22.99529824, -23.71114507, -24.40193668, -25.07037909,
#         -25.71881686, -26.34909818],
#        [321.38870889, -20.69353682, -21.49461581, -22.26869341,
#         -23.01324713, -23.73023182, -24.42205347, -25.09145977,
#         -25.74080451, -26.37194164],
#        [320.53381947, -20.70569658, -21.5098704 , -22.28562182,
#         -23.03147666, -23.74958269, -24.44242346, -25.11278511,
#         -25.76302944, -26.39501576],
#        [319.69446437, -20.71852379, -21.52548848, -22.30285058,
#         -23.04998055, -23.76919249, -24.46304204, -25.13435085,
#         -25.78548763, -26.41831666],
#        [318.86840755, -20.73193944, -21.54145372, -22.32037147,
#         -23.06875287, -23.78905626, -24.48390479, -25.15615288,
#         -25.80817516, -26.44184056]]

# plt.figure(figsize=(20, 20))
# shift = 20 * 10**6
# li_detunings = np.linspace(ideal.laser_detuning_li - shift, 
#                             ideal.laser_detuning_li + shift, 20)

# # Actually Er
# li_detunings = np.linspace(ideal.laser_detuning_er - shift, 
#                             ideal.laser_detuning_er + shift, 40)
# saturations = np.linspace(1, 2, 10)

# for i, vfinal in np.ndenumerate(li_final_velocities):
#     if vfinal < 0:
#         li_final_velocities[i[0]][i[1]] = -1000
#     if vfinal < 5 and vfinal > 0:
#         li_final_velocities[i[0]][i[1]] = 0
#     if vfinal > 5:
#         li_final_velocities[i[0]][i[1]] = 1000

# fig_li, ax_li = plt.subplots()
# im_li = ax_li.imshow(li_final_velocities)


# ax_li.set_xticks(np.arange(len(saturations)))
# ax_li.set_yticks(np.arange(len(li_detunings)))
# ax_li.set_xticklabels(list(map(str, np.round(saturations, 2))))
# ax_li.set_yticklabels(list(map(str, np.round(li_detunings, -6))))

# # Rotate the tick labels and set their alignment.
# plt.setp(ax_li.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")

# # Loop over data dimensions and create text annotations.
# for i in range(len(li_detunings)):
#     for j in range(len(saturations)):
#         text = ax_li.text(j, i, np.round(li_final_velocities[i][j]),
#                         ha="center", va="center", color="w", fontsize='2')

# ax_li.set_title("Motion of Er atoms in ZS (ideal detuning = -1172 MHz, cutoff 5 m/s)")
# ax_li.set_ylabel("detuning")
# ax_li.set_xlabel("saturation")


# fig_li.tight_layout()
# file_path = folder_location = os.path.join("C:\\", "Users","Lithium", "Documents", 
#                           "zeeman_slower", "figs", "er_final_velocities_binary.pdf")
# fig_li.savefig(file_path, bbox_inches="tight")



# folder_location = os.path.join("C:\\", "Users","Lithium", "Documents", 
#                                 "zeeman_slower", "figs")
# er_file_high_isat = os.path.join(folder_location, 
#                                   "er_final_velocities_high_isat.pickle")
# er_high_isat = zs.retrieve_heatmap_data(er_file_high_isat)

# shift = 100 * 10**6
# er_detunings = np.arange(ideal.laser_detuning_er - shift, 
#                           ideal.laser_detuning_er, 1 * 10**6)
# saturations = np.arange(1, 5.2, 0.2)
# vcutoff = 5

# binaries = np.zeros(np.shape(er_high_isat))

# for i, vfinal in np.ndenumerate(er_high_isat):
#     if vfinal < 0:
#         binaries[i[0]][i[1]] = -1000
#     if vfinal < vcutoff and vfinal > 0:
#         binaries[i[0]][i[1]] = 0
#     if vfinal > vcutoff:
#         binaries[i[0]][i[1]] = 1000


# plt.figure(figsize=(20, 20))
# fig, ax = plt.subplots()
# im = ax.imshow(binaries)


# ax.set_xticks(np.arange(len(saturations)))
# ax.set_yticks(np.arange(len(er_detunings)))
# ax.set_xticklabels(list(map(str, np.round(saturations, 2))))
# ax.set_yticklabels(list(map(str, np.round(er_detunings, -6))))

# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#           rotation_mode="anchor")

# # # Loop over data dimensions and create text annotations.
# # for i in range(len(er_detunings)):
# #     for j in range(len(saturations)):
# #         text = ax.text(j, i, np.round(er_high_isat[i][j]),
# #                         ha="center", va="center", color="w", fontsize='2')

# ax.set_title("Motion of Er atoms in ZS (ideal detuning = -1172 MHz, cutoff 5 m/s)")
# ax.set_ylabel("detuning")
# ax.set_xlabel("saturation")


# fig.tight_layout()
# file_path = folder_location = os.path.join("C:\\", "Users","Lithium", "Documents", 
#                           "zeeman_slower", "figs", "er_final_velocities_binary_high_isat.pdf")
# fig.savefig(file_path, bbox_inches="tight")




# folder_location = os.path.join("C:\\", "Users", "Lithium", "Documents", 
#                                "zeeman_slower", "figs")
# er_file_high_isat = os.path.join(folder_location, 
#                                  "er_final_velocities_high_isat.pickle")
# er_high_isat = zs.retrieve_heatmap_data(er_file_high_isat)

# shift = 100 * 10**6
# er_detunings = np.arange(ideal.laser_detuning_er - shift, 
#                           ideal.laser_detuning_er, 1 * 10**6)
# saturations = np.arange(1, 5.2, 0.2)

# s_cutoff = len(saturations)
# d_cutoff_lower = 100
# d_cutoff_upper = 80
# er_high_isat_zoom = er_high_isat[d_cutoff_upper:d_cutoff_lower, 0:s_cutoff]
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
# ax.set_xticklabels(list(map(str, np.round(saturations[0:s_cutoff], 2))))
# ax.set_yticklabels(list(map(str, np.round(er_detunings[d_cutoff_upper:d_cutoff_lower], -6))))

# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#           rotation_mode="anchor")

# # # Loop over data dimensions and create text annotations.
# # for i in range(len(er_detunings)):
# #     for j in range(len(saturations)):
# #         text = ax.text(j, i, np.round(er_high_isat[i][j]),
# #                         ha="center", va="center", color="w", fontsize='2')

# ax.set_title("Motion of Er atoms in ZS (ideal detuning = -1172 MHz, cutoff 5 m/s)")
# ax.set_ylabel("detuning")
# ax.set_xlabel("saturation")


# fig.tight_layout()
# file_path = folder_location = os.path.join("C:\\", "Users","Lithium", "Documents", 
#                           "zeeman_slower", "figs", "er_final_velocities_binary_high_isat_zoom5.pdf")
# fig.savefig(file_path, bbox_inches="tight")






