import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

fig1, ax1 = plt.subplots()

ax1.set_title(label="low current = 30.8 A, high current = 160.0 A, $\eta_{Er} = 0.486$")


fig1.set_size_inches(12, 8)

plt.show()