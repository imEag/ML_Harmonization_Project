import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mne.viz import plot_topomap
import os
import utils as us

# Uso del código
A, W, ch_names = us.get_spatial_filter('54x10')
ch_names = [x.replace(' ', '') for x in ch_names]
label = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
comp = 1
#fig3 = us.single_topomap(A[:, comp], ch_names, show=True, label='1', show_names=False)
#plt.show()

print("FIN")

A_thresholded=(np.abs(A[:, comp]) > np.abs(A[:, comp]).mean()).astype(int)

fig3 = us.single_topomap(A_thresholded, ch_names, show=True, label='1', show_names=False,cmap='Greys')
plt.show()