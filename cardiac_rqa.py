import read_cardiac as rc
import numpy
import matplotlib.pyplot as plt
from pathlib import Path
import os
# RQA parameter selection
import teaspoon.parameter_selection.MI_delay as AMI  # average mutual information --> delay (d)
import teaspoon.parameter_selection.FNN_n as FNN  # false nearest neighbours --> embedding dimension (m)
# RQA analysis

import scipy.stats as stats
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

# subject id
subject_id = '1010'

# connect to server and mount project folder
os.system("osascript -e 'mount volume \"smb://m40.cfi-asl.mcgill.ca/spl-projects/pain\"'")
path = Path('/Volumes/pain')
csv_path = path / 'info_summary_python_readable.csv'

# read RR data
rr_data = numpy.loadtxt(path / 'CardiacData' / f'{subject_id}.txt').astype(int)
# convert to HR
data = rc.rr2bpm(rr_data, window_size=500, resamp_rate=2)

# bin data to conditions
data = rc.bin_data(data, csv_path, subject_id)

# select measurement
B1 = data['B1'].bpm
B1 = data['B1'].rr

# parameter selection
d = AMI.MI_for_delay(B1, plotting=True, method='kraskov 1', h_method='sqrt', k=2, ranking=True)
m =