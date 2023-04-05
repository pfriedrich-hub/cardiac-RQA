import preprocessing as pr
from preprocessing import cardiac_data
import datetime
import rqa as rqa
import plot
import numpy
from pathlib import Path
import os
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
numpy.seterr(divide='ignore', invalid='ignore')  # ignore zero divide
# RQA analysis
from pyrqa.settings import Settings
from pyrqa.time_series import TimeSeries
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.image_generator import ImageGenerator


# settings:
rqa_conditions = ['B1', 'B2', 'T1B', 'T1O', 'T1F', 'T1S',
                  'T2B', 'T2O', 'T2F', 'T2S', 'T3B', 'T3O', 'T3F', 'T3S']
metric = 'rr'  # can be 'bpm' or 'rr'
zscore = True
# radius parameters
radius_estimation = 'fixed_rr'
r_start = .5
r_step = .005
# if fixed recurrence rate
rr_target = .05
# if recurrence rate interval
rr_lower = .1
rr_upper = .15

# connect to server and mount project folder
os.system("osascript -e 'mount volume \"smb://m40.cfi-asl.mcgill.ca/spl-projects/pain\"'")
project_path = Path('/Volumes/pain')
data_path = project_path / 'CardiacData'
csv_path = project_path / 'music_pain_py_readable.csv'
conditions = [['SPR', 'B1', 'B2', 'SC', 'T1', 'T2', 'T3'], ['B', 'O', 'F', 'S']]  # 0 - B, 1 - O, 2 - F, 3 - S
test_keys = {'T1': numpy.array((0, 1, 3, 2)), 'T2': numpy.array([2, 3, 0, 1]), 'T3': numpy.array([1, 0, 2, 3])}
subject_ids = pr.get_subject_ids(csv_path)[5:]  # some files are missing
rqa_settings = Settings(TimeSeries(()), analysis_type='Classic', similarity_measure=EuclideanMetric)  # RQA settings

# iterate across subjects, return data dictionary containing cardiac data, rqa parameters and rqa results
data = dict()
for subject_id in subject_ids:
    subj_dict = numpy.loadtxt(data_path / f'{subject_id}.txt').astype(int)  # read raw subject rr data
    subj_dict = pr.rr2bpm(subj_dict, resamp_rate=3)  # calculate HR
    subj_dict = pr.bin_data(subj_dict, subject_id, csv_path, conditions, test_keys)  # bin data to conditions
    subj_dict['rqa_params']['rqa_conditions'] = rqa_conditions
    subj_dict['rqa_params']['metric'] = metric.lower()
    subj_dict['rqa_params']['delay'], subj_dict['rqa_params']['embedding'] = rqa.rqa_params(subj_dict)
    subj_dict['rqa_params']['settings'] = rqa_settings  # create settings key in subject dictionary
    if zscore:
        subj_dict = pr.zscore(subj_dict)  # z-score (dont)
        subj_dict['rqa_params']['z-scored'] = True
    if radius_estimation == 'rr_interval':     # estimate radius for rr interval
        subj_dict = rqa.fixed_radius_for_rr_interval(subj_dict, r_start, r_step, rr_lower, rr_upper, plot=True)
    elif radius_estimation == 'fixed_rr':      # estimate radii for each
        subj_dict = rqa.fixed_rr_to_radius(subj_dict, rr_target, r_start, r_step)
    subj_dict = rqa.subject_rqa(subj_dict)  # get rqa results with selected parameters
    data[subject_id] = subj_dict  # append subject data to grand dataset


# ---- Plot results ----- #
# select conditions[0] for main conditions, conditions[1] for test conditions
block = 'T1'
plot_cnd = ['B', 'O', 'F', 'S']
plot_cnd = [block + c for c in plot_cnd]
for subject_id in subject_ids:
    # subj_dict = data[subject_id]
    # # plot cardiac data of single condition
    # plot.cardiac_condition(subj_dict, conditions=conditions, metric='rr')
    # # Recurrence Plots for each condition
    # plot.rp(subj_dict, conditions=plot_cnd)
    plot.rqa_results(data[subject_id], save=True)

# --- save and read processed data --- #
file_path = project_path / f'RQA_results'
# save
import pickle
file_path.mkdir(parents=True, exist_ok=True)  # create subject image directory
file_name = file_path / str('RQA_results_' + datetime.datetime.now().strftime('_%d.%m'))
with open(file_name, 'wb') as f:  # save the newly recorded calibration
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

# read
file_name = file_path / str('RQA_results_' + datetime.datetime.now().strftime('_%d.%m'))
with open(file_name, "rb") as f:
    data = pickle.load(f)


# go with rr - seperate into 4
# don't resample, bc arqa
# choose d, r, m for 14 conditions b1 b2 + 4 p/m * 3 T

# add to bar plot: SDNN to HRV, and mean rr



# m and d is averaged for each participant across conditions (maybe on raw?)
# should eventually be averaged across all participants

# r is chosen, so that RR is within 10-15% across conditions
# det avertl lam avdiagl rr


"""
# # plot
# fig, ax = plt.subplots()
# mat = ax.imshow(rm, cmap='Greys')
# ax.invert_yaxis()
# plt.xlabel('Time')
# plt.ylabel('Time')


# plot
sub = '1010'
condition = 'T1'
d = data[sub]['d']
dat = data[sub][condition].bpm
x = dat[:-2*d]
y = dat[d:-d]
z = dat[2*d:]
xyzs = numpy.array((x, y, z))
ax = plt.figure().add_subplot(projection='3d')
ax.plot(*xyzs, lw=0.5)
"""