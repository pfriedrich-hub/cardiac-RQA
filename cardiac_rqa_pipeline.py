import preprocessing as pr
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
metric = 'rr'  # can be 'bpm' or 'rr'
zscore = True
show_r = True  # show rr as function of radius
# radius parameters
r_start = .1
r_stop = .05
rr_lower = .1
rr_upper = .15

# connect to server and mount project folder
os.system("osascript -e 'mount volume \"smb://m40.cfi-asl.mcgill.ca/spl-projects/pain\"'")
data_path = Path('/Volumes/pain/CardiacData')
csv_path = Path('/Volumes/pain/info_summary_python_readable.csv')
conditions = (['SPR', 'B1', 'B2', 'SC', 'T1', 'T2', 'T3'], ['p', 'pm', 'pm+', 'pm-'])
test_keys = {'T1': numpy.array((2, 1, 3, 0)), 'T2': numpy.array([1, 3, 0, 2]), 'T3': numpy.array([0, 1, 3, 2])}
subject_ids = pr.get_subject_ids(csv_path)[5:-1]  # some files are missing
rqa_settings = Settings(TimeSeries(()), analysis_type='Classic', similarity_measure=EuclideanMetric)  # RQA settings

# iterate across subjects, return data dictionary containing cardiac data, rqa parameters and rqa results
data = dict()
for subject_id in subject_ids:
    subj_dict = numpy.loadtxt(data_path / f'{subject_id}.txt').astype(int)  # read raw subject rr data
    subj_dict = pr.rr2bpm(subj_dict, resamp_rate=2)  # calculate HR
    subj_dict = pr.bin_data(subj_dict, subject_id, csv_path, conditions, test_keys)  # bin data to conditions
    subj_dict['rqa_params']['metric'] = metric.lower()
    subj_dict['rqa_params']['delay'], subj_dict['rqa_params']['embedding'] = rqa.rqa_params(subj_dict, plot=show_r)
    subj_dict['rqa_params']['settings'] = rqa_settings  # create settings key in subject dictionary
    if zscore:
        subj_dict = pr.zscore(subj_dict)  # z-score NOT
        subj_dict['rqa_params']['z-scored'] = True
    subj_dict['rqa_params']['settings'].neighbourhood = FixedRadius(rqa.estimate_radius(subj_dict,
                        r_start=.1, r_step=.05, rr_lo=.1, rr_up=.15, conditions='all', plot=plot))  # estimate radius
    subj_dict = rqa.subject_rqa(subj_dict)  # get rqa results with selected parameters

    data[subject_id] = subj_dict  # append subject data to grand dataset


# Plot wrappers
# select conditions[0] for main conditions, conditions[1] for test conditions
cond = conditions[1]
for subject_id in subject_ids:
    subj_dict = data[subject_id]
    # plot cardiac data of single condition
    plot.cardiac_condition(subj_dict, conditions=cond, metric='rr')
    # Recurrence Plots for each condition
    plot.rp(subj_dict, conditions=cond)







# m and d is averaged for each participant across conditions (maybe on raw?)
# should eventually be averaged across all participants

# r is chosen, so that RR is within 10-15% across conditions



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