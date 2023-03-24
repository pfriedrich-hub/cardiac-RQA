import read_cardiac as rc
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


# connect to server and mount project folder
os.system("osascript -e 'mount volume \"smb://m40.cfi-asl.mcgill.ca/spl-projects/pain\"'")
data_path = Path('/Volumes/pain/CardiacData')
csv_path = Path('/Volumes/pain/info_summary_python_readable.csv')
conditions = ['SPR', 'B1', 'B2', 'SC', 'T1', 'T2', 'T3']
subject_ids = rc.get_subject_ids(csv_path)[5:-1]  # some files are missing

metric = 'rr'  # can be 'bpm' or 'rr'
rqa_settings = Settings(TimeSeries(()), analysis_type='Classic', similarity_measure=EuclideanMetric)  # RQA settings
plot = False

# iterate across subjects, return data dictionary containing cardiac data, rqa parameters and rqa results
data = dict()
for s_id in subject_ids:
    subj_dict = numpy.loadtxt(data_path / f'{s_id}.txt').astype(int)  # read raw subject rr data
    subj_dict = rc.rr2bpm(subj_dict, resamp_rate=2)  # calculate HR
    subj_dict = rc.bin_data(subj_dict, s_id, csv_path, conditions)  # bin data to conditions
    subj_dict['metric'] = metric
    subj_dict['rqa_params']['delay'], subj_dict['rqa_params']['embedding'] = rc.rqa_params(subj_dict)
    subj_dict['rqa_params']['settings'] = rqa_settings  # create settings key in subject dictionary
    subj_dict = rc.zscore(subj_dict)  # z-score
    subj_dict['rqa_params']['settings'].neighbourhood = FixedRadius(rc.estimate_radius(subj_dict,
                                    r_start=.1, r_step=.05, rr_lower=.1, rr_upper=.15))  # set radius
    subj_dict = rc.apply_rqa(subj_dict)  # get rqa results with selected parameters
    data[s_id] = subj_dict  # append subject data to grand dataset

# bar graphs
for s_id in subject_ids:
    subj_dict = data[s_id]
    fig, axis = plt.subplots(1, 1)
    rrs = []
    labels = []
    for c in subj_dict['conditions']:
        rrs.append(subj_dict[c]['rqa_result'].recurrence_rate)
        labels.append(c)
    axis.bar(range(len(rrs)), rrs, color='grey', tick_label=labels)
    axis.set_ylabel('%Recurrence')
    axis.set_xlabel('Condition')

#

# m and d is averaged for each participant
# % DET for each condition:
# r is chosen, so that RR is within (10-15%)



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