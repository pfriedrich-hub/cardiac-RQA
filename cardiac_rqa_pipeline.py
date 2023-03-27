import preprocessing as pr
import rqa as rqa
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
plot = True
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
test_keys = {'T1': numpy.array((2, 1, 3, 0)), 'T2':numpy.array([1, 3, 0, 2]), 'T3':numpy.array([0, 1, 3, 2])}
subject_ids = pr.get_subject_ids(csv_path)[5:-1]  # some files are missing
rqa_settings = Settings(TimeSeries(()), analysis_type='Classic', similarity_measure=EuclideanMetric)  # RQA settings

# iterate across subjects, return data dictionary containing cardiac data, rqa parameters and rqa results
data = dict()
for subject_id in subject_ids:
    subj_dict = numpy.loadtxt(data_path / f'{subject_id}.txt').astype(int)  # read raw subject rr data
    subj_dict = pr.rr2bpm(subj_dict, resamp_rate=2)  # calculate HR
    subj_dict = pr.bin_data(subj_dict, subject_id, csv_path, conditions, test_keys)  # bin data to conditions
    subj_dict['rqa_params']['metric'] = metric.lower()
    subj_dict['rqa_params']['delay'], subj_dict['rqa_params']['embedding'] = rqa.rqa_params(subj_dict, plot=False)
    subj_dict['rqa_params']['settings'] = rqa_settings  # create settings key in subject dictionary
    if zscore:
        subj_dict = pr.zscore(subj_dict)  # z-score NOT
    subj_dict['rqa_params']['settings'].neighbourhood = FixedRadius(rqa.estimate_radius(subj_dict,
                        r_start=.1, r_step=.05, rr_lo=.1, rr_up=.15, conditions='all', plot=plot))  # estimate radius
    subj_dict = rqa.subject_rqa(subj_dict)  # get rqa results with selected parameters
    data[subject_id] = subj_dict  # append subject data to grand dataset

# recurrence plots across participants and conditions
# select conditions[0] for main conditions, conditions[1] for test conditions
cond = conditions[1]
for subject_id in subject_ids:
    subj_dict = data[subject_id]
    fig, axes = plt.subplots(int(len(cond) / 2), 2)
    fig.suptitle(f'ID: {subject_id}')
    mats = []
    for ax, c in zip(fig.axes, cond):
        rm = subj_dict[c].rp.recurrence_matrix  # get matrix
        mat = ax.imshow(rm, cmap='Greys')
        ax.set_title(c)
        ax.invert_yaxis()
        # set time axis: I get time points in minutes
        rm_time = getattr(subj_dict[c], subj_dict['rqa_params']['metric'] + '_time')[:rm.shape[0]] / 60
        # II find ticks in datapoints for given set of times
        times = numpy.arange(0, rm_time.max(), int(numpy.ceil(rm_time.max() / 5)))  # 5 equally spaced time points in minutes
        # times = numpy.arange(0, 900, 100)  # seconds
        idx = [numpy.abs(rm_time - time).argmin() for time in times]  # get idx of closest datapoints to time points
        times = [str(int(x)) for x in times]  # turn ticks into list of strings
        ax.set_xticks(idx, times)
        ax.set_yticks(idx, times)


# bar graphs
for s_id in subject_ids:
    subj_dict = data[s_id]
    fig, axis = plt.subplots(3, 1)
    rrs = []
    dets = []
    lams = []
    labels = []
    for c in subj_dict['conditions']:
        rrs.append(subj_dict[c]['rqa_result'].recurrence_rate)
        dets.append(subj_dict[c]['rqa_result'].determinism)
        lams.append(subj_dict[c]['rqa_result'].laminarity)
        labels.append(c)
    axis[0].bar(range(len(rrs)), rrs, color='grey', tick_label=labels)
    axis[0].set_ylabel('%Recurrence')
    axis[0].set_xlabel('Condition')
    axis[0].set_title('Recurrence Rate')
    axis[1].bar(range(len(rrs)), dets, color='grey', tick_label=labels)
    axis[1].set_ylabel('%Determinism')
    axis[1].set_xlabel('Condition')
    axis[1].set_title('Determinism')
    axis[2].bar(range(len(rrs)), lams, color='grey', tick_label=labels)
    axis[2].set_ylabel('Laminarity')
    axis[2].set_xlabel('Condition')
    axis[2].set_title('Laminarity')




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