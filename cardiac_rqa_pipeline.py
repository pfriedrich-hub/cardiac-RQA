import preprocessing as pr
import plot
import rqa as rqa
import datetime, os, pickle, numpy
from pathlib import Path
from pyrqa.settings import Settings  # pyRQA dependencies
from pyrqa.time_series import TimeSeries
from pyrqa.metric import EuclideanMetric
numpy.seterr(divide='ignore', invalid='ignore')  # ignore zero divide
os.system("osascript -e 'mount volume \"smb://m40.cfi-asl.mcgill.ca/spl-projects/pain\"'")
project_path = Path('/Volumes/pain')
data_path = project_path / 'CardiacData'
csv_path = project_path / 'Analysis' / 'music_pain_py_readable.csv'

""" --- analysis settings: ---- """
rqa_conditions = ['B1', 'B2', 'T1B', 'T1O', 'T1F', 'T1S',  # conditions considered for rqa analysis
                  'T2B', 'T2O', 'T2F', 'T2S', 'T3B', 'T3O', 'T3F', 'T3S']
metric = 'rr'  # choose datatype from 'bpm' or 'rr'
zscore = True  # whether to z-score the data
""" --- radius parameters ---
set radius_estimation = 'fixed', to set a fixed recurrence rate across conditions and subjects,
or 'interval' to find a common radius across conditions of a subject, with recurrence rates within a given interval"""
radius_estimation = 'fixed'
r_start = .5  # starting value for radius estimation (save time and set it to >= .5)
r_step = .005  # step size of radius estimation (has to be small for fixed RR)
rr_target = .05  # set fixed recurrence rate, only applied if radius_estimation is 'fixed'
rr_lower, rr_upper = .1, .15  # set recurrence rate interval, applied if radius estimation is 'interval'

#  experiment settings
conditions = [['SPR', 'B1', 'B2', 'SC', 'T1', 'T2', 'T3'], ['B', 'O', 'F', 'S']]  # 0 - B, 1 - O, 2 - F, 3 - S
test_keys = {'T1': numpy.array((0, 1, 3, 2)), 'T2': numpy.array([2, 3, 0, 1]), 'T3': numpy.array([1, 0, 2, 3])}
# RQA settings
rqa_settings = Settings(TimeSeries(()), analysis_type='Classic', similarity_measure=EuclideanMetric, theiler_corrector=1)

""" --- Preprocessing ---
- iterate across subjects and return data dictionary containing cardiac data and rqa parameters 
"""
data = dict()  # cross subject data dict
subject_ids = pr.get_subject_ids(csv_path)[5:]  # some files are missing
for subject_id in subject_ids:
    subj_dict = numpy.loadtxt(data_path / f'{subject_id}.txt').astype(int)  # read raw subject rr data
    subj_dict = pr.rr2bpm(subj_dict, resamp_rate=3)  # calculate HR
    subj_dict = pr.bin_data(subj_dict, subject_id, csv_path, conditions, test_keys)  # bin data to conditions
    subj_dict['rqa_params']['rqa_conditions'] = rqa_conditions
    subj_dict['rqa_params']['metric'] = metric.lower()
    subj_dict['rqa_params']['delay'], subj_dict['rqa_params']['embedding'] = rqa.rqa_params(subj_dict)
    subj_dict['rqa_params']['settings'] = rqa_settings  # create settings key in subject dictionary
    data[subject_id] = subj_dict  # append subject data to grand dataset
data = pr.set_grand_mean_rqa_params(data)  # set mean m and d as rqa parameters across all participants

""" --- Auto-recurrence Quantification --- 
- iterate across subjects, apply z-score (conditional) and aRQA to selected datatype
"""
for subject_id in subject_ids:
    subj_dict = data[subject_id]
    if zscore:
        subj_dict = pr.zscore(subj_dict)  # z-score (dont)
        subj_dict['rqa_params']['z-scored'] = True
    if radius_estimation == 'interval':     # estimate radius for rr interval
        subj_dict = rqa.fixed_radius_for_rr_interval(subj_dict, r_start, r_step, rr_lower, rr_upper, plot=True)
    elif radius_estimation == 'fixed':      # estimate radii for each
        subj_dict = rqa.fixed_rr_to_radius(subj_dict, rr_target, r_start, r_step)
    subj_dict = rqa.subject_rqa(subj_dict)  # get rqa results with selected parameters
    data[subject_id] = subj_dict  # append subject data to grand dataset

""" --- save and read processed data --- """
# save
result_path = project_path / 'Analysis' / f'pyRQA_results'
result_path.mkdir(parents=True, exist_ok=True)  # create subject image directory
file_path = result_path / str('RQA_results' + datetime.datetime.now().strftime('_%d.%m') + '.pkl')
with open(file_path, 'wb') as f:  # save the newly recorded calibration
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
# read
result_path = project_path / 'Analysis' / f'pyRQA_results'
file_path = result_path / 'RQA_results_05.04.pkl'
with open(file_path, "rb") as f:
    data = pickle.load(f)


""" ---- Plot results ----- """
block = 'T1'
plot_cnd = ['B', 'O', 'F', 'S']
plot_cnd = [block + c for c in plot_cnd]
subject_ids = pr.get_subject_ids(csv_path)[5:]  # some files are missing
for subject_id in subject_ids:
    # # plot cardiac data of selected conditions
    # plot.cardiac_condition(data[subject_id], conditions=conditions, metric='rr')
    # # Recurrence Plots of selected conditions
    # plot.rp(data[subject_id], conditions=plot_cnd)
    plot.rqa_results(data[subject_id], save=True)

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