import numpy
import sys, os
import pandas as pd
from collections import namedtuple
from pyrqa.computation import RQAComputation
from pyrqa.neighbourhood import FixedRadius
from matplotlib import pyplot as plt
from scipy import stats

# RQA parameter selection
import teaspoon.parameter_selection.MI_delay as AMI  # average mutual information --> delay (d)
import teaspoon.parameter_selection.FNN_n as FNN  # false nearest neighbours --> embedding dimension (m)

# RQA computation
from pyrqa.time_series import TimeSeries
from pyrqa.computation import RQAComputation
from pyrqa.computation import RPComputation
from pyrqa.settings import Settings

def get_subject_ids(csv_path):
    """ Get subject ids from csv """
    subj_csv = pd.read_csv(csv_path, sep=';', on_bad_lines='skip')
    return [str(x) for x in list(subj_csv['Participants No.'])]

def rr2bpm(rr_data, resamp_rate):
    """
    Convert beat-to-beat (R-R) intervals to heart rate (bpm) and resample signal

    Parameters:
    rr_data (numpy.ndarray): Array containing the RR-intervall data (ms)
    resamp_rate (int): Resampling rate of the resulting BPM data (Hz)
        window_size is determined by resampling rate
    Returns:
    (named tuple): (BPM, RR, time points)
    """
    data = namedtuple('data', 'rr' 'bpm' 'bpm_time')
    window_size = int(1 / resamp_rate * 1000)  # convert sampling rate (Hz) to resampling interval (ms)
    time_pts = rr_data.sum()  # time points in ms
    temp_data = numpy.zeros(time_pts)
    data.bpm = numpy.zeros(int(time_pts / window_size))  # array to hold resampled signal
    for i in range(len(rr_data)):  # reconstruct time series from RR intervals
        temp_data[rr_data[:i].sum():rr_data[:i+1].sum()] = rr_data[i]
    for j, k in enumerate(range(0, time_pts-window_size, window_size)):  # resample time series
        data.bpm[j] = temp_data[k:k+window_size].sum() / window_size  # average value across time window
    data.bpm = 60 / data.bpm * 1000  # rescale from ms to bpm
    data.bpm_time = numpy.linspace(0, time_pts, int(time_pts / window_size))  # time points of resampled signal in ms
    data.rr_time = rr_data.cumsum()
    data.rr = rr_data
    return data

def bin_data(data, subject_id, csv_path, conditions):
    """
    returns dictionary of subject data by conditions

    Parameters:
    data (named tuple): Tuple containing RR interval, BPM and times (ms)
    csv_path (str): path to csv file containing participant data
    condition (list of str): condition names
    id (str): subject id

    Returns:
    (dictionary): time stamps, named tuple (BPM, time points (ms))
    """
    subj_csv = pd.read_csv(csv_path, sep=';', on_bad_lines='skip')
    subj_data = subj_csv[subj_csv['Participants No.'] == int(subject_id)]
    subj_dict = {'t_stamp': dict(subj_data.iloc[0, 12:26])}
    for key in subj_dict['t_stamp'].keys():  # convert time stamps to ms
        subj_dict['t_stamp'][key] = \
            sum(int(x) * 60 ** i for i, x in enumerate(reversed(subj_dict['t_stamp'][key].split(':'))))
        subj_dict['t_stamp'][key] *= 1000
    for c in conditions:
        subj_dict[c] = dict()
        subj_dict[c]['cardiac'] = namedtuple('data', 'rr' 'bpm' 'bpm_time')
        subj_dict[c]['cardiac'].bpm_time = data.bpm_time[numpy.logical_and(data.bpm_time >= subj_dict['t_stamp'][c+'_start'],
                                                                 data.bpm_time <= subj_dict['t_stamp'][c+'_end'])]
        subj_dict[c]['cardiac'].bpm = data.bpm[numpy.logical_and(data.bpm_time >= subj_dict['t_stamp'][c+'_start'],
                                                                 data.bpm_time <= subj_dict['t_stamp'][c+'_end'])]
        subj_dict[c]['cardiac'].rr = data.rr[numpy.logical_and(data.rr_time >= subj_dict['t_stamp'][c + '_start'],
                                                         data.rr_time <= subj_dict['t_stamp'][c + '_end'])]
    del subj_dict['t_stamp']  # remove time stamp data from dataset
    subj_dict['id'] = subject_id
    subj_dict['conditions'] = conditions
    subj_dict['rqa_params'] = dict()
    return subj_dict

def zscore(subj_dict):
    for c in subj_dict['conditions']:
        subj_dict[c]['cardiac'].bpm = stats.zscore(subj_dict[c]['cardiac'].bpm)
        subj_dict[c]['cardiac'].rr = stats.zscore(subj_dict[c]['cardiac'].rr)
    return subj_dict

def rqa_params(subj_dict): # parameter selection
    delay = 0
    embedding = 0
    for c in subj_dict['conditions']:
        if subj_dict['metric'] == 'bpm':
            c_data = subj_dict[c]['cardiac'].bpm
        elif subj_dict['metric'] == 'rr':
            c_data = subj_dict[c]['cardiac'].rr
        else:
            raise TypeError("metric must be 'rr' or 'bpm'")
        delay += (AMI.MI_for_delay(c_data, plotting=False,  # method='kraskov 1',
                                  h_method='sqrt', k=2, ranking=True))
        embedding += (FNN.FNN_n(ts=c_data, tau=delay, maxDim=50, plotting=False)[1])
    d = int(numpy.ceil(delay / len(subj_dict.keys())))  # mean delay parameter
    m = int(numpy.ceil(embedding / len(subj_dict.keys())))  # mean embedding parameter
    return (d, m)

def estimate_radius(subj_dict, r_start=.1, r_step=.05, rr_lower=.1, rr_upper=.15, plot=False, axis=None):
    """
    estimate radius parameter for fixed %REC interval
    Args:
        rr (float): %REC threshold
    """
    subj_dict['rqa_params']['settings'].neighbourhood = FixedRadius(r_start)
    rr_list = []
    rr_list.append(rr_across_conditions(subj_dict))
    while not all(ele >= rr_lower and ele < rr_upper for ele in rr_list[-1][1]):  # iterate over radii
        r_start += r_step
        subj_dict['rqa_params']['settings'].neighbourhood = FixedRadius(r_start)
        rr_list.append(rr_across_conditions(subj_dict))
    if plot:  ## feedback on r estimation
        if not axis:
            fig, axis = plt.subplots(1, 1)
        x_val = [x[0] for x in rr_list]
        y_val = numpy.array([x[1] for x in rr_list])
        for i in range(y_val.shape[-1]):
            axis.plot(x_val, y_val)
        axis.set_xlabel('radius')
        axis.set_ylabel('%REC')
    return r_start

def rr_across_conditions(subj_dict):
    d = subj_dict['rqa_params']['delay']
    m = subj_dict['rqa_params']['embedding']
    settings = subj_dict['rqa_params']['settings']
    rr_list = []
    for c in subj_dict['conditions']:
        if subj_dict['metric'] == 'bpm':
            c_data = subj_dict[c]['cardiac'].bpm
        elif subj_dict['metric'] == 'rr':
            c_data = subj_dict[c]['cardiac'].rr
        time_series = TimeSeries(c_data.tolist(), embedding_dimension=m, time_delay=d)
        settings.time_series_x, settings.time_series_y = time_series, time_series
        sys.stdout = open(os.devnull, 'w')  # suppress print
        computation = RQAComputation.create(settings, verbose=True)
        sys.stdout = sys.__stdout__
        rqa_result = computation.run()
        if rqa_result.recurrence_rate >= 0.15:
            print('overshooting! radius: %f, \n%%REC: %.4f'
                  % (settings.neighbourhood.radius, rqa_result.recurrence_rate))
            break
        else:
            rr_list.append(rqa_result.recurrence_rate)
    return (settings.neighbourhood.radius, rr_list)

def apply_rqa(subj_dict):
    d = subj_dict['rqa_params']['delay']
    m = subj_dict['rqa_params']['embedding']
    settings = subj_dict['rqa_params']['settings']
    for c in subj_dict['conditions']:
        if subj_dict['metric'] == 'bpm':
            c_data = subj_dict[c]['cardiac'].bpm
        elif subj_dict['metric'] == 'rr':
            c_data = subj_dict[c]['cardiac'].rr
        time_series = TimeSeries(c_data.tolist(), embedding_dimension=m, time_delay=d)
        settings.time_series_x, settings.time_series_y = time_series, time_series
        sys.stdout = open(os.devnull, 'w')  # suppress print
        computation = RQAComputation.create(settings, verbose=True)
        sys.stdout = sys.__stdout__
        rqa_result = computation.run()
        rp_computation = RPComputation.create(settings)
        rp = rp_computation.run()
        subj_dict[c]['rqa_result'] = rqa_result
        subj_dict[c]['rp'] = rp
    return subj_dict


# convenience function
def read_cardiac(data_path, csv_path, conditions, id, resamp_rate):
    # read RR data
    data = numpy.loadtxt(data_path / f'{id}.txt').astype(int)
    # convert to HR
    data = rr2bpm(data, resamp_rate=resamp_rate)
    # bin data to conditions
    data = bin_data(data, csv_path, conditions, id)
    return data

# old stuff
def rr2bpm_old(rr_data, window_size, resamp_rate):
    """
    Convert beat-to-beat (R-R) intervals to heart rate (bpm) and resample signal

    Parameters:
    rr_data (numpy.ndarray): Array containing the RR-intervall data (ms)
    window_size (int): Size of the time window over which BPM is calculated (ms)
    resamp_rate (int): Re-sampling rate of the resulting BPM data (Hz)

    Returns:
    (named tuple): (BPM, RR, time points)
    """
    data = namedtuple('data', 'bpm' 'rr' 'times')
    resamp_rate = int(1 / resamp_rate * 1000)  # convert sampling rate (Hz) to resampling interval (ms)
    time_pts = rr_data.sum()  # time points in ms
    temp_data = numpy.zeros(time_pts)
    data.rr = numpy.zeros(int(time_pts / resamp_rate))  # array to hold resampled signal
    data.times = numpy.linspace(0, time_pts, int(time_pts / resamp_rate))  # time points of resampled signal in ms
    for i in range(len(rr_data)):  # reconstruct time series from RR intervals
        temp_data[rr_data[:i].sum():rr_data[:i + 1].sum()] = rr_data[i]
    for j, k in enumerate(range(0, time_pts - window_size, resamp_rate)):  # resample time series
        data.rr[j] = temp_data[k:k + window_size].sum() / window_size  # average value across time window
    data.bpm = 60 / data.rr * 1000  # rescale from ms to bpm
    return data

#
# def rr2bpm_dev(rr_data, method='windowed', windowsize=1000, rescale=True, resamp_rate=0.5, show=False):
#     """
#     convert rr to heart rate (pauls version)
#     method (string): 'windowed' or 'interpolation'
#     windowsize: if method is 'windowed', set window size in miliseconds.
#     rescale (boolean): rescale the time series
#     resamp_rate (int): Re-sampling rate of the resulting BPM data (Seconds)
#     """
#     times = numpy.arange(0, rr_data.sum())  # time points in ms
#     rr_idx = numpy.cumsum(rr_data) - 1  # get sampling times
#     rr_samp = numpy.empty(times.shape)
#     rr_samp[:] = numpy.nan
#     rr_samp[rr_idx] = rr_data  # rr across time
#     if method == 'interpolation':
#         rr_interp = numpy.interp(x=times, xp=times[rr_idx], fp=rr_data)
#         hr_data = 60 / rr_interp * 1000
#     elif method == 'windowed':
#         for idx in range(0, len(rr_samp), windowsize):
#             rr_samp[idx:idx + windowsize] = numpy.nanmean(rr_samp[idx:idx+windowsize])
#         hr_data = 60 / rr_samp * 1000
#     if rescale:
#         times = numpy.arange(hr_data.sum())  # get time points in ms
#         hr_data, times = signal.resample(x=hr_data, num=int(len(hr_data) / resamp_rate), t=times)
#     if show:
#         fig, axis = plt.subplots(1, 1)
#         axis.plot(times, hr_data)
#         axis.set_yticks(numpy.arange(50, 160, 10))
#         axis.set_xlabel('Time (ms)')
#         axis.set_ylabel('Local BPM')
#     return hr_data
#
