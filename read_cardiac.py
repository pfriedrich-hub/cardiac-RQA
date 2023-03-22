from matplotlib import pyplot as plt
import numpy
import pandas as pd
from scipy import signal
import copy
from collections import namedtuple

def rr2bpm(rr_data, window_size, resamp_rate):
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
        temp_data[rr_data[:i].sum():rr_data[:i+1].sum()] = rr_data[i]
    for j, k in enumerate(range(0, time_pts-window_size, resamp_rate)):  # resample time series
        data.rr[j] = temp_data[k:k+window_size].sum() / window_size  # average value across time window
    data.bpm = 60 / data.rr * 1000  # rescale from ms to bpm
    return data

def bin_data(data, csv_path, subject_id, conditions=['SPR', 'B1', 'B2', 'SC', 'T1', 'T2', 'T3']):
    """
    get data sequence conditions

    Parameters:
    data (named tuple): Tuple containing RR interval, BPM and times (ms)
    window_size (int): Size of the time window over which BPM is calculated (ms)
    resamp_rate (int): Re-sampling rate of the resulting BPM data (Hz)

    Returns:
    (dictionary): time stamps, named tuple (BPM, time points (ms))
    """
    subj_csv = pd.read_csv(csv_path, sep=';', on_bad_lines='skip')
    subj_data = subj_csv[subj_csv['Participants No.'] == int(subject_id)]
    data_dict = {'t_stamp': dict(subj_data.iloc[0, 12:26])}
    for key in data_dict['t_stamp'].keys():  # convert time stamps to ms
        data_dict['t_stamp'][key] = \
            sum(int(x) * 60 ** i for i, x in enumerate(reversed(data_dict['t_stamp'][key].split(':'))))
        data_dict['t_stamp'][key] *= 1000
    for c in conditions:
        data_dict[c] = namedtuple('data', 'bpm' 'rr' 'times')
        start_idx = numpy.where(data.times >= data_dict['t_stamp'][c+'_start'])[0][0]
        stop_idx = numpy.where(data.times >= data_dict['t_stamp'][c+'_end'])[0][0]
        data_dict[c].bpm, data_dict[c].rr, data_dict[c].times = data.bpm[start_idx:stop_idx],\
            data.rr[start_idx:stop_idx], data.times[start_idx:stop_idx]
    return data_dict

def rr_to_hr(rr_data, method='windowed', windowsize=1000, rescale=True, resamp_rate=0.5, show=False):
    """
    convert rr to heart rate (pauls version)
    method (string): 'windowed' or 'interpolation'
    windowsize: if method is 'windowed', set window size in miliseconds.
    rescale (boolean): rescale the time series
    resamp_rate (int): Re-sampling rate of the resulting BPM data (Seconds)
    """
    times = numpy.arange(0, rr_data.sum())  # time points in ms
    rr_idx = numpy.cumsum(rr_data) - 1  # get sampling times
    rr_samp = numpy.empty(times.shape)
    rr_samp[:] = numpy.nan
    rr_samp[rr_idx] = rr_data  # rr across time
    if method == 'interpolation':
        rr_interp = numpy.interp(x=times, xp=times[rr_idx], fp=rr_data)
        hr_data = 60 / rr_interp * 1000
    elif method == 'windowed':
        for idx in range(0, len(rr_samp), windowsize):
            rr_samp[idx:idx + windowsize] = numpy.nanmean(rr_samp[idx:idx+windowsize])
        hr_data = 60 / rr_samp * 1000
    if rescale:
        times = numpy.arange(hr_data.sum())  # get time points in ms
        hr_data, times = signal.resample(x=hr_data, num=int(len(hr_data) / resamp_rate), t=times)
    if show:
        fig, axis = plt.subplots(1, 1)
        axis.plot(times, hr_data)
        axis.set_yticks(numpy.arange(50, 160, 10))
        axis.set_xlabel('Time (ms)')
        axis.set_ylabel('Local BPM')
    return hr_data
