import numpy
import pandas as pd
from collections import namedtuple
from scipy import stats

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
    data = namedtuple('data', 'rr' 'bpm' 'bpm_time' 'rr_time')
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

def bin_data(data, subject_id, csv_path, conditions, test_keys):
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
    subj_dict = {'t_stamp': subj_data.iloc[0, 12:26]}
    for key in subj_dict['t_stamp'].keys():  # convert time stamps to ms
        subj_dict['t_stamp'][key] = \
            sum(int(x) * 60 ** i for i, x in enumerate(reversed(subj_dict['t_stamp'][key].split(':'))))
        subj_dict['t_stamp'][key] *= 1000
    cardiac_data = namedtuple('cardiac_data', 'bpm_time bpm rr_time rr rqa rp')
    subj_dict['raw'] = cardiac_data(data.bpm_time, data.bpm, data.rr_time, data.rr, None, None)
    for condition in conditions[0]:
        bpm_time = data.bpm_time[numpy.logical_and(data.bpm_time >= subj_dict['t_stamp'][condition + '_start'],
                                                data.bpm_time <= subj_dict['t_stamp'][condition + '_end'])]
        bpm_time = (bpm_time - bpm_time.min()) / 1000  # reset start time to zero
        bpm = data.bpm[numpy.logical_and(data.bpm_time >= subj_dict['t_stamp'][condition + '_start'],
                                        data.bpm_time <= subj_dict['t_stamp'][condition + '_end'])]
        rr_time = data.rr_time[numpy.logical_and(data.rr_time >= subj_dict['t_stamp'][condition + '_start'],
                                                data.rr_time <= subj_dict['t_stamp'][condition + '_end'])]
        rr_time = (rr_time - rr_time.min()) / 1000
        rr = data.rr[numpy.logical_and(data.rr_time >= subj_dict['t_stamp'][condition + '_start'],
                                    data.rr_time <= subj_dict['t_stamp'][condition + '_end'])]
        if condition in ['T1', 'T2', 'T3']:  # additionally append data of the 4 separate test conditions
            bpm_times = numpy.array_split(bpm_time, 4)
            bpm_times = [time_arr - time_arr.min() for time_arr in bpm_times]
            rr_times = numpy.array_split(rr_time, 4)
            rr_times = [time_arr - time_arr.min() for time_arr in rr_times]
            test_data = numpy.array((bpm_times, numpy.array_split(bpm, 4),
                                     rr_times, numpy.array_split(rr, 4)), dtype='object')
            for idx, key in enumerate(test_keys[condition]):     # get cardiac data for pain/music condition
                subj_dict[condition + conditions[1][key]]= (cardiac_data(test_data[0, idx], test_data[1, idx],
                                                        test_data[2, idx], test_data[3, idx], None, None))
        subj_dict[condition] = cardiac_data(bpm_time, bpm, rr_time, rr, None, None)  # save cardiac data for main conditions
    del subj_dict['t_stamp']  # remove time stamp data from dataset
    subj_dict['id'] = subject_id
    subj_dict['conditions'] = conditions
    subj_dict['rqa_params'] = dict()
    subj_dict['rqa_params']['z-scored'] = False
    return subj_dict

def zscore(subj_dict):
    for key in subj_dict:
        if isinstance(subj_dict[key], tuple):
            subj_dict[key] = subj_dict[key]._replace(bpm=stats.zscore(subj_dict[key].bpm))
            subj_dict[key] = subj_dict[key]._replace(rr=stats.zscore(subj_dict[key].rr))
    return subj_dict

# --- convenience functions and dev ----#
def read_cardiac(data_path, csv_path, conditions, s_id, resamp_rate):
    # read RR data
    data = numpy.loadtxt(data_path / f'{s_id}.txt').astype(int)
    # convert to HR
    data = rr2bpm(data, resamp_rate=resamp_rate)
    # bin data to conditions
    data = bin_data(data, csv_path, conditions, id)
    return data

def rr2bpm_dev(rr_data, method='windowed', window_size=1000, rescale=True, resamp_rate=0.5, show=False):
    """
    convert rr to heart rate (pauls version)
    method (string): 'windowed' or 'interpolation'
    window_size: if method is 'windowed', set window size in miliseconds.
    rescale (boolean): rescale the time series
    resamp_rate (int): Re-sampling rate of the resulting BPM data (Seconds)
    """
    from scipy import signal
    from matplotlib import pyplot as plt
    times = numpy.arange(0, rr_data.sum())  # time points in ms
    rr_idx = numpy.cumsum(rr_data) - 1  # get sampling times
    rr_samp = numpy.empty(times.shape)
    rr_samp[:] = numpy.nan
    rr_samp[rr_idx] = rr_data  # rr across time
    if method == 'interpolation':
        rr_interp = numpy.interp(x=times, xp=times[rr_idx], fp=rr_data)
        hr_data = 60 / rr_interp * 1000
    elif method == 'windowed':
        for idx in range(0, len(rr_samp), window_size):
            rr_samp[idx:idx + window_size] = numpy.nanmean(rr_samp[idx:idx+window_size])
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


    # # create single time series across T1,T2,T3 for each measure bpm_time, bpm, rr_time, rr
    # for key in conditions[1]:  # fetch time series across T1,T2,T3 for each pain / music condition
    #     data = numpy.concatenate(numpy.asarray(test_conditions[key]), axis=1)
    #     subc_data = []
    #     for i in range(4):
    #         if i == 0 or i == 2:  # add-up times to continuous series across T1 T2 T3
    #             data[i][1] += data[i][0].max() + numpy.diff(data[i][0]).mean()  # stitch times
    #             data[i][2] += data[i][1].max() + numpy.diff(data[i][1]).mean()
    #         subc_data.append(numpy.concatenate(data[i]))
    #     subj_dict[key] = cardiac_data(subc_data[0], subc_data[1], subc_data[2], subc_data[3], None, None)
    #     subj_dict[key] = cardiac_data(subc_data[0], subc_data[1], subc_data[2], subc_data[3], None, None)
