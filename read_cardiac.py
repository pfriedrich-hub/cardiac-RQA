from pathlib import Path
from matplotlib import pyplot as plt
import numpy
import pandas as pd
# connect to server and mount project folder
import os
os.system("osascript -e 'mount volume \"smb://m40.cfi-asl.mcgill.ca/spl-projects/pain\"'")
path = Path('/Volumes/pain')

subject_id = '1010'
csv_path = path / 'info_summary_python_readable.csv'

def rr_to_hr(rr_data, method='windowed', windowsize=750, show=False):
    """
    convert rr to hrt
    method:
    windowsize: if method = 'windowed', set window size in miliseconds.
        note: cannot be smaller than 750 due to low and irregular sampling rate
    """
    times = numpy.arange(0, rr_data.sum())  # time points in ms
    rr_idx = numpy.cumsum(rr_data) - 1  # get sampling times
    rr_samp = numpy.empty(times.shape)
    rr_samp[:] = numpy.nan
    rr_samp[rr_idx] = rr_data  # rr across time
    # plt.figure()
    # plt.scatter(times, rr_samp)
    if method == 'simple':
        hr_data = 60 / rr_samp * 1000
    elif method == 'windowed':
        for idx in range(0, len(rr_samp), windowsize):
            rr_samp[idx:idx + windowsize] = numpy.nanmean(rr_samp[idx:idx+windowsize])
        hr_data = 60 / rr_samp * 1000
    if show:
        fig, axis = plt.subplots(1, 1)
        axis.plot(times, hr_data)
        axis.set_yticks(numpy.arange(50, 160, 10))
        axis.set_xlabel('Time (ms)')
        axis.set_ylabel('Local BPM')
    return hr_data

def get_condition_data(hr_data, csv_path):
    subj_csv = pd.read_csv(path / 'info_summary_python_readable.csv', sep=';', on_bad_lines='skip')
    subj_data = subj_csv[subj_csv['Participants No.'] == int(subject_id)]
    condition_data = dict(subj_data.iloc[0, 12:25])
    for key in condition_data.keys():  # convert to ms
        condition_data[key] = sum(int(x) * 60 ** i for i, x in enumerate(reversed(condition_data[key].split(':'))))
        condition_data[key] *= 1000
    condition_data['SPR'] = hr_data[condition_data['SPR_start']:condition_data['SPR_end']]
    condition_data['B1'] = hr_data[condition_data['B1_start']:condition_data['B1_end']]
    condition_data['B2'] = hr_data[condition_data['B2_start']:condition_data['B2_end']]
    condition_data['SC'] = hr_data[condition_data['SC_start']:condition_data['SC_end']]
    condition_data['T1'] = hr_data[condition_data['T1_start']:condition_data['T1_end']]
    condition_data['T2'] = hr_data[condition_data['T2_start']:condition_data['T2_end']]
    condition_data['T3'] = hr_data[condition_data['T3_start']:]
    return condition_data

"""
# read rr data
rr_data = numpy.loadtxt(path / 'CardiacData' / f'{subject_id}.txt').astype(int)  # [1000:1010]
# convert to hr
hr_data = rr_to_hr(rr_data, method='windowed', windowsize=1000, show=True)
# bin data in conditions
condition_data = get_condition_data(hr_data, csv_path)

plt.plot(condition_data['B1'])
"""
