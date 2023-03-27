import preprocessing as pr
import rqa as rqa
import numpy
from pathlib import Path
import os
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
numpy.seterr(divide='ignore', invalid='ignore')  # ignore zero divide

# connect to server and mount project folder
os.system("osascript -e 'mount volume \"smb://m40.cfi-asl.mcgill.ca/spl-projects/pain\"'")
data_path = Path('/Volumes/pain/CardiacData')
csv_path = Path('/Volumes/pain/info_summary_python_readable.csv')
conditions = ['SPR', 'B1', 'B2', 'SC', 'T1', 'T2', 'T3']
subject_ids = pr.get_subject_ids(csv_path)[5:-1]  # some files are missing
s_id = subject_ids[1]
data = numpy.loadtxt(data_path / f'{s_id}.txt').astype(int)
subj_dict = numpy.loadtxt(data_path / f'{s_id}.txt').astype(int)  # read raw subject rr data
subj_dict = pr.rr2bpm(subj_dict, resamp_rate=2)  # calculate HR
subj_dict = pr.bin_data(subj_dict, s_id, csv_path, conditions)  # bin data to conditions

# get data
rr_data = subj_dict['T1']['cardiac'].rr
times = numpy.arange(0, rr_data.sum())  # time points in ms
rr_idx = numpy.cumsum(rr_data) - 1  # get sampling times
rr_samp = numpy.empty(times.shape)
rr_samp[:] = numpy.nan
rr_samp[rr_idx] = rr_data  # rr across time


# scatter plot
fig, axis = plt.subplots(1, 1)
axis.scatter(times, rr_samp)
axis.set_xticks(numpy.arange(0, 90000, 10000))
axis.set_xlim(0, 90000)
ticks = numpy.arange(0, 90000, 10000) / 1000  # get ticks in seconds
ticks = [str(int(x)) for x in ticks]
axis.set_xticks(numpy.arange(0, 90000, 10000), ticks)
axis.set_ylim(700,1100)
axis.set_xlabel('Times (s)')
axis.set_ylabel('R-R interval (ms)')


# plot split into 4
time_pts = rr_data.sum()  # time points in ms
temp_data = numpy.zeros(time_pts)
for i in range(len(rr_data)):  # reconstruct time series from RR intervals
    temp_data[rr_data[:i].sum():rr_data[:i + 1].sum()] = rr_data[i]

fig, axis = plt.subplots(4, 1)
axis[0].plot(times[:150000], temp_data[:150000])
axis[1].plot(times[150000:300000], temp_data[150000:300000])
axis[2].plot(times[300000:450000], temp_data[300000:450000])
axis[3].plot(times[450000:], temp_data[450000:])
for i in range(1, 5):
    axis[i-1].set_xticks(numpy.arange(0, 150000, 10000))
    ticks = numpy.arange(150000*(i-1), 150000*i, 10000) / 1000  # get ticks in seconds
    ticks = [str(int(x)) for x in ticks]
    # axis.set_xlim(0, 150000)
    axis[i-1].set_xticks(numpy.arange(150000*(i-1), 150000*i, 10000), ticks)
    axis[i-1].set_ylim(700, 1100)
    axis[i-1].set_ylabel('R-R interval (ms)')
axis[3].set_xlabel('Times (s)')




import numpy as np
import slab
from scipy.signal import chirp
t = np.linspace(0, 10, 1500)
w = chirp(t, f0=1, f1=16000, t1=500, method='linear')
chrp = slab.Sound(w)
chrp.spectrum()

plt.plot(t, w)
plt.title("Linear Chirp, f(0)=6, f(10)=1")
plt.xlabel('t (sec)')
plt.show()


fs = 44800
T = 4
t = np.arange(0, int(T*fs)) / fs
def plot_spectrogram(title, w, fs):
    ff, tt, Sxx = spectrum(w, fs=fs, nperseg=256, nfft=576)
    fig, ax = plt.subplots()
    ax.pcolormesh(tt, ff[:145], Sxx[:145], cmap='gray_r',
                  shading='gouraud')
    ax.set_title(title)
    ax.set_xlabel('t (sec)')
    ax.set_ylabel('Frequency (Hz)')
    ax.grid(True)

w = chirp(t, f0=1500, f1=250, t1=T, method='quadratic')
plot_spectrogram(f'Quadratic Chirp, f(0)=1500, f({T})=250', w, fs)
plt.show()