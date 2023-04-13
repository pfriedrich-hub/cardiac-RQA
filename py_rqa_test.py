import scipy.stats
from pyrqa.computation import RPComputation
import numpy
from pathlib import Path
from pyrqa.settings import Settings
from pyrqa.time_series import TimeSeries
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
import teaspoon.parameter_selection.MI_delay as AMI  # average mutual information --> delay (d)
import teaspoon.parameter_selection.FNN_n as FNN
from pyrqa.computation import RQAComputation

path = Path.cwd() / 'data'

# input data
data = numpy.loadtxt(path / 'rr_data.txt').astype(int)  # read raw subject rr data

# zscore
data = scipy.stats.zscore(data)
data = data.astype('float16')

delay = AMI.MI_for_delay(data, plotting=True, method='basic', h_method='sturge', ranking=True)
# k = 1 + 3.322 * numpy.log(len(data))
embedding = FNN.FNN_n(ts=data, tau=delay, maxDim=20, plotting=True)[1]
radius = .75

time_series = TimeSeries(data, embedding_dimension=embedding, time_delay=delay)
rqa_settings = Settings(time_series, analysis_type='Classic', similarity_measure=EuclideanMetric,
                        neighbourhood=FixedRadius(radius), theiler_corrector=0)

computation = RQAComputation.create(rqa_settings, verbose=True)
rqa_result = computation.run()
rp_computation = RPComputation.create(rqa_settings)
rp = rp_computation.run()

rqa_result.min_diagonal_line_length = 2
rqa_result.min_vertical_line_length = 2
# rqa_results.min_white_vertical_line_length = 2

# plot.rp(rp_computation.recurrence_matrix)
print(rqa_result)



#  rqa output correlation test across platforms

matlab_results = [0.0439679218967922, 0.262490087232355, 2.34751773049645, 0.651219705830869, 0.364393338620143,
                  2.24146341463415]
pyrqa_results = [0.049143, 0.263608, 2.346021, 0.636276, 0.386727, 2.248485]
labels = ['RR', 'DET', 'L', 'L_entr', 'LAM', 'TT']
from matplotlib import pyplot as plt
plt.figure()
plt.scatter(range(len(labels)), matlab_results, label='CRP Toolbox')
plt.scatter(range(len(labels)), pyrqa_results, label='pyRQA')
plt.xticks(range(len(labels)), labels)
corr_coef = numpy.corrcoef(matlab_results, pyrqa_results)[0, 1]
plt.text(0.9, 0.9, f'correlation coefficient: {corr_coef}')
plt.legend()


# pyrqa
# Recurrence rate (RR): 0.049143
# Determinism (DET): 0.263608
# Average diagonal line length (L): 2.346021
# Entropy diagonal lines (L_entr): 0.636276
# Laminarity (LAM): 0.386727
# Trapping time (TT): 2.248485

# matlab
# RR 0.0439679218967922
# DET 0.262490087232355
# Meanline 2.34751773049645
# Maxline 9
# Entropy (diagonals) 0.651219705830869
# LAM 0.364393338620143
# Trapping time 2.24146341463415
# Max_vertical line 5


# Rec-Time-first-type (white gaps) 15.3904347826087 unique to Marwan
# Rec-Time-second-type 19.5922818791946	unique to Marwan
# Rec-Period-density-entropy 0.717861865632092	unique to Marwan
# cluster-coefficient 0.491098886989673	unique to Marwan
# transitivity 0.490505127231295 unique to Marwan


# Minimum diagonal line length (L_min): 2
# Minimum vertical line length (V_min): 2
# Minimum white vertical line length (W_min): 2
# Determinism (DET): 0.263608
# Average diagonal line length (L): 2.346021
# Longest diagonal line length (L_max): 9
# Divergence (DIV): 0.111111
# Entropy diagonal lines (L_entr): 0.636276
# Laminarity (LAM): 0.386727
# Trapping time (TT): 2.248485
# Longest vertical line length (V_max): 5
# Entropy vertical lines (V_entr): 0.619231
# Average white vertical line length (W): 24.119896
# Longest white vertical line length (W_max): 240
# Longest white vertical line length inverse (W_div): 0.004167
# Entropy white vertical lines (W_entr): 3.953891
# Ratio determinism / recurrence rate (DET/RR): 5.364122
# Ratio laminarity / determinism (LAM/DET): 1.467052


