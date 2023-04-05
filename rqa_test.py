import scipy.stats

import rqa as rqa
import plot as plot
import numpy
from pathlib import Path
from pyrqa.settings import Settings
from pyrqa.time_series import TimeSeries
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
import teaspoon.parameter_selection.MI_delay as AMI  # average mutual information --> delay (d)
import teaspoon.parameter_selection.FNN_n as FNN

show = True
path = Path.cwd() / 'data'

# lorenz sequence from R
# data = numpy.loadtxt(path / 'lorData.txt', comments="#", delimiter=",", skiprows=1)

#  rr sample data
data = numpy.loadtxt(path / 'rr_data.txt').astype(int)  # read raw subject rr data


# zscore
data = scipy.stats.zscore(data)

delay = AMI.MI_for_delay(data, plotting=True, method='basic', h_method='sturge', ranking=True)
# k = 1 + 3.322 * numpy.log(len(data))

embedding = FNN.FNN_n(ts=data, tau=delay, maxDim=20, plotting=True)[1]

# radius = data.std() * 1.2

radius = .75

rqa_settings = Settings(TimeSeries(()), analysis_type='Classic', similarity_measure=EuclideanMetric,
                        neighbourhood=FixedRadius(radius), theiler_corrector=delay)

rqa_results = rqa.auto_rqa(data, delay, embedding, rqa_settings)


plot.rp(rqa_results[1].recurrence_matrix)
print(rqa_results[0])
