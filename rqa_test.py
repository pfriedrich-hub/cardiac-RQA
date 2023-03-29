import rqa as rqa
import plot as plot
import numpy
from pathlib import Path
from pyrqa.settings import Settings
from pyrqa.time_series import TimeSeries
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric

show = True
path = Path.cwd() / 'data'

# load lorenz sequence from R
lor_data = numpy.loadtxt(path / 'lorData.txt', comments="#", delimiter=",", skiprows=1)
# x, y, z = lor_data[:, 1], lor_data[:, 2], lor_data[:, 3]
# load sample data
rr_data = numpy.loadtxt(path / 'rr_data.txt').astype(int)  # read raw subject rr data

# data = lor_data[:, 1]
data = rr_data

delay = rqa.get_delay(data, plot=show)
embedding = rqa.get_embedding_dim(data, delay, plot=show)
radius = data.std() * 1.2

rqa_settings = Settings(TimeSeries(()), analysis_type='Classic', similarity_measure=EuclideanMetric,
                        neighbourhood=FixedRadius(radius), theiler_corrector=1)
rqa_results = rqa.auto_rqa(data, delay, embedding, rqa_settings)
plot.rp(rqa_results[1].recurrence_matrix)

print(rqa_results[0])

