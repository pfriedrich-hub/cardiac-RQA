""" cardiac_RQA implementation """

"""I lorenz example """
import os
import matplotlib
# matplotlib.use('TkAgg')
import numpy
import scipy.stats as stats
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# load lorenz sequence from R
fpath = os.getcwd() + '/data/lorenzdata/lorData.txt'
lorData = numpy.loadtxt(fpath, comments="#", delimiter=",", skiprows=1)
x = lorData[:, 1]
y = lorData[:, 2]
z = lorData[:, 3]

# # # plot
# xyzs = numpy.array((x,y,z))
# ax = plt.figure().add_subplot(projection='3d')
# ax.plot(*xyzs, lw=0.5)

# ---- AMI ----- #
import teaspoon.parameter_selection.MI_delay as AMI
d = AMI.MI_for_delay(x, plotting=False, method='kraskov 1', h_method='sqrt', k=2, ranking=True)
# 8 instead of 9

# ---- FNN ----- #
import teaspoon.parameter_selection.FNN_n as FNN
m = FNN.FNN_n(ts=x, tau=d, maxDim=10, plotting=False)[1]
# 2 instead of 3

#SD
SD = stats.tstd(lorData)[1:].mean()


# ----- CRQA ---- #
from pyrqa.settings import Settings
from pyrqa.time_series import TimeSeries
from pyrqa.computation import RQAComputation
from pyrqa.analysis_type import Cross
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric

# zscore
xyz = stats.zscore(numpy.array((lorData[:, 1], lorData[:, 2], lorData[:, 3])), axis=1)
x, y, z = xyz[0], xyz[1], xyz[2]
# in case of normalizing, calculate new r parameter:
r = 20
r_norm = r / SD

# create time series
ts1 = TimeSeries(x.tolist(), embedding_dimension=3, time_delay=9)
ts2 = TimeSeries(y.tolist(), embedding_dimension=3, time_delay=9)
time_series = (ts1, ts2)

settings = Settings(time_series,
                    analysis_type=Cross,
                    neighbourhood=FixedRadius(0.5),
                    similarity_measure=EuclideanMetric,
                    theiler_corrector=0)
# run CRQA
CRQA_computation = RQAComputation.create(settings, verbose=True)
CRQA_result = CRQA_computation.run()
print(CRQA_result)



# create RP
from pyrqa.computation import RPComputation
RP_computation = RPComputation.create(settings)
CRP_result = RP_computation.run()
# from pyrqa.image_generator import ImageGenerator
# img = ImageGenerator.generate_recurrence_plot(CRP_result.recurrence_matrix_reverse)#.recurrence_matrix_reverse)
# img.show()

# ImageGenerator.save_recurrence_plot(CRP_result.recurrence_matrix_reverse, 'cross_recurrence_plot.png')

# DCRP - diagonal cross recurrence profile
rm = RP_computation.recurrence_matrix  # get matrix
# plot
fig, ax = plt.subplots()
mat = ax.imshow(rm, cmap='Greys')
ax.invert_yaxis()
plt.xlabel('Time')
plt.ylabel('Time')

rm = numpy.rot90(rm)  # rotate matrix
lag = range(-20, 20)
diaglines = []
for l in lag:
    rr_lag = numpy.count_nonzero(numpy.diag(rm, k=l)) / rm.size  # get % REC for each diagonal
    diaglines.append(rr_lag)  # main diagonal
plt.figure()
plt.plot(lag, diaglines)
plt.xlabel('Lag')
plt.ylabel('% REC')




""" based on Analyzing Multivariate Dynamics Using Cross-Recurrence Quantification Analysis (CRQA),
 Diagonal-Cross-Recurrence Profiles (DCRP), and Multidimensional Recurrence Quantification Analysis (MdRQA)
  – A Tutorial in R
Sebastian Wallot1* and Giuseppe Leonardi2 """




""" alternative lorenz sequence"""
# generate a sequence of data points of the Lorenz-system dynamics

# Create an image of the Lorenz attractor.
# The maths behind this code is described in the scipython blog article
# at https://scipython.com/blog/the-lorenz-attractor/
# Christian Hill, January 2016.
# Updated, January 2021 to use scipy.integrate.solve_ivp.

# Lorenz paramters and initial conditions.
sigma, beta, rho = 10, 2.667, 28
u0, v0, w0 = -13, -14, 47

# Maximum time point and total number of time points.
tmax, n = 20, 1000

def lorenz(t, X, sigma, beta, rho):
    """The Lorenz equations."""
    u, v, w = X
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    return up, vp, wp

# Integrate the Lorenz equations.
soln = solve_ivp(lorenz, (0, tmax), (u0, v0, w0), args=(sigma, beta, rho),
                 dense_output=True)

# Interpolate solution onto the time grid, t.
t = numpy.linspace(0, tmax, n)
x, y, z = soln.sol(t)  # doesnt match R results


def lorenz(xyz, *, s=10, r=28, b=2.667):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return numpy.array([x_dot, y_dot, z_dot])

dt = 0.02
num_steps = 1000

xyzs = numpy.empty((num_steps + 1, 3))  # Need one more for the initial values
xyzs[0] = (-13, -14, 47)  # Set initial values
# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
    xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt
