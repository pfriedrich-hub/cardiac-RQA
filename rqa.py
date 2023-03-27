import numpy
import os, sys
# RQA parameter selection
import teaspoon.parameter_selection.MI_delay as AMI  # average mutual information --> delay (d)
import teaspoon.parameter_selection.FNN_n as FNN  # false nearest neighbours --> embedding dimension (m)
# RQA computation
from pyrqa.time_series import TimeSeries
from pyrqa.computation import RPComputation
from pyrqa.computation import RQAComputation
from pyrqa.neighbourhood import FixedRadius
from matplotlib import pyplot as plt

def rqa_params(subj_dict, metric=None, plot=False): # parameter selection
    """
    Find delay parameter d and embedding parameter m across conditions within Subject.
    :param subj_dict: Dictionary containing subject data.
    :return (int): d, m
    """
    delay = 0
    embedding = 0
    for condition in subj_dict['conditions']:
        try:
            data = getattr(subj_dict[condition], subj_dict['rqa_params']['metric'])
        except:
            raise TypeError("metric must be 'rr' or 'bpm'")
        delay += (AMI.MI_for_delay(data, plotting=plot,  # method='kraskov 1',
                                  h_method='sqrt', k=2, ranking=True))
        embedding += (FNN.FNN_n(ts=data, tau=delay, maxDim=50, plotting=plot)[1])
    d = int(numpy.ceil(delay / len(subj_dict.keys())))  # mean delay parameter
    m = int(numpy.ceil(embedding / len(subj_dict.keys())))  # mean embedding parameter
    return (d, m)

def estimate_radius(subj_dict, r_start=.1, r_step=.05, rr_lo=.1, rr_up=.15, conditions='all', plot=False, axis=None):
    """
    Iteratively increase radius parameter until Recurrence Rates across conditions are within a specified interval.
    :param subj_dict: Dictionary containing subject data.
    :param r_start (int): Starting radius
    :param r_step (int): Step size for increasing radius
    :param rr_lo (int): Lower interval threshold
    :param rr_up (int): Upper interval threshold
    :param condition (string): Can be 'all', a list of condition names or a single condition name (e.g. 'B1')
    :param plot (boolean):
    :param axis (matplotlib.pyplot axis):
    :return:
    """
    global d, m, settings  # global params for apply_rqa()
    settings = subj_dict['rqa_params']['settings']
    d = subj_dict['rqa_params']['delay']
    m = subj_dict['rqa_params']['embedding']
    sd = getattr(subj_dict['raw'], subj_dict['rqa_params']['metric']).std() # standard deviation of cardiac data
    radius, step_size = r_start * sd, r_step * sd  # rescale radius by SD
    settings.neighbourhood = FixedRadius(radius)  # set radius to starting value
    plot_list = []
    if conditions == 'all':
        conditions = subj_dict['conditions']
    if isinstance(conditions, str):
        conditions = [conditions]
    while True:  # iterate over radii
        rrs = numpy.array(())
        for condition in conditions:  # get rr across conditions for current radius
            data = getattr(subj_dict[condition], subj_dict['rqa_params']['metric'])
            rr = apply_rqa(data)[0].recurrence_rate
            rrs = numpy.append(rrs, rr)
        plot_list.append(numpy.append(radius, rrs))
        if any(rrs > rr_up):  # if any rr oversteps threshold, print warning and break
            print('overshooting in %s: \nradius: %.4f, %%REC: %s' %
                  (numpy.array(subj_dict['conditions'])[rrs > rr_up], radius, rrs[rrs > rr_up]))
            break
        if not all(ele > rr_lo and ele < rr_up for ele in rrs):  # return radius if rr limits are satisfied
            radius += step_size
            settings.neighbourhood = FixedRadius(radius)
            continue
        else:
            print(f'Estimated radius: {radius}')
            break
    if plot:
        if not axis: fig, axis = plt.subplots(1, 1)
        plot_list = numpy.asarray(plot_list)
        for i in range(0, len(rrs)):
            axis.plot(plot_list[:,0], plot_list[:,i+1], label=conditions[i])
        axis.set_xlabel('radius')
        axis.set_ylabel('%REC')
        axis.legend()
    return radius

def subject_rqa(subj_dict):
    """
    Apply RQA on each condition with selected parameters d, m, and r.
    :param subj_dict:
    :return: subject dictionary containing rqa and rp results for all conditions
    """
    global d, m, settings
    d = subj_dict['rqa_params']['delay']
    m = subj_dict['rqa_params']['embedding']
    settings = subj_dict['rqa_params']['settings']
    for key in subj_dict:
        if isinstance(subj_dict[key], tuple):
            data = getattr(subj_dict[key], subj_dict['rqa_params']['metric'])
            rqa, rp = apply_rqa(data)
            subj_dict[key] = subj_dict[key]._replace(rqa=rqa, rp=rp)
    return subj_dict

def apply_rqa(data):
    global d, m, settings
    if isinstance(data, numpy.ndarray):
        data = data.tolist()
    time_series = TimeSeries(data, embedding_dimension=m, time_delay=d)
    settings.time_series_x, settings.time_series_y = time_series, time_series
    sys.stdout = open(os.devnull, 'w')  # suppress print
    computation = RQAComputation.create(settings, verbose=True)
    sys.stdout = sys.__stdout__
    rqa_result = computation.run()
    rp_computation = RPComputation.create(settings)
    rp = rp_computation.run()
    return rqa_result, rp
