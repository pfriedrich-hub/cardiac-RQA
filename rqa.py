import numpy
import os, sys, copy
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
    """ # todo maybe change to grand mean
    Find delay parameter d and embedding parameter m across conditions within Subject.
    :param subj_dict: Dictionary containing subject data.
    :return (int): d, m
    """
    d = []
    m = []
    for condition in subj_dict['rqa_params']['rqa_conditions']:
        try:
            data = getattr(subj_dict[condition], subj_dict['rqa_params']['metric'])
        except:
            raise TypeError("metric must be 'rr' or 'bpm'")
        delay = get_delay(data, plot=plot)
        embedding = get_embedding_dim(data, delay, plot=plot)
        d.append(delay)
        m.append(embedding)
    d = int(numpy.ceil(numpy.mean(d)))  # mean delay parameter
    m = int(numpy.ceil(numpy.mean(m)))  # mean embedding parameter
    return (d, m)

def get_delay(data, plot=False):
    delay = AMI.MI_for_delay(data, plotting=plot,  method='basic',
                               h_method='sturge', ranking=True)
    return delay

def get_embedding_dim(data, delay, plot=False):
    embedding = FNN.FNN_n(ts=data, tau=delay, maxDim=50, plotting=plot)[1]
    return embedding + 1  # add first dim, as it is not counted by FNN

def fixed_rr_to_radius(subj_dict, rr_target=.05, r_start=.5, r_step=.005):
    """
    Find radius for each condition for the desired recurrence rate.
    :param subj_dict:
    :param target_rr:
    :param r_start:
    :param r_step:
    :return:
    """
    print(f'Estimating radii for {subj_dict["id"]}')
    settings = subj_dict['rqa_params']['settings']
    d = subj_dict['rqa_params']['delay']
    m = subj_dict['rqa_params']['embedding']
    conditions = subj_dict['rqa_params']['rqa_conditions']
    subj_dict['rqa_params']['radii'] = dict()
    for condition in conditions:  # get rr across conditions for current radius
        data = getattr(subj_dict[condition], subj_dict['rqa_params']['metric'])
        radius, step_size = r_start * data.std(), r_step * data.std()  # rescale radius by SD of data
        settings.neighbourhood = FixedRadius(radius)  # set radius to starting value
        while True:  # iterate over radii
            rr = auto_rqa(data, d, m, settings)[0].recurrence_rate
            if not rr_target - 1e-3 < rr < rr_target + 1e-3:  # check if rr within tolerance interval of target_rr
                if rr > rr_target + 1e-3:
                    print(f'Overshooting recurrence rate: {rr.astype("float16")} in {condition}')
                    break
                radius += step_size
                settings.neighbourhood = FixedRadius(radius)
                continue
            else:
                print(f'Estimated radius for {condition}: {radius.astype("float16")}')
                break
        subj_dict['rqa_params']['radii'][condition] = radius
    return subj_dict

def fixed_radius_for_rr_interval(subj_dict, rr_lower=.1, rr_upper=.15, r_start=.1, r_step=.05, plot=False, axis=None):
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
    # global d, m, settings  # global params for apply_rqa()
    conditions = subj_dict['rqa_params']['rqa_conditions']
    settings = subj_dict['rqa_params']['settings']
    d = subj_dict['rqa_params']['delay']
    m = subj_dict['rqa_params']['embedding']
    sd = getattr(subj_dict['raw'], subj_dict['rqa_params']['metric']).std()  # standard deviation of cardiac data
    radius, step_size = r_start * sd, r_step * sd  # rescale radius by SD
    settings.neighbourhood = FixedRadius(radius)  # set radius to starting value
    plot_list = []
    while True:  # iterate over radii
        rrs = numpy.array(())
        for condition in conditions:  # get rr across conditions for current radius
            data = getattr(subj_dict[condition], subj_dict['rqa_params']['metric'])
            rr = auto_rqa(data, d, m, settings)[0].recurrence_rate
            rrs = numpy.append(rrs, rr)
        plot_list.append(numpy.append(radius, rrs))

        # ---- option I: check if all rr across conditions within limits ---- #
        # if any(rrs > rr_up):  # if any rr oversteps threshold, print warning and break
        #     print('overshooting in %s: \nradius: %.4f, %%REC: %s' %
        #           (numpy.array(conditions)[rrs > rr_up], radius, rrs[rrs > rr_up]))
        #     break
        # if not all(ele > rr_lo and ele < rr_up for ele in rrs):  # return radius if rr limits are satisfied
        #     radius += step_size
        #     settings.neighbourhood = FixedRadius(radius)
        #     continue

        # -----  option II: return radius if mean rr within limits ---- #
        if not rr_lower < numpy.median(rrs) < rr_upper:
            radius += step_size
            settings.neighbourhood = FixedRadius(radius)
            continue
        else:
            print(f'Estimated radius: {radius}')
            if any(rrs > rr_upper):  # if any rr oversteps threshold, print warning and break
                print('overshooting in %s: \nradius: %.4f, %%REC: %s' %
                      (numpy.array(conditions)[rrs > rr_upper], radius, rrs[rrs > rr_upper]))
            break
    if plot:
        if not axis: fig, axis = plt.subplots(1, 1)
        plot_list = numpy.asarray(plot_list)
        for i in range(0, len(rrs)):
            axis.plot(plot_list[:,0], plot_list[:, i+1], label=conditions[i])
        axis.set_xlabel('radius')
        axis.set_ylabel('%REC')
        axis.legend()
    subj_dict['rqa_params']['settings'].neighbourhood = FixedRadius(radius)
    subj_dict['rqa_params']['radius'] = radius
    return subj_dict

def subject_rqa(subj_dict):
    """
    Apply RQA on each condition with selected parameters d, m, and r.
    :param subj_dict:
    :return: subject dictionary containing rqa and rp results for all conditions
    """
    conditions = subj_dict['rqa_params']['rqa_conditions']
    d = subj_dict['rqa_params']['delay']
    m = subj_dict['rqa_params']['embedding']
    settings = subj_dict['rqa_params']['settings']
    if 'radius' in subj_dict['rqa_params'].keys():
        settings.neighbourhood = FixedRadius(subj_dict['rqa_params']['radius'])
    for condition in conditions:
        if 'radii' in subj_dict['rqa_params'].keys():  # get separate radius for each condition
            settings.neighbourhood = FixedRadius(subj_dict['rqa_params']['radii'][condition])
        if isinstance(subj_dict[condition], tuple):
            data = getattr(subj_dict[condition], subj_dict['rqa_params']['metric'])
            rqa, rp = auto_rqa(data, d, m, settings)
            subj_dict[condition] = subj_dict[condition]._replace(rqa=copy.deepcopy(rqa), rp=copy.deepcopy(rp))
    return subj_dict

def auto_rqa(data, d, m, settings):
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
