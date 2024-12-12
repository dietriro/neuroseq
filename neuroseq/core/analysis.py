from neuroseq.common.config import *
from neuroseq.core.parameters import NetworkParameters
from neuroseq.core.performance import PerformanceSingle


def get_stds_for_opt(network_type, experiment_id, experiment_num, metric, num_samples,
                     experiment_type=ExperimentType.OPT_GRID, percentile=0.75):
    """
    Calculates the standard deviation for a number of epochs starting from percentile for all selected experiments. The
    selected experiments are defined by the experiment_type, experiment_id, and experiment_num.

    :param network_type:
    :type network_type:
    :param experiment_id:
    :type experiment_id:
    :param experiment_num:
    :type experiment_num:
    :param metric:
    :type metric:
    :param num_samples:
    :type num_samples:
    :param experiment_type:
    :type experiment_type:
    :param percentile:
    :type percentile:
    :return:
    :rtype:
    """
    std_all = np.zeros(num_samples)
    for experiment_subnum in range(num_samples):
        p = NetworkParameters(network_type=network_type)
        p.load_experiment_params(experiment_type=experiment_type, experiment_id=experiment_id,
                                 experiment_num=experiment_num, experiment_subnum=experiment_subnum)

        pf = PerformanceSingle(p)
        pf.load_data(network_type, experiment_type=experiment_type, experiment_id=experiment_id,
                     experiment_num=experiment_num, experiment_subnum=experiment_subnum)

        all_mean = np.mean(pf.data[metric], axis=0)
        all_mean = all_mean[-int(len(all_mean) * percentile):]
        std = np.std(all_mean)

        std_all[experiment_subnum] = std

    return std_all
