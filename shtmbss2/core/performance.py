import os.path

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from shtmbss2.common.config import *
from shtmbss2.core.logging import log
from shtmbss2.core.parameters import NetworkParameters, PlottingParameters
from shtmbss2.core.helpers import moving_average, id_to_symbol
from shtmbss2.common.config import NeuronType
from shtmbss2.core.data import get_experiment_folder
from shtmbss2.common.plot import plot_panel_label


class Performance(ABC):
    def __init__(self, parameters, performance_type='single'):
        """

        :param parameters:
        :type parameters: NetworkParameters
        """
        self.p = parameters

        self.type = performance_type
        self.data = None

    @abstractmethod
    def init_data(self):
        pass

    @abstractmethod
    def get_statistic(self, statistic, metric, episode='all', percentile=None):
        pass

    @abstractmethod
    def get_all(self, metric, sequence_id=None):
        pass

    @abstractmethod
    def get_performance_dict(self, final_result=False):
        pass

    @abstractmethod
    def get_data_size(self):
        pass

    @abstractmethod
    def add_data_point(self, data_point, metric, sequence_id):
        pass

    def compute(self, neuron_events, method=PerformanceType.ALL_SYMBOLS, t_min=None):
        log.info(f"Computing performance for {len(self.p.experiment.sequences)} Sequences.")

        ratio_fp_activation = 0.5
        ratio_fn_activation = 0.5

        if t_min is None:
            t_min = self.p.encoding.t_exc_start

        t_min_org = t_min
        t_max_next = 0

        edge_activity = dict()
        node_activity = dict()

        for i_seq, seq in enumerate(self.p.experiment.sequences):
            seq_performance = {metric: list() for metric in PerformanceMetrics.get_all()}

            for i_element, element in enumerate(seq[1:]):
                if method == PerformanceType.LAST_SYMBOL and i_element < len(seq) - 2:
                    continue

                # define min/max for time window of spikes
                t_max = t_min + self.p.encoding.dt_stm

                log.debug(f"t_min = {t_min},  t_max = {t_max}")

                # calculate target vector
                output = np.zeros(self.p.network.num_symbols)
                target = np.zeros(self.p.network.num_symbols)
                target[SYMBOLS[element]] = 1

                num_dAPs = np.zeros(self.p.network.num_symbols)
                num_som_spikes = np.zeros(self.p.network.num_symbols)
                counter_correct = 0

                t_min_next = t_min + self.p.encoding.dt_stm
                t_max_next = t_max + self.p.encoding.dt_stm

                for i_symbol in range(self.p.network.num_symbols):
                    # get dAP's for subpopulation during current time window
                    num_dAPs[i_symbol] = np.sum([t_min < item < t_max for sublist in
                                                 neuron_events[NeuronType.Dendrite][i_symbol] for item in
                                                 sublist])

                    # get somatic spikes for subpopulation during next time window
                    num_som_spikes[i_symbol] = np.sum([t_min_next < item < t_max_next for sublist in
                                                       neuron_events[NeuronType.Soma][i_symbol] for item in
                                                       sublist])

                    if num_som_spikes[i_symbol] >= (ratio_fn_activation * self.p.network.pattern_size):
                        node_activity[id_to_symbol(i_symbol)] = SomaState.ACTIVE
                    elif id_to_symbol(i_symbol) not in node_activity.keys():
                        node_activity[id_to_symbol(i_symbol)] = SomaState.INACTIVE

                    if (i_symbol != SYMBOLS[element] and
                            num_dAPs[i_symbol] >= (ratio_fp_activation * self.p.network.pattern_size)):
                        output[i_symbol] = 1
                    elif (i_symbol == SYMBOLS[element] and
                          num_dAPs[i_symbol] >= (ratio_fn_activation * self.p.network.pattern_size)):
                        counter_correct += 1
                        output[i_symbol] = 1
                        edge_activity[(seq[i_element], element)] = DendriteState.PREDICTIVE
                    elif (i_symbol == SYMBOLS[element] and
                          0 < num_dAPs[i_symbol] < (ratio_fn_activation * self.p.network.pattern_size)):
                        edge_activity[(seq[i_element], element)] = DendriteState.WEAK


                # num_dAPs_total += num_dAPs

                # calculate Euclidean distance between output and target vector
                # determine prediction error, FP and FN
                error = np.sqrt(sum((output - target) ** 2))
                false_positive = sum(np.heaviside(output - target, 0))
                false_negative = sum(np.heaviside(target - output, 0))

                seq_performance[PerformanceMetrics.ERROR].append(error)
                seq_performance[PerformanceMetrics.FP].append(false_positive)
                seq_performance[PerformanceMetrics.FN].append(false_negative)
                seq_performance[PerformanceMetrics.ACTIVE_SOMAS].append(num_som_spikes[SYMBOLS[element]])
                # seq_performance[PerformanceMetrics.ACTIVE_DENDRITES].append(num_dAPs[SYMBOLS[element]])

                t_min += self.p.encoding.dt_stm

            # calculate dAP error

            for metric in PerformanceMetrics.get_all():
                if metric == PerformanceMetrics.DD:
                    continue
                self.add_data_point(np.mean(seq_performance[metric]), metric, sequence_id=i_seq)

            t_min += self.p.encoding.dt_seq

        # calculate dendritic duplicate data for all symbols
        num_dAPs_total = np.zeros((self.p.network.num_symbols, self.p.network.num_neurons))
        for i_symbol in range(self.p.network.num_symbols):
            for i_neuron in range(self.p.network.num_neurons):
                events_i = neuron_events[NeuronType.Dendrite][i_symbol][i_neuron]
                num_dAPs_total[i_symbol, i_neuron] += np.sum(np.logical_and(events_i > t_min_org,
                                                                            events_i < t_max_next))
        num_dAPs_sum = np.sum(np.heaviside(num_dAPs_total - 1, 0))

        # add dendritic duplicate data
        for i_seq in range(len(self.p.experiment.sequences)):
            self.add_data_point(np.mean(num_dAPs_sum), PerformanceMetrics.DD, sequence_id=i_seq)

        return node_activity, edge_activity


    def plot(self, plt_config, statistic, sequences="mean", fig_title="", plot_dd=False):
        """

        :param plot_dd:
        :type plot_dd:
        :param fig_title:
        :type fig_title:
        :param plt_config:
        :type plt_config: PlottingParameters
        :param statistic:
        :type statistic:
        :param sequences:
        :type sequences:
        :return:
        :rtype:
        """
        fig_size = plt_config.performance.size
        if plot_dd:
            fig_size[0] = fig_size[0] / 3 * 4
        fig, axs = plt.subplots(1, 3+int(plot_dd), figsize=plt_config.performance.size, dpi=plt_config.performance.dpi)

        fig.suptitle(fig_title, x=0.5, y=0.95, fontsize=plt_config.performance.fontsize.title)

        axs[0].set_ylabel("Prediction error")
        axs[0].set_xlabel("# Training Episodes")

        axs[1].set_ylabel("Rel. frequency")
        axs[1].set_xlabel("# Training Episodes")

        target_pattern_size_line = [self.p.network.pattern_size / self.p.network.num_neurons
                                    for _ in range(self.get_data_size())]
        axs[2].plot(target_pattern_size_line, linestyle="dashed", color=f"grey",
                    label=f"Target pattern size ({self.p.network.pattern_size})")
        axs[2].set_ylabel("Rel. no. of active neurons")
        axs[2].set_xlabel("# Training Episodes")

        panels = ['A', 'B', 'C']

        if plot_dd:
            axs[3].set_ylabel("No. of duplicate dAP's")
            axs[3].set_xlabel("# Training Episodes")

            panels.append('D')

        for i_plot, letter in enumerate(panels):
            panel_label_pos = (-0.05, 1.14)
            plot_panel_label(letter, panel_label_pos, axs[i_plot], size=plt_config.performance.fontsize.sub_title)

            axs[i_plot].title.set_size(plt_config.performance.fontsize.sub_title)
            axs[i_plot].xaxis.label.set_size(plt_config.performance.fontsize.axis_labels)
            axs[i_plot].yaxis.label.set_size(plt_config.performance.fontsize.axis_labels)
            axs[i_plot].tick_params(axis='both', labelsize=plt_config.performance.fontsize.tick_labels)

        return fig, axs

    def plot_seq(self, axs, plt_config, perf_errors, perf_fps, perf_fns, num_active_somas_post, perf_dds, plot_dd=False,
                 i_col=1):
        # Plot 1: Performance error
        axs[0].plot(moving_average(perf_errors), color=f"C{i_col}", linewidth=plt_config.performance.line_width)

        # Plot 2: False positives/negatives
        axs[1].plot(moving_average(perf_fps), color=f"C{i_col}", label="False-positives",
                    linewidth=plt_config.performance.line_width)
        axs[1].plot(moving_average(perf_fns), linestyle="dashed", color=f"C{i_col}", label="False-negatives",
                    linewidth=plt_config.performance.line_width)

        # Plot 3: Number of active neurons
        rel_num_active_neurons = moving_average(np.array(num_active_somas_post) / self.p.network.num_neurons)
        axs[2].plot(rel_num_active_neurons, color=f"C{i_col}", linewidth=plt_config.performance.line_width)

        if plot_dd:
            axs[3].plot(moving_average(perf_dds), color=f"C{i_col}", label="Dendritic duplicates",
                        linewidth=plt_config.performance.line_width)

        return axs

    def plot_bounds(self, axs, error_low, error_up, fp_low, fp_up, fn_low, fn_up, active_somas_low, active_somas_up,
                    dd_low, dd_up, plot_dd=False):
        x = range(self.get_data_size())

        axs[0].fill_between(x, moving_average(error_low), moving_average(error_up), facecolor='grey')

        axs[1].fill_between(x, moving_average(fp_low), moving_average(fp_up), facecolor='grey')
        axs[1].fill_between(x, moving_average(fn_low), moving_average(fn_up), facecolor='grey')

        active_somas_low = moving_average(np.array(active_somas_low) / self.p.network.num_neurons)
        active_somas_up = moving_average(np.array(active_somas_up) / self.p.network.num_neurons)
        axs[2].fill_between(x, active_somas_low, active_somas_up, facecolor='grey')

        if plot_dd:
            axs[3].fill_between(x, moving_average(dd_low), moving_average(dd_up), facecolor='grey')

        return axs

    def plot_legend(self, axs, plt_config):
        """

        :param axs:
        :type axs:
        :param plt_config:
        :type plt_config: PlottingParameters
        :return:
        :rtype:
        """
        axs[0].legend(["Prediction error"], fontsize=plt_config.performance.fontsize.legend)
        axs[1].legend(["False-positives", "False-negatives"], fontsize=plt_config.performance.fontsize.legend)
        axs[2].legend(["Target", "Actual"], fontsize=plt_config.performance.fontsize.legend)
        axs[3].legend(["Duplicate dAP's"], fontsize=plt_config.performance.fontsize.legend)

        return axs

    def load_data(self, net, experiment_type, experiment_id, experiment_num, experiment_subnum=None, instance_id=None):
        folder_path = get_experiment_folder(net, experiment_type, experiment_id, experiment_num,
                                            experiment_subnum=experiment_subnum, instance_id=instance_id)

        file_path = join(folder_path, "performance.npz")
        data_performance = np.load(file_path)

        for metric_name in data_performance.files:
            if instance_id is not None and type(self) is PerformanceMulti:
                self.data[instance_id][metric_name] = data_performance[metric_name].tolist()
            else:
                self.data[metric_name] = data_performance[metric_name].tolist()



class PerformanceSingle(Performance):
    def __init__(self, parameters):
        super().__init__(parameters)

        self.init_data()

    def init_data(self):
        self.data = dict()
        for metric_name in PerformanceMetrics.get_all():
            self.data[metric_name] = [[] for _ in self.p.experiment.sequences]

    def load_data(self, net, experiment_type, experiment_id, experiment_num, experiment_subnum=None, instance_id=None):
        self.init_data()
        super().load_data(net, experiment_type, experiment_id, experiment_num, experiment_subnum=experiment_subnum,
                          instance_id=instance_id)

    def get_statistic(self, statistic, metric, episode='all', percentile=None):
        if 'all' in episode:
            kwargs = {'axis': 0}
            if statistic == StatisticalMetrics.PERCENTILE and percentile is not None:
                kwargs['q'] = percentile
            result = NP_STATISTICS[statistic](np.array(self.data[metric]), **kwargs)
            if episode == 'all-mean':
                return np.mean(result, **kwargs)
            else:
                return result
        elif episode == 'last':
            return NP_STATISTICS[statistic](np.array(self.data[metric])[:, -1])

    def get_all(self, metric, sequence_id=None):
        if sequence_id is None:
            return self.data[metric]
        else:
            return self.data[metric][sequence_id]

    def get_performance_dict(self, final_result=False, running_avgs=None, decimals=5):
        if running_avgs is None:
            running_avgs = [0.5]
        performance = dict()
        if final_result:
            # find first training epoch from which error didn't fall below 0 anymore
            error = self.get_all(PerformanceMetrics.ERROR)
            error_max_reverse = np.flip(np.max(error, axis=0))
            error_max_reverse = np.cumsum(error_max_reverse)
            first_zero_id = np.argmax(error_max_reverse > 0)
            if first_zero_id == 0:
                # learning not finished yet
                performance[f"num-epochs"] = -1
            else:
                performance[f"num-epochs"] = len(error_max_reverse) - first_zero_id
            # add data from all metrics
            for metric_name in PerformanceMetrics.get_all():
                metric = self.get_all(metric_name)
                metric_means = np.mean(metric, axis=0)

                # add mean of all training epochs to dict
                for running_avg in running_avgs:
                    start_id = int(len(metric_means) * (1-running_avg))
                    end_id = len(metric_means)
                    performance[f"{metric_name}_running-avg-{running_avg}"] = (
                        np.round(np.mean(metric_means[start_id:end_id]), decimals))

                # add final value to dict
                performance[f"{metric_name}_last"] = np.round(metric_means[-1], decimals)
        else:
            performance = self.data
        return performance

    def get_data_size(self):
        size = -1
        for metric_i in self.data.values():
            for seq_i in metric_i:
                if size < 0 or len(seq_i) < size:
                    size = len(seq_i)
        return size

    def add_data_point(self, data_point, metric, sequence_id):
        self.data[metric][sequence_id].append(data_point)

    def plot(self, plt_config, statistic=StatisticalMetrics.MEAN, sequences="statistic", fig_show=False, plot_dd=False):
        fig, axs = super().plot(plt_config, statistic=statistic, sequences=sequences, plot_dd=plot_dd)

        sequence_range = None

        if type(sequences) is str:
            if sequences == "statistic":
                axs = self.plot_seq(axs, plt_config, self.get_statistic(statistic, PerformanceMetrics.ERROR),
                                    self.get_statistic(statistic, PerformanceMetrics.FP),
                                    self.get_statistic(statistic, PerformanceMetrics.FN),
                                    self.get_statistic(statistic, PerformanceMetrics.ACTIVE_SOMAS),
                                    self.get_statistic(statistic, PerformanceMetrics.DD), i_col=1, plot_dd=plot_dd)
            elif sequences == "all":
                sequence_range = range(len(self.p.experiment.sequences))
        elif type(sequences) in [range, list]:
            sequence_range = sequences

        if sequence_range is not None:
            for i_seq in sequence_range:
                self.plot_seq(axs, plt_config, self.get_all(PerformanceMetrics.ERROR, i_seq),
                              self.get_all(PerformanceMetrics.FP, i_seq),
                              self.get_all(PerformanceMetrics.FN, i_seq),
                              self.get_all(PerformanceMetrics.ACTIVE_SOMAS, i_seq),
                              self.get_statistic(statistic, PerformanceMetrics.DD), i_col=1, plot_dd=plot_dd)

        axs = self.plot_legend(axs, plt_config=plt_config)

        if fig_show:
            fig.show()

        return fig, axs


class PerformanceMulti(Performance):
    def __init__(self, parameters, num_instances):
        super().__init__(parameters)

        self.num_instances = num_instances

        self.init_data()

    def init_data(self):
        self.data = list()
        for i_inst in range(self.num_instances):
            data_inst = dict()
            for metric_name in PerformanceMetrics.get_all():
                data_inst[metric_name] = [[] for _ in self.p.experiment.sequences]
            self.data.append(data_inst)

    def load_data(self, net, experiment_type, experiment_id, experiment_num, experiment_subnum=None, instance_id=None):
        self.init_data()

        folder_path = get_experiment_folder(net, experiment_type, experiment_id, experiment_num,
                                            experiment_subnum=experiment_subnum)

        for i_inst in range(self.num_instances):
            inst_folder_name = f"{i_inst:02d}"
            inst_folder_path = join(folder_path, inst_folder_name)

            if not os.path.exists(inst_folder_path):
                raise FileNotFoundError(f"Instance folder does not exist: {inst_folder_path}")

            super().load_data(net, experiment_type, experiment_id, experiment_num, experiment_subnum=experiment_subnum,
                              instance_id=i_inst)

    def get_statistic(self, statistic, metric, episode='all', percentile=None):
        if 'all' in episode:
            kwargs = {'axis': 0}
            if statistic == StatisticalMetrics.PERCENTILE and percentile is not None:
                kwargs['q'] = percentile
            mean_inst = [np.mean(np.array(self.data[i_inst][metric]), axis=0) for i_inst in range(self.num_instances)]
            result = NP_STATISTICS[statistic](mean_inst, **kwargs)
            if episode == 'all-mean':
                return np.mean(result)
            else:
                return result
        elif episode == 'last':
            return NP_STATISTICS[statistic]([NP_STATISTICS[statistic](np.array(self.data[metric])[:, -1])
                                             for _ in range(self.num_instances)])

    def get_all(self, metric, sequence_id=None, instance_id=None):
        if instance_id is None:
            log.error("Instance id cannot be non for class PerformanceMulti.")
            return
        if sequence_id is None:
            return self.data[instance_id][metric]
        else:
            return self.data[instance_id][metric][sequence_id]

    def get_performance_dict(self, final_result=False, running_avgs=None, decimals=5):
        if running_avgs is None:
            running_avgs = [0.5]
        performance = dict()
        first_zero_finished = False
        performance[f"num-epochs_avg"] = 0
        if final_result:
            for metric_name in PerformanceMetrics.get_all():
                for i_inst in range(self.num_instances):
                    metric = self.get_all(metric_name, instance_id=i_inst)
                    metric_means = np.mean(metric, axis=0)

                    # add running avgs to dict
                    for running_avg in running_avgs:
                        start_id = int(len(metric_means) * (1 - running_avg))
                        end_id = len(metric_means)
                        metric_running_avg = np.mean(metric_means[start_id:end_id])
                        # add mean of all training epochs to dict
                        performance[f"{metric_name}_running-avg-{running_avg}"] = (metric_running_avg +
                                                    performance.get(f"{metric_name}_running-avg-{running_avg}", 0))

                    # add final value to dict
                    metric_last = metric_means[-1]
                    performance[f"{metric_name}_last"] = performance.get(f"{metric_name}_last", 0) + metric_last

                # compute average of running avgs
                for running_avg in running_avgs:
                    performance[f"{metric_name}_running-avg-{running_avg}"] = np.round(
                            performance[f"{metric_name}_running-avg-{running_avg}"] / self.num_instances, decimals)

                # compute average of last metric value
                performance[f"{metric_name}_last"] = np.round(performance[f"{metric_name}_last"] / self.num_instances,
                                                              decimals)

            num_success = 0
            for i_inst in range(self.num_instances):
                # find first training epoch from which error didn't fall below 0 anymore
                error = self.get_all(PerformanceMetrics.ERROR, instance_id=i_inst)
                error_max_reverse = np.flip(np.max(error, axis=0))
                error_max_reverse = np.cumsum(error_max_reverse)
                first_zero_id = np.argmax(error_max_reverse > 0)
                if first_zero_id > 0:
                    performance[f"num-epochs_avg"] = (performance.get(f"num-epochs_avg", 0) +
                                                      len(error_max_reverse) - first_zero_id)
                    num_success += 1

            # compute average of num-epochs
            if performance.get(f"num-epochs_avg", 0) == 0 or num_success < self.num_instances / 2:
                performance[f"num-epochs_avg"] = -1
            else:
                performance[f"num-epochs_avg"] = int(np.round(performance[f"num-epochs_avg"] /
                                                              num_success))

        else:
            performance = self.data
        return performance

    def get_data_size(self):
        size = -1
        for inst_i in self.data:
            for metric_i in inst_i.values():
                for seq_i in metric_i:
                    if size < 0 or len(seq_i) < size:
                        size = len(seq_i)
        return size

    def add_data_point(self, data_point, metric, sequence_id, instance_id=None):
        self.data[instance_id][metric][sequence_id].append(data_point)

    def plot(self, plt_config, statistic=StatisticalMetrics.MEDIAN, sequences=None, instances="statistic",
             fig_show=True, fig_title="", plot_dd=False):
        fig, axs = super().plot(plt_config, statistic=statistic, sequences=sequences, fig_title=fig_title,
                                plot_dd=plot_dd)

        axs = self.plot_seq(axs, plt_config, self.get_statistic(statistic, PerformanceMetrics.ERROR),
                            self.get_statistic(statistic, PerformanceMetrics.FP),
                            self.get_statistic(statistic, PerformanceMetrics.FN),
                            self.get_statistic(statistic, PerformanceMetrics.ACTIVE_SOMAS),
                            self.get_statistic(statistic, PerformanceMetrics.DD), i_col=1, plot_dd=plot_dd)
        if statistic == StatisticalMetrics.MEDIAN:
            axs = self.plot_bounds(axs,
                                   error_low=self.get_statistic(StatisticalMetrics.PERCENTILE, PerformanceMetrics.ERROR,
                                                                percentile=5),
                                   error_up=self.get_statistic(StatisticalMetrics.PERCENTILE, PerformanceMetrics.ERROR,
                                                               percentile=95),
                                   fp_low=self.get_statistic(StatisticalMetrics.PERCENTILE, PerformanceMetrics.FP,
                                                             percentile=5),
                                   fp_up=self.get_statistic(StatisticalMetrics.PERCENTILE, PerformanceMetrics.FP,
                                                            percentile=95),
                                   fn_low=self.get_statistic(StatisticalMetrics.PERCENTILE, PerformanceMetrics.FN,
                                                             percentile=5),
                                   fn_up=self.get_statistic(StatisticalMetrics.PERCENTILE, PerformanceMetrics.FN,
                                                            percentile=95),
                                   active_somas_low=self.get_statistic(StatisticalMetrics.PERCENTILE,
                                                                       PerformanceMetrics.ACTIVE_SOMAS,
                                                                       percentile=5),
                                   active_somas_up=self.get_statistic(StatisticalMetrics.PERCENTILE,
                                                                      PerformanceMetrics.ACTIVE_SOMAS,
                                                                      percentile=95),
                                   dd_low=self.get_statistic(StatisticalMetrics.PERCENTILE, PerformanceMetrics.DD,
                                                             percentile=5),
                                   dd_up=self.get_statistic(StatisticalMetrics.PERCENTILE, PerformanceMetrics.DD,
                                                            percentile=95),
                                   plot_dd=plot_dd
                                   )

        axs = self.plot_legend(axs, plt_config=plt_config)

        if fig_show:
            fig.show()

        # plt.subplots_adjust(wspace=0.3, bottom=0.15, top=0.8, left=0.05, right=0.98)

        plt.subplots_adjust(wspace=plt_config.performance.padding.w_space,
                            bottom=plt_config.performance.padding.bottom,
                            top=plt_config.performance.padding.top,
                            left=plt_config.performance.padding.left,
                            right=plt_config.performance.padding.right,
                            )

        return fig, axs
