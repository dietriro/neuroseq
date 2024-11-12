import time
import warnings
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from shtmbss2.core.helpers import Process
from shtmbss2.common.config import *
from shtmbss2.core.logging import log
from shtmbss2.core.data import get_last_experiment_num, get_experiment_folder
from shtmbss2.core.parameters import NetworkParameters, PlottingParameters
from shtmbss2.core.performance import PerformanceMulti

if RuntimeConfig.backend == Backends.BRAIN_SCALES_2:
    from shtmbss2.brainscales2.network import SHTMTotal
elif RuntimeConfig.backend == Backends.NEST:
    from shtmbss2.nest.network import SHTMTotal
else:
    raise Exception(f"Backend {RuntimeConfig.backend} not implemented yet. "
                    f"Please choose among [{Backends.BRAIN_SCALES_2}, {Backends.NEST}]")

np.set_printoptions(threshold=np.inf, suppress=True, linewidth=np.inf)
warnings.filterwarnings(action='ignore', category=UserWarning)


class ParallelExecutor:
    def __init__(self, num_instances, experiment_id, experiment_type=ExperimentType.EVAL_MULTI, experiment_map=None,
                 experiment_num=None, experiment_subnum=None, parameter_ranges=None, fig_save=False):
        self.num_instances = num_instances
        self.experiment_id = experiment_id

        self.experiment_type = experiment_type
        self.experiment_map = experiment_map
        self.experiment_num = experiment_num
        self.experiment_subnum = experiment_subnum

        self.parameter_ranges = parameter_ranges

        self.fig_save = fig_save

    # @staticmethod
    def __run_experiment(self, process_id, file_save_status, lock, experiment_type, experiment_id, experiment_num,
                         seed_offset, experiment_subnum=None, additional_parameters=None, parameter_ranges=None,
                         steps=None, p=None):
        if additional_parameters is None:
            additional_parameters = dict()

        shtm = SHTMTotal(use_on_chip_plasticity=RuntimeConfig.plasticity_location == PlasticityLocation.ON_CHIP,
                         experiment_type=experiment_type, experiment_subnum=experiment_subnum,
                         instance_id=process_id, seed_offset=seed_offset, p=p,
                         **{**additional_parameters, "experiment.id": experiment_id})
        shtm.init_backend(offset=0)

        # set save_auto to false in order to minimize file lock timeouts
        shtm.p.experiment.save_auto = False
        shtm.p.experiment.save_final = False
        shtm.experiment_num = experiment_num

        if RuntimeConfig.plasticity_location == PlasticityLocation.ON_CHIP:
            shtm.init_plasticity_rule()

        lock.acquire(block=True)
        shtm.init_neurons()
        lock.release()

        shtm.init_connections(debug=False)
        shtm.init_external_input()

        if RuntimeConfig.backend == Backends.BRAIN_SCALES_2:
            shtm.init_rec_exc()

        shtm.init_prerun()

        shtm.run(steps=steps, plasticity_enabled=RuntimeConfig.plasticity_location != PlasticityLocation.ON_CHIP,
                 run_type=RunType.SINGLE)

        # wait until it's this processes turn to save data (order)
        while process_id > 0 and file_save_status[process_id - 1] < 1:
            time.sleep(0.1)

        lock.acquire(block=True)
        shtm.save_full_state(optimized_parameter_ranges=parameter_ranges,
                             save_setup=experiment_subnum == 0 or (experiment_subnum is None and process_id == 0))
        lock.release()

        # signal other processes, that this process has finished the data saving process
        file_save_status[process_id] = 1

    def run(self, steps=None, additional_parameters=None, p=None, seed_offset=0, parallel=True, plot_perf_dd=False):

        lock = mp.Lock()
        file_save_status = mp.Array("i", [0 for _ in range(self.num_instances)])

        # retrieve experiment num for new experiment
        if self.experiment_num is None:
            self.experiment_num = get_last_experiment_num(SHTMTotal, self.experiment_id, self.experiment_type) + 1

        if seed_offset is None:
            seed_offset = int(time.time())

        log.handlers[LogHandler.STREAM].setLevel(logging.ESSENS)

        processes = []
        for i_inst in range(self.num_instances):
            if self.experiment_subnum is None:
                log.essens(f'Starting network {i_inst}')

            if parallel:
                processes.append(Process(target=self.__run_experiment, args=(i_inst, file_save_status, lock,
                                                                             self.experiment_type,
                                                                             self.experiment_id,
                                                                             self.experiment_num, seed_offset,
                                                                             self.experiment_subnum,
                                                                             additional_parameters,
                                                                             self.parameter_ranges,
                                                                             steps, p)))
                processes[i_inst].start()
            else:
                proc = Process(target=self.__run_experiment, args=(i_inst, file_save_status, lock,
                                                                   self.experiment_type,
                                                                   self.experiment_id,
                                                                   self.experiment_num, seed_offset,
                                                                   self.experiment_subnum,
                                                                   additional_parameters,
                                                                   self.parameter_ranges,
                                                                   steps, p))
                proc.start()
                proc.join()

        if parallel:
            for i_inst in range(self.num_instances):
                processes[i_inst].join()

                # Check if an exception occurred in the sub-process, then raise this exception
                if processes[i_inst].exception:
                    exc, trc = processes[i_inst].exception
                    print(trc)
                    raise exc

                if self.experiment_subnum is None:
                    log.essens(f"Finished simulation {i_inst + 1}/{self.num_instances}")

        # save figure of performance
        if self.fig_save:
            # retrieve parameters for performed experiment
            p = NetworkParameters(network_type=SHTMTotal)
            p.load_experiment_params(experiment_type=self.experiment_type, experiment_id=self.experiment_id,
                                     experiment_num=self.experiment_num)

            # retrieve plotting parameters
            p_plot = PlottingParameters(network_type=SHTMTotal)
            p_plot.load_default_params()

            # retrieve performance data for entire set of instances for subnum
            pf = PerformanceMulti(p, self.num_instances)
            pf.load_data(SHTMTotal, experiment_type=self.experiment_type, experiment_id=self.experiment_id,
                         experiment_num=self.experiment_num, experiment_map=self.experiment_map)

            # save figure
            fig, _ = pf.plot(p_plot, statistic=StatisticalMetrics.MEDIAN, fig_show=False, plot_dd=plot_perf_dd)
            figure_path = join(get_experiment_folder(self.experiment_type, self.experiment_id, self.experiment_num,
                                                     experiment_map=self.experiment_map), "performance")
            for plot_file_type in RuntimeConfig.plot_file_types:
                fig.savefig(f"{figure_path}.{plot_file_type}", dpi=p_plot.performance.dpi)
                plt.close(fig)

        log.handlers[LogHandler.STREAM].setLevel(logging.INFO)

        return self.experiment_num
