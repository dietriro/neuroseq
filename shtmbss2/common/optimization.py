import warnings
import numpy as np
import itertools
import yaml
import matplotlib.pyplot as plt

from shtmbss2.core.helpers import Process
from shtmbss2.common.config import *
from shtmbss2.core.data import (load_config, get_last_experiment_num, get_experiment_folder, get_last_instance,
                                save_instance_setup, load_yaml)
from shtmbss2.common.executor import ParallelExecutor
from shtmbss2.core.parameters import NetworkParameters, PlottingParameters
from shtmbss2.core.performance import PerformanceMulti
from shtmbss2.core.logging import log

np.set_printoptions(threshold=np.inf, suppress=True, linewidth=np.inf)
warnings.filterwarnings(action='ignore', category=UserWarning)


class GridSearch:
    supported_experiment_types = [ExperimentType.OPT_GRID, ExperimentType.OPT_GRID_MULTI]

    def __init__(self, experiment_type, model_type, experiment_id, experiment_map=None, experiment_num=None):
        if experiment_type not in self.supported_experiment_types:
            log.error(f"Unsupported experiment-type selected: {experiment_type}.\n"
                      f"Please choose one of: {self.supported_experiment_types}")
            return

        # define model/experiment specific parameters
        self.model_type = model_type
        self.experiment_id = experiment_id
        self.experiment_num = experiment_num
        self.experiment_type = experiment_type
        self.experiment_map = experiment_map

        self.continuation_id = None

        # define config parameters and load config
        self.config = None
        self.parameter_matching = None
        self.fig_save = None
        self.num_instances = None
        self.order_randomization = None
        self.seed_offset = None
        self.plot_perf_dd = None

        self.init_experiment_num()

        self.load_config()

    def init_experiment_num(self):
        # retrieve experiment num for new experiment
        last_experiment_num = get_last_experiment_num(self.experiment_id, self.experiment_type, self.experiment_map)
        if self.experiment_num is None:
            self.experiment_num = last_experiment_num + 1

        if self.experiment_num <= last_experiment_num:
            self.continuation_id = get_last_instance(self.experiment_type, self.experiment_id, self.experiment_num,
                                                     experiment_map=self.experiment_map)
            if self.continuation_id > 1:
                self.continuation_id -= 1

    def load_config(self):
        if self.continuation_id is not None:
            # load config from existing experiment
            config_path = get_experiment_folder(self.experiment_type, self.experiment_id, self.experiment_num,
                                                experiment_map=self.experiment_map)
            config_name = f"config_{self.experiment_type}.yaml"
            self.config = load_yaml(config_path, config_name)
        else:
            # load default config
            self.config = load_config(self.model_type, self.experiment_type)
        self.parameter_matching = self.config["experiment"]["parameter_matching"]
        self.fig_save = self.config["experiment"]["fig_save"]
        self.num_instances = self.config["experiment"]["num_instances"]
        self.order_randomization = self.config["experiment"]["order_randomization"]
        self.seed_offset = self.config["experiment"].get("seed_offset", None)
        self.plot_perf_dd = self.config["experiment"].get("plot_perf_dd", True)

    def save_config(self):
        folder_path_experiment = get_experiment_folder(self.experiment_type, self.experiment_id, self.experiment_num,
                                                       experiment_map=self.experiment_map, instance_id=None)
        config_file_name = f"config_{self.experiment_type}.yaml"
        file_path = join(folder_path_experiment, config_file_name)
        with open(file_path, 'w') as file:
            yaml.dump(self.config, file)

    def __run_experiment(self, optimized_parameters, experiment_id, experiment_num, instance_id, steps=None,
                         optimized_parameter_ranges=None, fig_save=False, plot_perf_dd=True, save_setup=False):
        model = self.model_type(use_on_chip_plasticity=RuntimeConfig.plasticity_location == PlasticityLocation.ON_CHIP,
                                experiment_type=ExperimentType.OPT_GRID, experiment_id=experiment_id,
                                experiment_num=experiment_num, instance_id=instance_id, seed_offset=0,
                                **{**optimized_parameters, "experiment.id": experiment_id})
        model.init_backend(offset=0)

        # set save_auto to false in order to minimize file lock timeouts
        model.p.experiment.save_auto = False
        model.p.experiment.save_final = False

        if RuntimeConfig.plasticity_location == PlasticityLocation.ON_CHIP:
            model.init_plasticity_rule()

        model.init_neurons()
        model.init_connections(debug=False)
        model.init_external_input()

        if RuntimeConfig.backend == Backends.BRAIN_SCALES_2:
            model.init_rec_exc()

        model.init_prerun()

        model.run(steps=steps, plasticity_enabled=RuntimeConfig.plasticity_location != PlasticityLocation.ON_CHIP,
                  run_type=RunType.SINGLE)

        model.save_full_state(running_avg_perc=0.5, optimized_parameter_ranges=optimized_parameter_ranges,
                              save_setup=save_setup)

        # retrieve plotting parameters
        p_plot = PlottingParameters(network_type=self.model_type)
        p_plot.load_default_params()

        # save figure of performance
        if fig_save:
            fig, _ = model.performance.plot(p_plot, statistic=StatisticalMetrics.MEDIAN, fig_show=False,
                                            plot_dd=plot_perf_dd)
            figure_path = join(get_experiment_folder(self.experiment_type, self.experiment_id,
                                                     experiment_num, experiment_map=self.experiment_map,
                                                     instance_id=instance_id), "performance")
            fig.savefig(figure_path, dpi=p_plot.performance.dpi)
            plt.close(fig)

            # close figure in order to make it garbage-collectable
            plt.close(fig)

    def __run_experiment_multi(self, optimized_parameters, experiment_id, experiment_num, experiment_subnum, steps=None,
                               optimized_parameter_ranges=None, fig_save=False, seed_offset=None, plot_perf_dd=True):

        # run experiments using parallel-executor
        pe = ParallelExecutor(num_instances=self.num_instances, experiment_id=experiment_id,
                              experiment_type=self.experiment_type, experiment_map=self.experiment_map,
                              experiment_num=experiment_num, experiment_subnum=experiment_subnum,
                              parameter_ranges=optimized_parameter_ranges, fig_save=False)
        experiment_num = pe.run(steps=steps, additional_parameters=optimized_parameters, seed_offset=seed_offset,
                                plot_perf_dd=plot_perf_dd)

        # retrieve parameters for performed experiment
        p = NetworkParameters(network_type=self.model_type)
        p.load_experiment_params(experiment_type=ExperimentType.OPT_GRID_MULTI, experiment_id=self.experiment_id,
                                 experiment_subnum=experiment_subnum, experiment_num=experiment_num)

        # retrieve plotting parameters
        p_plot = PlottingParameters(network_type=self.model_type)
        p_plot.load_default_params()

        # retrieve performance data for entire set of instances for subnum
        pf = PerformanceMulti(p, self.num_instances)
        pf.load_data(self.model_type, experiment_type=ExperimentType.OPT_GRID_MULTI, experiment_id=self.experiment_id,
                     experiment_map=self.experiment_map, experiment_num=experiment_num,
                     experiment_subnum=experiment_subnum)

        # save figure of performance
        if fig_save:
            fig, _ = pf.plot(p_plot, statistic=StatisticalMetrics.MEDIAN, fig_show=False)
            figure_path = join(get_experiment_folder(self.experiment_type, self.experiment_id, experiment_num,
                                                     experiment_map=self.experiment_map,
                                                     experiment_subnum=experiment_subnum),
                               "performance")
            fig.savefig(figure_path, dpi=p_plot.performance.dpi)
            plt.close(fig)

        # save performance and parameter data to overall sheet for experiment
        save_instance_setup(net=self.model_type, parameters=p,
                            performance=pf.get_performance_dict(final_result=True,
                                                                running_avgs=p.performance.running_avgs, decimals=3),
                            experiment_num=self.experiment_num, experiment_subnum=experiment_subnum,
                            instance_id=None, optimized_parameters=optimized_parameters)

    def run(self, steps=None):
        log.handlers[LogHandler.STREAM].setLevel(logging.ESSENS)

        parameter_names = list()
        parameter_values = list()
        parameter_ranges = dict()
        for parameter_name, parameter_config in self.config["parameters"].items():
            if not parameter_config["enabled"]:
                continue
            if ("values" in parameter_config.keys() and parameter_config["values"] is not None
                    and type(parameter_config["values"]) is list and len(parameter_config["values"]) > 0):
                parameter_values.append(parameter_config["values"])
                parameter_ranges[f"optimized-params.{parameter_name}"] = parameter_config["values"]
            else:
                parameter_values.append(np.arange(start=parameter_config["min"],
                                                  stop=parameter_config["max"] + parameter_config["step"],
                                                  step=parameter_config["step"],
                                                  dtype=parameter_config["dtype"]).tolist())
                parameter_ranges[f"optimized-params.{parameter_name}"] = (parameter_config["min"],
                                                                          parameter_config["max"],
                                                                          parameter_config["step"])
            parameter_names.append(parameter_name)

        # create parameter combinations based on selected matching type of parameter values
        if self.parameter_matching == ParameterMatchingType.ALL:
            parameter_combinations = list(itertools.product(*parameter_values))
        elif self.parameter_matching == ParameterMatchingType.SINGLE:
            log.warning(f"Parameter matching type 'single' is not implemented yet. Continuing with type 'all'.")
            parameter_combinations = list(itertools.product(*parameter_values))
        else:
            log.error(f"Unknown parameter matching type '{self.parameter_matching}'.")
            return
        num_combinations = len(parameter_combinations)

        # set number of digits for file saving/loading into instance folders
        if self.experiment_type == ExperimentType.OPT_GRID:
            RuntimeConfig.instance_digits = len(str(num_combinations))
        elif self.experiment_type == ExperimentType.OPT_GRID_MULTI:
            RuntimeConfig.subnum_digits = len(str(num_combinations))



        if self.continuation_id is not None:
            log.essens(f"Continuing grid-search for {num_combinations - self.continuation_id} "
                       f"parameter combinations of {parameter_names}")
        else:
            log.essens(f"Starting grid-search for {num_combinations} parameter combinations "
                       f"of {parameter_names}")

        if self.order_randomization:
            rnd = np.random.RandomState(100)
            id_order = rnd.choice(len(parameter_combinations), len(parameter_combinations), replace=False)
        else:
            id_order = list(range(len(parameter_combinations)))

        setup_saved = False
        for run_i, param_id in enumerate(id_order):
            parameter_combination = parameter_combinations[param_id]

            if self.continuation_id is not None:
                if run_i < self.continuation_id:
                    continue
                elif run_i == self.continuation_id:
                    log.info(f"Skipped {run_i} parameter combinations. "
                             f"Continuing evaluation with combination {run_i} now.")

            log.essens(f"Starting grid-search run {run_i + 1}/{num_combinations} "
                       f"for {parameter_combination}")
            parameters = {p_name: p_value for p_name, p_value in zip(parameter_names, parameter_combination)}

            if self.experiment_type == ExperimentType.OPT_GRID:
                success = False
                while not success:
                    try:
                        proc = Process(target=self.__run_experiment, args=(parameters, self.experiment_id,
                                                                        self.experiment_num, run_i, steps,
                                                                        parameter_ranges, self.fig_save,
                                                                        self.plot_perf_dd, not setup_saved))
                        proc.start()
                        proc.join()

                        # Check if an exception occurred in the sub-process, then raise this exception
                        if proc.exception:
                            exc, trc = proc.exception
                            print(trc)
                            raise exc
                    except RuntimeError as e:
                        log.error(f"RuntimeError encountered, running experiment {run_i} again")
                        # log.error(e)
                    else:
                        success = True
            else:
                success = False
                while not success:
                    try:
                        self.__run_experiment_multi(parameters, self.experiment_id, self.experiment_num, run_i,
                                                    steps=steps, optimized_parameter_ranges=parameter_ranges,
                                                    fig_save=self.fig_save, seed_offset=self.seed_offset,
                                                    plot_perf_dd=self.plot_perf_dd)
                        success = True
                    except (RuntimeError, FileNotFoundError) as e:
                        success = False
                        log.warning(f"Encountered an error during execution. Re-running experiment with id {run_i}")

            log.essens(f"Finished grid-search run {run_i + 1}/{num_combinations}")
            log.essens(f"\tParameters: {parameter_combination}")

            setup_saved = True

            if run_i == 0:
                self.save_config()

        log.handlers[LogHandler.STREAM].setLevel(logging.INFO)
