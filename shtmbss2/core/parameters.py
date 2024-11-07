import inspect
from abc import ABC

from shtmbss2.common.config import *
from shtmbss2.core.logging import log
from shtmbss2.core.data import load_config, get_experiment_folder, load_yaml


class ParameterGroup:
    _to_evaluate: list = list()

    def dict(self, exclude_none=False):
        p_dict_original = self.__dict__
        p_dict = dict()
        for v_name, v_instance in vars(self).items():
            if not (v_name.startswith('_') or inspect.isfunction(v_instance)):
                if not inspect.isclass(v_instance) and issubclass(type(v_instance), ParameterGroup):
                    p_dict[v_name] = v_instance.dict(exclude_none=exclude_none)
                # ToDo: Possibly replace with more sophisticated test
                elif v_name == "network_type":
                    continue
                else:
                    if exclude_none and p_dict_original[v_name] is None:
                        continue
                    p_dict[v_name] = p_dict_original[v_name]
        return p_dict

    def evaluate(self, parameters, recursive=True):
        for param_name in self._to_evaluate:
            if hasattr(self, param_name):
                value = getattr(self, param_name)
                try:
                    value = eval(value)
                except Exception as e:
                    log.warning(f"Could not evaluate parameter {param_name}.")
                    log.warning(e)
                    value = None
                setattr(self, param_name, value)
            else:
                log.warning(f"Could not find parameter {param_name} for class {self.__str__}")

        if recursive:
            for v_name, v_instance in vars(self).items():
                if not (v_name.startswith('_') or inspect.isfunction(v_instance)):
                    if not inspect.isclass(v_instance) and issubclass(type(v_instance), ParameterGroup):
                        v_instance.evaluate(parameters, recursive=recursive)


class Parameters(ParameterGroup):
    def __init__(self, network_type):
        self.network_type = network_type
        self.config_type = None

    def get_custom_param(self, param_name):
        # Set specific parameters loaded from individual configuration
        category_objs = param_name.split('.')
        category_obj = self
        for category_name in category_objs:
            category_obj = getattr(category_obj, category_name)
        return category_obj

    def set_custom_params(self, params):
        # Set specific parameters loaded from individual configuration
        for name, value in params.items():
            category_objs = name.split('.')
            category_obj = self
            for category_name in category_objs[:-1]:
                category_obj = getattr(category_obj, category_name)
            setattr(category_obj, category_objs[-1], value)

        log.debug(f"Successfully set parameters")

    def load_default_params(self, custom_params=None):
        default_params = load_config(self.network_type, config_type=self.config_type)
        self.set_params(self, default_params)

        log.debug(f"Successfully loaded parameters for '{self.network_type}'")

        if custom_params is not None:
            self.set_custom_params(custom_params)

    def load_experiment_params(self, experiment_type, experiment_id, experiment_num, experiment_subnum=None,
                               instance_id=None, custom_params=None):
        if ((experiment_type == ExperimentType.EVAL_MULTI or experiment_type == ExperimentType.OPT_GRID_MULTI)
                and instance_id is None):
            instance_id = 0

        experiment_folder_path = get_experiment_folder(self.network_type, experiment_type, experiment_id,
                                                       experiment_num, experiment_subnum=experiment_subnum,
                                                       instance_id=instance_id)

        saved_params = load_yaml(experiment_folder_path, f"config_{self.config_type}.yaml")

        self.set_params(self, saved_params)

        if custom_params is not None:
            self.set_custom_params(custom_params)

    def set_params(self, category_obj, parameters):
        for name, value in parameters.items():
            if type(value) is dict:
                if hasattr(category_obj, name.lower()):
                    self.set_params(getattr(category_obj, name.lower()), value)
                else:
                    log.warn(f"'{category_obj}' does not have an object '{name.lower()}'")
                    continue
            else:
                if hasattr(category_obj, name):
                    setattr(category_obj, name, value)

    @staticmethod
    def print_parameter(d, name=""):
        for key, value in d.items():
            if type(value) is dict:
                if name == "":
                    new_name = key
                else:
                    new_name = f"{name}.{key}"
                Parameters.print_parameter(value, new_name)
                continue

            if name == "":
                new_name = key
            else:
                new_name = f"{name}.{key}"
            print(f"""{new_name}:
      name: ""
      description: ""
      original_value: """"")


class NetworkParameters(Parameters):
    def __init__(self, network_type):
        self.experiment = NetworkParameterGroups.Experiment()
        self.plotting = NetworkParameterGroups.Plotting()
        self.performance = NetworkParameterGroups.Performance()
        self.network = NetworkParameterGroups.Network()
        self.backend = NetworkParameterGroups.Backend()
        self.encoding = NetworkParameterGroups.Encoding()
        self.plasticity = NetworkParameterGroups.Plasticity()
        self.neurons = NetworkParameterGroups.Neurons()
        self.synapses = NetworkParameterGroups.Synapses()
        self.calibration = NetworkParameterGroups.Calibration()

        super().__init__(network_type)

        self.config_type = ConfigType.NETWORK


class PlottingParametersBase(ParameterGroup):
    def __init__(self):
        self.fontsize = PlottingParameterGroups.Fontsize()
        self.padding = PlottingParameterGroups.Padding()

        self.size: list = None
        self.dpi: int = None
        self.line_width: int = None


class PlottingParameters(Parameters):
    def __init__(self, network_type):
        self.performance = PlottingParameterGroups.Performance()
        self.events = PlottingParameterGroups.Events()

        super().__init__(network_type)

        self.config_type = ConfigType.PLOTTING


### Network Parameters ###
class NetworkParameterGroups:
    class Experiment(ParameterGroup):
        def __init__(self):
            self.type: str = None
            self.id: str = None
            self.opt_id: str = None
            self.map_name: str = None
            self.sequences: list = None
            self.seq_repetitions: int = None
            self.runtime: float = None
            self.episodes: int = None
            self.run_add_calib: bool = None
            self.save_final: bool = None
            self.save_auto: bool = None
            self.save_auto_epoches: int = None
            self.generate_rand_seed_offset: bool = None
            self.seed_offset: int = None
            self.log_weights: bool = None
            self.log_permanence: bool = None

    class Plotting(ParameterGroup):
        def __init__(self):
            self.size: list = None
            self.file_type: str = None
            self.save_figure: bool = None

    class Performance(ParameterGroup):
        def __init__(self):
            self.compute_performance: bool = None
            self.method: str = None
            self.running_avgs: list = None

    class Network(ParameterGroup):
        def __init__(self):
            self.replay_mode: str = None
            self.num_symbols: int = None
            self.num_neurons: int = None
            self.pattern_size: int = None
            self.ext_indiv: bool = None

    class Backend(ParameterGroup):
        def __init__(self):
            self.module_name: str = None
            self.neuron_name: str = None

    class Encoding(ParameterGroup):
        def __init__(self):
            self.dt_stm: float = None
            self.dt_seq: float = None
            self.t_exc_start: float = None
            self.t_scaling_factor: float = None
            self.encoding_type: str = None
            self.num_repetitions: int = None
            self.probabilities: list = None

    class Plasticity(ParameterGroup):
        def __init__(self):
            self.type: str = None
            self.enable_structured_stdp: bool = None
            self.execution_start: float = None
            self.execution_interval: float = None
            self.learning_factor: float = None
            self.learning_rate_decay: float = None
            self.weight_learning: bool = None
            self.weight_learning_scale: float = None
            self.permanence_init_min: float = None
            self.permanence_init_max: float = None
            self.permanence_max: float = None
            self.permanence_threshold: float = None
            self.correlation_threshold: int = None
            self.num_coactive_neurons: int = None
            self.w_mature: float = None
            self.y: float = None
            self.lambda_plus: float = None
            self.lambda_minus: float = None
            self.lambda_h: float = None
            self.homeostasis_depression_rate: float = None
            self.target_rate_h: float = None
            self.tau_plus: float = None
            self.tau_h: float = None
            self.delta_t_min: float = None
            self.delta_t_max: float = None
            self.dt: float = None
        _to_evaluate: list = ["w_mature"]

    class Neurons(ParameterGroup):
        def __init__(self):
            self.inhibitory = NetworkParameterGroups.Inhibitory()
            self.inhibitory_global = NetworkParameterGroups.InhibitoryGlobal()
            self.excitatory = NetworkParameterGroups.Excitatory()
            self.dendrite = NetworkParameterGroups.Dendrite()

    class Inhibitory(ParameterGroup):
        def __init__(self):
            self.c_m: float = None
            self.v_rest: float = None
            self.v_reset: float = None
            self.v_thresh: float = None
            self.tau_m: float = None
            self.tau_syn_I: float = None
            self.tau_syn_E: float = None
            self.tau_refrac: float = None

    class InhibitoryGlobal(ParameterGroup):
        def __init__(self):
            self.c_m: float = None
            self.v_rest: float = None
            self.v_reset: float = None
            self.v_thresh: float = None
            self.tau_m: float = None
            self.tau_syn_I: float = None
            self.tau_syn_E: float = None
            self.tau_refrac: float = None

    class Excitatory(ParameterGroup):
        def __init__(self):
            self.c_m: float = None
            self.v_rest: float = None
            self.v_reset: float = None
            self.v_thresh: float = None
            self.tau_m: float = None
            self.tau_syn_I: float = None
            self.tau_syn_E: float = None
            self.tau_syn_ext: float = None
            self.tau_syn_den: float = None
            self.tau_syn_inh: float = None
            self.tau_refrac: float = None

    class Dendrite(ParameterGroup):
        def __init__(self):
            self.I_p: float = None
            self.tau_dAP: float = None
            self.theta_dAP: float = None

    class Synapses(ParameterGroup):
        def __init__(self):
            self.dyn_inh_weights: bool = None
            self.dyn_weight_calculation: bool = None
            self.w_exc_inh_dyn: float = None
            self.w_ext_exc: float = None
            self.w_exc_exc: float = None
            self.w_exc_inh: float = None
            self.w_inh_exc: float = None
            self.p_exc_exc: float = None
            self.receptor_ext_exc: str = None
            self.receptor_exc_exc: str = None
            self.receptor_exc_inh: str = None
            self.receptor_inh_exc: str = None
            self.delay_ext_exc: float = None
            self.delay_exc_exc: float = None
            self.delay_exc_inh: float = None
            self.delay_inh_exc: float = None
            self.j_ext_exc_psp: float = None
            self.j_exc_inh_psp: float = None
            self.j_inh_exc_psp: float = None
        _to_evaluate: list = ["j_ext_exc_psp",
                              "j_exc_inh_psp",
                              "j_inh_exc_psp"]

    class Calibration(ParameterGroup):
        def __init__(self):
            self.v_rest_calib: float = None
            self.padi_bus_dacen_extension: int = None
            self.correlation_amplitude: float = None
            self.correlation_time_constant: int = None


### Plotting Parameters ###
class PlottingParameterGroups:
    class Performance(PlottingParametersBase):
        def __init__(self):
            super().__init__()

    class Events(PlottingParametersBase):
        def __init__(self):
            super().__init__()

    class Fontsize(ParameterGroup):
        def __init__(self):
            self.title: int = None
            self.sub_title: int = None
            self.legend: int = None
            self.axis_labels: int = None
            self.subplot_labels: int = None
            self.tick_labels: int = None

    class Padding(ParameterGroup):
        def __init__(self):
            self.subplot_labels: int = None
            self.x_axis: int = None
            self.w_space: float = None
            self.left: float = None
            self.right: float = None
            self.top: float = None
            self.bottom: float = None