import time
import copy
import pickle
import yaml
import sys
import multiprocessing as mp
import itertools as it
import glob
import sys


from types import ModuleType, FunctionType
from gc import get_referents
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from pyNN.random import NumpyRNG
from tabulate import tabulate
from abc import abstractmethod
from copy import deepcopy

from neuroseq.common.config import *
from neuroseq.common import learning
from neuroseq.core.logging import log
from neuroseq.core.parameters import NetworkParameters, PlottingParameters
from neuroseq.core.performance import PerformanceSingle
from neuroseq.core.helpers import (Process, id_to_symbol, calculate_trace,
                                   psp_max_2_psc_max)
from neuroseq.common.plot import plot_dendritic_events
from neuroseq.core.data import (save_experimental_setup, save_instance_setup, get_experiment_folder,
                                get_experiment_file)
from neuroseq.core.map import Map, LabelTypes

if RuntimeConfig.backend == Backends.BRAIN_SCALES_2:
    import pynn_brainscales.brainscales2 as pynn

    from pynn_brainscales.brainscales2.populations import Population, PopulationView
    from pynn_brainscales.brainscales2.connectors import AllToAllConnector, FixedNumberPreConnector
    from pynn_brainscales.brainscales2.standardmodels.cells import SpikeSourceArray
    from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse
    from pynn_brainscales.brainscales2.projections import Projection
elif RuntimeConfig.backend == Backends.NEST:
    import pyNN.nest as pynn

    from pyNN.nest.populations import Population, PopulationView
    from pyNN.nest.connectors import AllToAllConnector, FixedNumberPreConnector
    from pyNN.nest.standardmodels.cells import SpikeSourceArray
    from pyNN.nest.standardmodels.synapses import StaticSynapse
    from pyNN.nest.projections import Projection
else:
    raise Exception(f"Backend {RuntimeConfig.backend} not implemented yet. "
                    f"Please choose among [{Backends.BRAIN_SCALES_2}, {Backends.NEST}]")

ID_PRE = 0
ID_POST = 1
NON_PICKLE_OBJECTS = ["post_somas", "projection", "shtm"]

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


class SHTMBase(ABC):
    def __init__(self, experiment_type=ExperimentType.EVAL_SINGLE, experiment_id=None, experiment_num=None,
                 experiment_subnum=None, instance_id=None, seed_offset=None, p=None, **kwargs):
        if experiment_type == ExperimentType.OPT_GRID:
            self.optimized_parameters = kwargs
        else:
            self.optimized_parameters = None

        # Define passed variables
        self.experiment_id = experiment_id
        self.experiment_num = experiment_num
        self.experiment_subnum = experiment_subnum
        self.experiment_episodes = 0
        self.instance_id = instance_id
        self.network_mode = NetworkState.PREDICTIVE

        # Load pre-defined parameters
        self.p_plot: PlottingParameters = PlottingParameters(network_type=self)
        if p is None:
            self.p: NetworkParameters = NetworkParameters(network_type=self)
            self.load_params(experiment_type, experiment_id, experiment_num, instance_id, **kwargs)
        else:
            self.p: NetworkParameters = deepcopy(p)
        self.p.experiment.type = experiment_type

        if RuntimeConfig.plasticity_location != self.p.plasticity.location:
            self.p.plasticity.location = RuntimeConfig.plasticity_location

        # Load map data and create new map
        self.map = Map(self.p.experiment.map_name, self.p.experiment.sequences, save_history=True)

        # Declare neuron populations
        self.neurons_exc = None
        self.neurons_inh = None
        self.neurons_inh_global = None
        self.neurons_ext = None
        self.neurons_add = None

        # Declare connections
        self.ext_to_exc = None
        self.exc_to_exc = None
        self.exc_to_inh = None
        self.inh_to_exc = None
        self.inh_to_inh_global = None
        self.inh_to_exc_global = None

        # Declare recordings
        self.rec_neurons_exc = None
        self.spike_times_ext = None
        self.spike_times_ext_indiv = None
        self.max_spike_time = None
        self.last_ext_spike_time = None
        self.neuron_events = None
        self.neuron_events_hist = None
        self.neuron_thresholds_hist = None

        self.run_state = False
        self.target = None

        self.performance = PerformanceSingle(parameters=self.p)

        if seed_offset is None:
            if self.p.experiment.generate_rand_seed_offset:
                self.p.experiment.seed_offset = int(time.time())
            elif self.p.experiment.seed_offset is None:
                self.p.experiment.seed_offset = 0
        else:
            self.p.experiment.seed_offset = seed_offset

        if self.p.experiment.type in [ExperimentType.EVAL_MULTI, ExperimentType.EVAL_SINGLE,
                                      ExperimentType.OPT_GRID_MULTI]:
            instance_offset = self.instance_id if self.instance_id is not None else 0
        else:
            instance_offset = 0
        np.random.seed(self.p.experiment.seed_offset + instance_offset)

        self.random_seed = self.p.experiment.seed_offset + instance_offset

    def load_params(self, experiment_type, experiment_id, experiment_num, instance_id, **kwargs):
        self.p_plot.load_default_params(network_mode=self.network_mode, map_name=self.p.experiment.map_name)
        if experiment_type == ExperimentType.OPT_GRID and instance_id > 0:
            self.p.load_experiment_params(experiment_type=ExperimentType.OPT_GRID, experiment_id=experiment_id,
                                          experiment_num=experiment_num, experiment_subnum=0,
                                          custom_params=kwargs)
        else:
            self.p.load_default_params(custom_params=kwargs)

            self.p.evaluate(parameters=self.p, recursive=True)

            if self.p.plasticity.tau_h is None:
                self.p.plasticity.tau_h = self.__compute_time_constant_dendritic_rate(dt_stm=self.p.encoding.dt_stm,
                                                                                      dt_seq=self.p.encoding.dt_seq,
                                                                                      target_firing_rate=self.p.plasticity.y
                                                                                      )

            # dynamically calculate new weights, scale by 1/1000 for "original" pynn-nest neurons
            if self.p.synapses.dyn_weight_calculation:
                self.p.synapses.w_ext_exc = psp_max_2_psc_max(self.p.synapses.j_ext_exc_psp,
                                                              self.p.neurons.excitatory.tau_m,
                                                              self.p.neurons.excitatory.tau_syn_ext,
                                                              self.p.neurons.excitatory.c_m) / 1000
                self.p.synapses.w_exc_inh = psp_max_2_psc_max(self.p.synapses.j_exc_inh_psp,
                                                              self.p.neurons.inhibitory.tau_m,
                                                              self.p.neurons.inhibitory.tau_syn_E,
                                                              self.p.neurons.inhibitory.c_m)
                self.p.synapses.w_inh_exc = abs(psp_max_2_psc_max(self.p.synapses.j_inh_exc_psp,
                                                                  self.p.neurons.excitatory.tau_m,
                                                                  self.p.neurons.excitatory.tau_syn_inh,
                                                                  self.p.neurons.excitatory.c_m)) / 1000

        # check if number of symbols is high enough
        max_symbol = id_to_symbol(self.p.network.num_symbols)
        for seq_i in self.p.experiment.sequences:
            max_symbol = max(seq_i + [max_symbol])
        if max_symbol > id_to_symbol(self.p.network.num_symbols):
            log.warning(f"The number of symbols used in sequences exceeds the number of symbols specified "
                        f"({SYMBOLS[max_symbol]} > {self.p.network.num_symbols}).\n"
                        "Setting the number of symbols to the maximum value used.")
            self.p.network.num_symbols = SYMBOLS[max_symbol] + 1

    def init_network(self):
        self.init_neurons()
        self.init_connections()
        self.init_external_input()

    def init_neurons(self):
        self.neurons_exc = self.init_all_neurons_exc()

        self.neurons_inh = self.init_neurons_inh()
        self.neurons_inh_global = self.init_neurons_inh(num_neurons=1,
                                                        tau_refrac=self.p.neurons.inhibitory_global.tau_refrac)

        if self.p.network.ext_indiv:
            self.neurons_ext = [Population(self.p.network.num_neurons, SpikeSourceArray())
                                for _ in range(self.p.network.num_symbols)]
        else:
            self.neurons_ext = Population(self.p.network.num_symbols, SpikeSourceArray())

    @abstractmethod
    def init_all_neurons_exc(self, num_neurons=None):
        pass

    @abstractmethod
    def init_neurons_exc(self, num_neurons=None):
        pass

    @staticmethod
    def init_neurons_exc_post_preprocess(dendrites, somas):
        for i in range(len(dendrites)):
            dendrites.actual_hwparams[i].multicompartment.enable_conductance = True
            dendrites.actual_hwparams[i].multicompartment.i_bias_nmda = 120
            dendrites.actual_hwparams[i].multicompartment.connect_soma_right = True
            dendrites.actual_hwparams[i].refractory_period.reset_holdoff = 0

        for i in range(len(somas)):
            somas.actual_hwparams[i].multicompartment.connect_soma = True

    @abstractmethod
    def init_neurons_inh(self, num_neurons=None, tau_refrac=None):
        pass

    def init_external_input(self, init_recorder=False, init_performance=False):
        if self.p.network.ext_indiv:
            spike_times = [[list() for _ in range(self.p.network.num_neurons)]
                           for _ in range(self.p.network.num_symbols)]
        else:
            spike_times = [list() for _ in range(self.p.network.num_symbols)]
        spike_times_sym = [list() for _ in range(self.p.network.num_symbols)]
        spike_time = None

        if self.p.encoding.encoding_type == EncodingType.PROBABILISTIC:
            seq_distribution = np.round(
                np.array(self.p.encoding.probabilities) * self.p.encoding.num_repetitions).astype(int)
            if np.sum(seq_distribution) > self.p.encoding.num_repetitions:
                log.warn(f"Accumulated sum of repetitions per sequence exceeds total number of repetitions "
                         f"({np.sum(seq_distribution)} > {self.p.encoding.num_repetitions}).")

        starting_symbols = {sym: 0 for sym in SYMBOLS.keys()}
        sequence_offset = self.p.encoding.t_exc_start
        for _ in range(self.p.encoding.num_repetitions):
            if self.p.encoding.encoding_type == EncodingType.PROBABILISTIC:
                i_seq = np.random.choice(len(self.p.experiment.sequences), p=self.p.encoding.probabilities)
                sequences = [self.p.experiment.sequences[i_seq]]
            else:
                sequences = copy.copy(self.p.experiment.sequences)

            for i_seq, sequence in enumerate(sequences):
                for i_element, element in enumerate(sequence):
                    if self.network_mode == NetworkState.REPLAY:
                        if i_element == 0 and self.p.network.replay_mode == ReplayMode.PARALLEL:
                            spike_time = self.p.encoding.t_exc_start
                        elif i_element == 0 and self.p.network.replay_mode == ReplayMode.CONSECUTIVE:
                            spike_time = sequence_offset + i_element * self.p.encoding.dt_stm
                        else:
                            break
                    else:
                        spike_time = sequence_offset + i_element * self.p.encoding.dt_stm

                    if self.p.network.ext_indiv:
                        if i_element == 0:
                            range_start = starting_symbols[element] * self.p.network.pattern_size
                            range_end = (starting_symbols[element] + 1) * self.p.network.pattern_size

                            if range_start > self.p.network.num_neurons or range_end > self.p.network.num_neurons:
                                log.error(f"Neuron range for ext_indiv [{range_start}, {range_end}] is out of bounds"
                                          f"for num_neurons={self.p.network.num_neurons}.")
                                raise Exception(f"Neuron range for ext_indiv [{range_start}, {range_end}] is out of "
                                                f"bounds for num_neurons={self.p.network.num_neurons}.")

                            neuron_range = range(range_start, range_end)
                            starting_symbols[element] += 1
                        else:
                            neuron_range = range(self.p.network.num_neurons)
                        for i_neuron in neuron_range:
                            if 0 <= i_neuron < self.p.network.num_neurons:
                                spike_times[SYMBOLS[element]][i_neuron].append(spike_time)
                            else:
                                log.warning(f"Tried to create spike times for neuron id [{i_neuron}], which is out of "
                                            f"range for num_neurons={self.p.network.num_neurons}. Skipping spike.")
                    else:
                        spike_times[SYMBOLS[element]].append(spike_time)
                    spike_times_sym[SYMBOLS[element]].append(spike_time)
                sequence_offset = spike_time + self.p.encoding.dt_seq

        self.last_ext_spike_time = max([max(s, default=0) for s in spike_times_sym], default=0)

        log.debug(f'Spike times:')
        for i_letter, letter_spikes in enumerate(spike_times_sym):
            log.debug(f'{list(SYMBOLS.keys())[i_letter]}: {spike_times_sym[i_letter]}')

        for i_sym in range(self.p.network.num_symbols):
            if self.p.network.ext_indiv:
                self.neurons_ext[i_sym].set(spike_times=spike_times[i_sym])
            else:
                self.neurons_ext[i_sym:i_sym + 1].set(spike_times=spike_times[i_sym])

        self.spike_times_ext = spike_times_sym
        self.spike_times_ext_indiv = spike_times

        if init_performance:
            log.info(f'Initialized external input for sequence(s) {self.p.experiment.sequences}')
            # Initialize performance containers
            self.performance.init_data()

    def init_connections(self, exc_to_exc=None, exc_to_inh=None):
        self.ext_to_exc = []
        for i in range(self.p.network.num_symbols):
            if self.p.network.ext_indiv:
                neurons_ext_i = self.neurons_ext[i]
                connector = pynn.OneToOneConnector()
            else:
                neurons_ext_i = PopulationView(self.neurons_ext, [i])
                connector = pynn.AllToAllConnector()

            self.ext_to_exc.append(Projection(
                neurons_ext_i,
                self.get_neurons(NeuronType.Soma, symbol_id=i),
                connector,
                synapse_type=StaticSynapse(weight=self.p.synapses.w_ext_exc, delay=self.p.synapses.delay_ext_exc),
                receptor_type=self.p.synapses.receptor_ext_exc))

        self.exc_to_exc = []
        num_connections = int(self.p.network.num_neurons * self.p.synapses.p_exc_exc)
        i_w = 0
        for i in range(self.p.network.num_symbols):
            for j in range(self.p.network.num_symbols):
                if i == j:
                    continue
                if exc_to_exc is not None:
                    weight = exc_to_exc[i_w]
                    weight[np.isnan(weight)] = 0
                elif not self.p.plasticity.enable_structured_stdp:
                    weight = np.random.uniform(self.p.plasticity.permanence_init_min * 0.01,
                                               self.p.plasticity.permanence_init_max * 0.01,
                                               size=(self.p.network.num_neurons, self.p.network.num_neurons))
                else:
                    weight = self.p.synapses.w_exc_exc
                seed = j + i * self.p.network.num_symbols + self.p.experiment.seed_offset
                if self.instance_id is not None:
                    seed += self.instance_id * self.p.network.num_symbols ** 2
                self.exc_to_exc.append(Projection(
                    self.get_neurons(NeuronType.Soma, symbol_id=i),
                    self.get_neurons(NeuronType.Dendrite, symbol_id=j),
                    FixedNumberPreConnector(num_connections, rng=NumpyRNG(seed=j + i * self.p.network.num_symbols)),
                    synapse_type=StaticSynapse(weight=weight, delay=self.p.synapses.delay_exc_exc),
                    receptor_type=self.p.synapses.receptor_exc_exc,
                    label=f"exc-exc_{id_to_symbol(i)}>{id_to_symbol(j)}"))
                i_w += 1

        self.exc_to_inh = []
        for i in range(self.p.network.num_symbols):
            weight = self.p.synapses.w_exc_inh if exc_to_inh is None else exc_to_inh[i]
            self.exc_to_inh.append(Projection(
                self.get_neurons(NeuronType.Soma, symbol_id=i),
                PopulationView(self.neurons_inh, [i]),
                AllToAllConnector(),
                synapse_type=StaticSynapse(weight=weight, delay=self.p.synapses.delay_exc_inh),
                receptor_type=self.p.synapses.receptor_exc_inh))

        self.inh_to_exc = []
        for i in range(self.p.network.num_symbols):
            self.inh_to_exc.append(Projection(
                PopulationView(self.neurons_inh, [i]),
                self.get_neurons(NeuronType.Soma, symbol_id=i),
                AllToAllConnector(),
                synapse_type=StaticSynapse(weight=self.p.synapses.w_inh_exc, delay=self.p.synapses.delay_inh_exc),
                receptor_type=self.p.synapses.receptor_inh_exc))

        self.inh_to_inh_global = []
        for i in range(self.p.network.num_symbols):
            self.inh_to_inh_global.append(Projection(
                PopulationView(self.neurons_inh, [i]),
                self.neurons_inh_global,
                AllToAllConnector(),
                synapse_type=StaticSynapse(weight=0),
                receptor_type=self.p.synapses.receptor_exc_inh))

        self.inh_to_exc_global = []
        for i in range(self.p.network.num_symbols):
            inh_to_exc_global_i = list()
            for k_symbol in range(self.p.network.num_symbols):
                inh_to_exc_global_i.append(Projection(
                    self.neurons_inh_global,
                    self.get_neurons(NeuronType.Soma, symbol_id=k_symbol),
                    AllToAllConnector(),
                    synapse_type=StaticSynapse(weight=self.p.synapses.w_inh_exc),
                    receptor_type=self.p.synapses.receptor_inh_exc))
            self.inh_to_exc_global.append(inh_to_exc_global_i)

    def init_prerun(self):
        pass

    def init_backend(self, **kwargs):
        pass

    def __compute_time_constant_dendritic_rate(self, dt_stm, dt_seq, target_firing_rate, calibration=0):
        """ Adapted from Bouhadjour et al. 2022
        Compute time constant of the dendritic AP rate,

        The time constant is set such that the rate captures how many dAPs a neuron generated
        all along the period of a batch

        Parameters
        ----------
        calibration : float
        target_firing_rate : float

        Returns
        -------
        float
           time constant of the dendritic AP rate
        """

        t_exc = (((len(self.p.experiment.sequences[0]) - 1) * dt_stm + dt_seq + calibration)
                 * len(self.p.experiment.sequences))

        log.debug("\nDuration of a sequence set %d ms" % t_exc)

        return target_firing_rate * t_exc

    @abstractmethod
    def reset(self, store_to_cache=False):
        pass

    def set_state(self, new_network_mode, target=None):
        self.target = target
        # define new neuron/network params based on network-state
        if new_network_mode == NetworkState.PREDICTIVE:
            v_thresh = self.p.neurons.excitatory.v_thresh
            theta_dAP = self.p.neurons.dendrite.theta_dAP
            weight_factor = 1
            self.p.replay.target = None
        elif new_network_mode == NetworkState.REPLAY:
            v_thresh = self.p.replay.v_thresh
            theta_dAP = self.p.replay.theta_dAP
            weight_factor = self.p.replay.weight_factor_exc_inh
            self.neuron_events_hist = list()
            self.neuron_thresholds_hist = list()
            self.p.replay.target = target
        else:
            return
        self.experiment_episodes = 0
        self.p.experiment.episodes = 0

        # update neuron parameters
        for i_sym in range(len(self.neurons_exc)):
            if target is not None and target == i_sym:
                v_thresh_tmp = v_thresh * self.p.replay.scaling_target
            else:
                v_thresh_tmp = v_thresh
            self.neurons_exc[i_sym].set(V_th=v_thresh_tmp,
                                        theta_dAP=theta_dAP)

        # update exc-inh weights
        if self.p.synapses.dyn_weight_calculation:
            self.p.synapses.j_exc_inh_psp = (1.2 * self.p.neurons.inhibitory.v_thresh /
                                             self.p.network.pattern_size / weight_factor)
            self.p.synapses.w_exc_inh = psp_max_2_psc_max(self.p.synapses.j_exc_inh_psp,
                                                          self.p.neurons.inhibitory.tau_m,
                                                          self.p.neurons.inhibitory.tau_syn_E,
                                                          self.p.neurons.inhibitory.c_m)
        for exc_to_inh in self.exc_to_inh:
            weights = np.full(exc_to_inh.get("weight", format="array").shape, self.p.synapses.w_exc_inh)
            exc_to_inh.set(weight=weights)

        # for new network (v2)
        for inh_global in self.inh_to_inh_global:
            weights = np.full(inh_global.get("weight", format="array").shape,
                              self.p.synapses.w_exc_inh * self.p.network.pattern_size * 2)
            inh_global.set(weight=weights)

        self.network_mode = new_network_mode
        RuntimeConfig.file_prefix = self.network_mode

        if self.network_mode == NetworkState.REPLAY:
            self.neuron_thresholds_hist.append(
                [self.neurons_exc[i_sym].get("V_th") for i_sym in range(self.p.network.num_symbols)])

        self.map.reset_graph_history()

        # reload plotting parameters for new network state
        self.p_plot.load_default_params(network_mode=self.network_mode, map_name=self.p.experiment.map_name)

    def update_adapt_thresholds(self, num_active_neuron_thresh=None):
        if num_active_neuron_thresh is None:
            num_active_neuron_thresh = self.p.network.pattern_size * 1.5

        con_id = self.p.network.num_symbols - 1
        # compare the spike times of all pre/post neurons. If a pre-neuron caused a post-neuron to fire (i.e. spiked
        # within a window before the post and has a connection to post) then set a trace-offset and reduce the adaptive
        # threshold subsequently.
        for i_sym in range(1, self.p.network.num_symbols):
            num_active_neurons = 0
            trace_offset = 1.0
            spikes_i = self.neuron_events[NeuronType.Soma][i_sym]
            num_active_cons = 0
            for k_sym in range(self.p.network.num_symbols):
                if i_sym == k_sym:
                    continue
                weights = self.exc_to_exc[con_id].get("weight", format="array")
                spikes_k = self.neuron_events[NeuronType.Soma][k_sym]
                num_active_neurons = 0
                for i_neuron, spikes_i_i in enumerate(spikes_i):
                    num_active_neurons += int(len(spikes_i_i) > 0)
                    if num_active_cons >= self.p.network.pattern_size:
                        continue
                    for k_neuron, spikes_k_k in enumerate(spikes_k):
                        if not np.isnan(weights[i_neuron, k_neuron]) and weights[i_neuron, k_neuron] > 0:
                            delta_t = np.array([comb_i[0] - comb_i[1] for comb_i in it.product(spikes_k_k, spikes_i_i)])
                            log.debug(f"delta_t[{id_to_symbol(i_sym)}, {id_to_symbol(k_sym)}: {delta_t}")
                            # The value 56 is suited for theta_dAP = 59 and v_thresh = 6.5
                            if np.any((delta_t < self.p.replay.threshold_delta_t_up) & (delta_t > 4)):
                                log.info(f"delta_t[{id_to_symbol(i_sym)}, {id_to_symbol(k_sym)}: {delta_t}")
                                trace_offset = self.p.replay.scaling_trace
                                num_active_cons += 1
                                break
                con_id += 1

            V_th = self.neurons_exc[i_sym].get("V_th")
            V_th_new = V_th * trace_offset

            if self.target is None:
                # calculate a value representing the difference between the number of active neurons
                # and the set threshold
                ambig_perc = self.p.network.pattern_size / self.p.network.num_neurons
                perc_act_neurons = num_active_neurons / self.p.network.num_neurons

                if perc_act_neurons >= ambig_perc:
                    ambig_offset = np.e ** (-8 * (perc_act_neurons - ambig_perc)) * self.p.replay.max_scaling_loc
                else:
                    ambig_offset = np.e ** (20 * (perc_act_neurons - ambig_perc)) * self.p.replay.max_scaling_loc

                V_th_new *= (1 - ambig_offset)

                log.debug(f"[{id_to_symbol(i_sym)}]  N_act_neu = {num_active_neurons},  "
                          f"target_offset' = {ambig_offset}")

            self.neurons_exc[i_sym].set(V_th=V_th_new)
            if V_th != V_th_new:
                log.debug(f"[{id_to_symbol(i_sym)}]  V_th = {V_th},  V_th' = {V_th_new}")

    def run_sim(self, runtime):
        pynn.run(runtime)
        self.run_state = True

    @abstractmethod
    def get_neurons(self, neuron_type, symbol_id=None):
        pass

    @abstractmethod
    def get_neuron_data(self, neuron_type, neurons=None, value_type="spikes", symbol_id=None, neuron_id=None,
                        runtime=None, dtype=None):
        pass

    @staticmethod
    def add_offset_to_events(events, offset):
        """
        Adds the defined offset in ms to a list of events for a list of neurons.

        :param events: The list of events grouped in [neurons][events per neuron].
        :type events: list
        :param offset: The offset in ms.
        :type offset: float
        :return: The list of events grouped in [neurons][events per neuron] with added offset.
        :rtype: list
        """
        for i in range(len(events)):
            if len(events[i]) > 0:
                if type(events[i]) == np.ndarray:
                    events[i] += offset
                elif type(events[i]) == list:
                    events[i] = [event + offset for event in events[i]]
                else:
                    events[i] += offset * events[i].units
        return events

    def plot_events(self, neuron_types="all", symbols="all", size=None, x_lim_lower=None, x_lim_upper=None, seq_start=0,
                    seq_end=None, fig_title="", file_path=None, run_id=None, show_grid=False, separate_seqs=False,
                    replay_runtime=None, plot_dendritic_trace=True, enable_y_ticks=True, x_tick_step=None,
                    plot_thresholds=False, large_layout=False, custom_labels=None):
        if size is None:
            size = self.p_plot.events.size

        if self.network_mode == NetworkState.REPLAY and replay_runtime is None:
            log.error("Replay runtime not set. Aborting.")
            return

        if type(neuron_types) is str and neuron_types == "all":
            neuron_types = [NeuronType.Dendrite, NeuronType.Soma, NeuronType.Inhibitory, NeuronType.InhibitoryGlobal]
        elif type(neuron_types) is list:
            pass
        else:
            return

        if run_id is None:
            if x_lim_lower is None:
                x_lim_lower = 0.
            if x_lim_upper is None:
                x_lim_upper = self.p.experiment.runtime
        else:
            single_run_length = self.calc_runtime_single() - self.p.encoding.t_exc_start
            if x_lim_lower is None:
                x_lim_lower = run_id * single_run_length
            if x_lim_upper is None:
                x_lim_upper = (run_id + 1) * single_run_length - self.p.encoding.dt_seq * 0.9
                if self.network_mode == NetworkState.REPLAY:
                    x_lim_upper = self.max_spike_time + self.p.encoding.t_exc_start

        if type(symbols) is str and symbols == "all":
            symbols = range(self.p.network.num_symbols)
        elif type(symbols) is list:
            pass

        if custom_labels is None:
            custom_labels = list()
        custom_labels = [{"color": "C7", "label": "External"}] + custom_labels

        if not separate_seqs:
            n_cols = 1
        elif self.network_mode == NetworkState.REPLAY:
            n_cols = len(self.neuron_events_hist)
        else:
            n_cols = len(self.p.experiment.sequences)

        n_rows = len(symbols)

        if plot_thresholds:
            fig = plt.figure(figsize=size)
            subfigs = fig.subfigures(2, 1,
                                     hspace=-0.1,
                                     height_ratios=[
                                         (n_rows - self.p_plot.events.padding.threshold_ratio) / (n_rows + 1),
                                         (1 + self.p_plot.events.padding.threshold_ratio) / (n_rows + 1)])

            axs = subfigs[0].subplots(nrows=len(symbols), ncols=n_cols, sharex="col", sharey="row")
            axs_th = subfigs[1].subplots(nrows=1, ncols=n_cols + 1, sharey="row")

        else:
            fig, axs = plt.subplots(self.p.network.num_symbols, n_cols, sharex="col", sharey="row", figsize=size)

        if seq_end is None:
            seq_end = seq_start + self.p.experiment.runtime

        ax = None
        y_label = "no-data"

        for i_seq in range(n_cols):
            spike_offset = 0
            neuron_events = self.neuron_events
            if n_cols > 1:
                if self.network_mode == NetworkState.PREDICTIVE:
                    x_lim_upper = (x_lim_lower + (len(self.p.experiment.sequences[i_seq]) - 0.5)
                                   * self.p.encoding.dt_stm + self.p.encoding.t_exc_start)
                else:
                    spike_offset = i_seq * (replay_runtime + self.p.encoding.dt_seq)
                    x_lim_upper = x_lim_lower + replay_runtime
                    neuron_events = self.neuron_events_hist[i_seq]

            for i_symbol in symbols:
                if len(symbols) == 1:
                    ax = axs[i_seq]
                elif n_cols == 1:
                    ax = axs[i_symbol]
                else:
                    ax = axs[i_symbol, i_seq]

                for neurons_i in neuron_types:
                    # Retrieve and plot spikes from selected neurons
                    spikes = deepcopy(neuron_events[neurons_i][i_symbol])
                    if neurons_i == NeuronType.Inhibitory:
                        spikes.append([])
                    elif neurons_i == NeuronType.InhibitoryGlobal:
                        if self.network_mode == NetworkState.PREDICTIVE:
                            continue
                        for _ in range(self.p.network.num_neurons + 1):
                            spikes.insert(0, [])
                    else:
                        spikes.insert(0, [])
                    if spike_offset > 0:
                        spikes = self.add_offset_to_events(spikes, spike_offset)
                    if neurons_i == NeuronType.Dendrite:
                        spikes_post = deepcopy(neuron_events[NeuronType.Soma][i_symbol])
                        if spike_offset > 0:
                            spikes_post = self.add_offset_to_events(spikes_post, spike_offset)
                        plot_dendritic_events(ax, spikes[1:], spikes_post,
                                              tau_dap=self.p.neurons.dendrite.tau_dAP * self.p.encoding.t_scaling_factor,
                                              color=f"C{neurons_i.COLOR_ID}", label=neurons_i.get_name_print(),
                                              seq_start=seq_start + spike_offset, seq_end=seq_end + spike_offset,
                                              epoch_end=x_lim_upper, plot_trace=plot_dendritic_trace)
                    else:
                        ax.eventplot(spikes, linewidths=self.p_plot.events.events.width,
                                     linelengths=self.p_plot.events.events.height,
                                     label=neurons_i.get_name_print(), color=f"C{neurons_i.COLOR_ID}")

                # plot external spikes as reference lines
                # for i_sym in range(self.p.network.num_symbols):
                if self.p.network.ext_indiv:
                    spikes_ext_i = deepcopy(self.spike_times_ext_indiv[i_symbol])
                    spikes_ext_i.insert(0, [])
                    if spike_offset > 0:
                        spikes_ext_i = self.add_offset_to_events(spikes_ext_i, spike_offset)
                    # add negative offset to external spikes during replay for better visibility
                    if self.network_mode == NetworkState.REPLAY:
                        spikes_ext_i = self.add_offset_to_events(spikes_ext_i, -4)
                    ax.eventplot(spikes_ext_i, linewidths=self.p_plot.events.events.width,
                                 linelengths=self.p_plot.events.events.height, label="External", color=f"grey")
                else:
                    for spike_time_ext_sym_i in self.spike_times_ext[i_symbol]:
                        ax.plot([spike_time_ext_sym_i, spike_time_ext_sym_i], [0.6, self.p.network.num_neurons + 0.4],
                                c="grey",
                                label="External")

                # Configure the plot layout
                ax.set_xlim(x_lim_lower, x_lim_upper)
                # increase upper/lower space if large_layout is set - increases visibility of local/global inh.
                if large_layout:
                    ax.set_ylim(-3, self.p.network.num_neurons + 1 + int(self.network_mode == NetworkState.REPLAY) + 2)
                else:
                    ax.set_ylim(-1, self.p.network.num_neurons + 1 + int(self.network_mode == NetworkState.REPLAY))

                if i_seq < 1:
                    ax.set_ylabel(id_to_symbol(i_symbol), weight='bold',
                                  fontsize=self.p_plot.events.fontsize.subplot_labels)

                    # set ticks for y-axis only if enabled
                    if enable_y_ticks:
                        y_label = "Symbol & Neuron ID"
                        if self.network_mode == NetworkState.REPLAY and separate_seqs:
                            ax.tick_params(axis='y', labelsize=self.p_plot.events.fontsize.tick_labels)
                        else:
                            ax.yaxis.set_ticks(range(self.p.network.num_neurons + 2
                                                     + int(self.network_mode == NetworkState.REPLAY)))
                            # Generate y-tick-labels based on number of neurons per symbol
                            y_tick_labels = ['Inh', '', '0'] + ['' for _ in range(self.p.network.num_neurons - 2)] + [
                                str(self.p.network.num_neurons - 1)]
                            if self.network_mode == NetworkState.REPLAY:
                                y_tick_labels += ['Inh-G']
                            ax.set_yticklabels(y_tick_labels, rotation=45,
                                               fontsize=self.p_plot.events.fontsize.tick_labels)
                    else:
                        y_label = "Neuronal Subpopulations (Locations)"
                        # for major ticks
                        ax.set_yticks([])
                        # for minor ticks
                        ax.set_yticks([], minor=True)

                if show_grid:
                    ax.grid(True, which='both', axis='both')

                if (x_lim_upper - x_lim_lower) / self.p.encoding.dt_stm > 200:
                    log.info("Minor ticks not set because the number of ticks would be too high.")
                elif (x_lim_upper - x_lim_lower) / self.p.encoding.dt_stm < 15:
                    if x_tick_step is None:
                        x_tick_step = self.p.encoding.dt_stm / 2
                    ax.xaxis.set_ticks(np.arange(x_lim_lower, x_lim_upper, x_tick_step))

                if self.network_mode == NetworkState.REPLAY and n_cols > 1 and i_symbol == 0:
                    ax.set_title(f"Replay {i_seq + 1}", fontsize=self.p_plot.events.fontsize.subplot_labels,
                                 pad=self.p_plot.events.padding.subplot_title)

            ax.tick_params(axis='x', labelsize=self.p_plot.events.fontsize.tick_labels)

            if self.network_mode == NetworkState.REPLAY and n_cols > 1:
                x_lim_lower += replay_runtime + self.p.encoding.dt_seq
            else:
                x_lim_lower += (len(
                    self.p.experiment.sequences[i_seq]) - 1) * self.p.encoding.dt_stm + self.p.encoding.dt_seq

        # plot updated thresholds before/after replays if enabled
        if plot_thresholds:
            for i_rep in range(n_cols + 1):
                ax_th = axs_th[i_rep]
                # plot bars for thresholds
                if i_rep == 0:
                    height_prev = np.full(self.p.network.num_symbols, self.p.replay.v_thresh)
                    ax_th.bar(range(self.p.network.num_symbols), color="lightgrey",
                              width=self.p_plot.thresholds.events.width,
                              height=np.full(self.p.network.num_symbols, self.p.replay.v_thresh))
                    ax_th.yaxis.set_ticks([4.5, 5.5, 6.5])
                    ax_th.yaxis.set_tick_params(labelsize=self.p_plot.thresholds.fontsize.tick_labels)
                    ax_th.set_ylim(4., 7)

                    ax_th.set_title(f"Initial", fontsize=self.p_plot.thresholds.fontsize.tick_labels,
                                    pad=self.p_plot.thresholds.padding.subplot_title)
                else:
                    height_prev = self.neuron_thresholds_hist[i_rep - 1]

                    ax_th.set_title(f"After replay {i_rep}",
                                    fontsize=self.p_plot.thresholds.fontsize.title,
                                    pad=self.p_plot.thresholds.padding.subplot_title)

                heights_back = np.max(np.array([self.neuron_thresholds_hist[i_rep], height_prev]), axis=0)
                heights_front = np.min(np.array([self.neuron_thresholds_hist[i_rep], height_prev]), axis=0)

                ax_th.bar(range(self.p.network.num_symbols), height=heights_back,
                          width=self.p_plot.thresholds.events.width, color="lightgrey")
                ax_th.bar(range(self.p.network.num_symbols), height=heights_front,
                          width=self.p_plot.thresholds.events.width, color="grey")

                # set tick labels
                ax_th.xaxis.set_ticks(symbols)
                y_tick_labels_th = [id_to_symbol(k_sym) for k_sym in symbols]
                ax_th.set_xticklabels(y_tick_labels_th,
                                      fontsize=self.p_plot.thresholds.fontsize.tick_labels)

                if i_rep > 0:
                    ax_th.yaxis.set_visible(False)

        if n_cols > 1:
            plt.subplots_adjust(wspace=self.p_plot.events.padding.w_space)

        # figure title
        fig.suptitle(fig_title, x=self.p_plot.events.location.title_x, y=self.p_plot.events.location.title_y,
                     fontsize=self.p_plot.events.fontsize.title)

        # x-axis label
        fig.text(self.p_plot.events.location.label_xaxis_x, self.p_plot.events.location.label_xaxis_y, "Time [ms]",
                 ha="center", fontsize=self.p_plot.events.fontsize.axis_labels)

        # y-axis label
        fig.text(self.p_plot.events.location.label_yaxis_x, self.p_plot.events.location.label_yaxis_y, y_label,
                 va="center", rotation="vertical", fontsize=self.p_plot.events.fontsize.axis_labels)

        if plot_thresholds:
            fig.text(self.p_plot.events.location.label_yaxis_x, 0.12, "Thresholds",
                     va="center", rotation="vertical", fontsize=self.p_plot.events.fontsize.axis_labels)

        # create custom legend for all plots
        custom_lines = [Line2D([0], [0], color=f"C{n.COLOR_ID}", label=n.NAME_PRINT, lw=4)
                        for n in neuron_types]
        if custom_labels is not None:
            for custom_label in custom_labels:
                custom_lines.append(Line2D([0], [0], lw=4, **custom_label))
        plt.figlegend(handles=custom_lines,
                      loc=(self.p_plot.events.location.legend_x, self.p_plot.events.location.legend_y),
                      ncol=len(custom_lines), labelspacing=0.2, fontsize=self.p_plot.events.fontsize.legend,
                      fancybox=True, borderaxespad=4, handlelength=1)

        # figure geometry
        plt.subplots_adjust(
            bottom=self.p_plot.events.padding.bottom,
            top=self.p_plot.events.padding.top,
        )

        if plot_thresholds:
            subfigs[1].subplots_adjust(
                bottom=self.p_plot.thresholds.padding.bottom,
            )

        # save figure if enabled
        if file_path is not None:
            plt.savefig(f"{file_path}.pdf")
            plt.savefig(f"{file_path}.svg")

            pickle.dump(fig, open(f'{file_path}.fig.pickle', 'wb'))
        else:
            fig.show()

    def plot_v_exc(self, alphabet_range, neuron_range='all', size=None, neuron_type=NeuronType.Soma, runtime=None,
                   show_legend=False, file_path=None):
        if size is None:
            size = (12, 10)

        if type(neuron_range) is str and neuron_range == 'all':
            neuron_range = range(self.p.network.num_neurons)
        elif type(neuron_range) is list or type(neuron_range) is range:
            pass
        else:
            return

        if type(runtime) is str:
            if str(runtime).lower() == 'max':
                runtime = self.last_ext_spike_time + (self.p.encoding.dt_seq - self.p.encoding.t_exc_start)
        elif type(runtime) is float or type(runtime) is int:
            pass
        else:
            runtime = self.p.experiment.runtime

        spike_times = [[]]
        header_spikes = list()

        fig, ax = plt.subplots(figsize=size)

        for alphabet_id in alphabet_range:
            # retrieve and save spike times
            spikes = self.neuron_events[neuron_type][alphabet_id]
            for neuron_id in neuron_range:
                # add spikes to list for printing
                spike_times[0].append(np.array(spikes[neuron_id]).round(5).tolist())
                header_spikes.append(f"{id_to_symbol(alphabet_id)}[{neuron_id}]")

                # retrieve voltage data
                data_v = self.get_neuron_data(neuron_type, value_type=RecTypes.V, symbol_id=alphabet_id,
                                              neuron_id=neuron_id, runtime=runtime)

                ax.plot(data_v.times, data_v, alpha=0.5, label=header_spikes[-1])

        # ax.xaxis.set_ticks(np.arange(0.02, 0.06, 0.01))
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)

        ax.set_xlabel("Time [ms]", labelpad=14, fontsize=26)
        ax.set_ylabel("Membrane Voltage [a.u.]", labelpad=14, fontsize=26)

        ax.set_xlim(0, runtime)

        if show_legend:
            plt.legend()

        # Print spike times
        print(tabulate(spike_times, headers=header_spikes) + '\n')

        fig.show()

        if file_path is not None:
            plt.savefig(f"{file_path}.pdf")

            pickle.dump(fig, open(f'{file_path}.fig.pickle',
                                  'wb'))  # This is for Python 3 - py2 may need `file` instead of `open`

    def plot_performance(self, statistic=StatisticalMetrics.MEAN, sequences="statistic", plot_dd=False):
        self.performance.plot(self.p_plot, statistic=statistic, sequences=sequences, plot_dd=plot_dd)

    def times_to_list(self, times, i_sym, ratio_fn_activation=0.5):
        # List element: tuple(avg_spike_time, i_sym, num_spikes)
        times_list = list()

        i_sym_times_sum = np.ediff1d(times)
        group_ids = np.where(i_sym_times_sum > 4)[0]
        group_ids += 1
        last_group_start = 0
        for group_id in group_ids:
            if len(times[last_group_start:group_id]) >= ratio_fn_activation * self.p.network.pattern_size:
                times_list.append((np.mean(times[last_group_start:group_id]), i_sym, group_id - last_group_start))
            last_group_start = group_id
        if len(times) > 0:
            if len(times[last_group_start:]) >= ratio_fn_activation * self.p.network.pattern_size:
                times_list.append((np.mean(times[last_group_start:]), i_sym, len(times) - last_group_start))
        else:
            times_list.append((0, i_sym, 0))

        return times_list

    def get_activity(self, ratio_fn_activation=0.5):
        edge_activity = dict()
        node_activity = dict()

        soma_times = list()
        dendrite_times = list()
        for i_sym in range(self.p.network.num_symbols):
            i_sym_times_soma = list()
            i_sym_times_dendrite = list()
            for i_neuron in range(self.p.network.num_neurons):
                i_sym_times_soma += list(self.neuron_events[NeuronType.Soma][i_sym][i_neuron])
                i_sym_times_dendrite += list(self.neuron_events[NeuronType.Dendrite][i_sym][i_neuron])

            soma_times += self.times_to_list(np.sort(np.array(i_sym_times_soma)), i_sym,
                                             ratio_fn_activation=ratio_fn_activation)
            dendrite_times += self.times_to_list(np.sort(np.array(i_sym_times_dendrite)), i_sym,
                                                 ratio_fn_activation=ratio_fn_activation)

        soma_times = [(a, b, c) for a, b, c in soma_times if a > 0]
        dendrite_times = [(a, b, c) for a, b, c in dendrite_times if a > 0]
        soma_times.sort()
        dendrite_times.sort()
        num_weights_thresh = 1

        for i_soma, soma_i in enumerate(soma_times[:-1]):
            node_activity[id_to_symbol(soma_i[1])] = SomaState.ACTIVE
            for i_dend, dend_i in enumerate(dendrite_times):
                for k_soma, soma_k in enumerate(soma_times[i_soma + 1:]):
                    if soma_k[1] != dend_i[1]:
                        continue
                    # ToDO: remove once self connections have been established
                    if soma_i[1] == soma_k[1]:
                        continue
                    if not 4 < soma_k[0] - dend_i[0] < self.p.neurons.dendrite.tau_dAP:
                        break
                    # get weights for connection from soma_i to soma_i+1
                    i_con = soma_i[1] * (self.p.network.num_symbols - 1) + soma_k[1] - (
                        1 if soma_k[1] > soma_i[1] else 0)
                    weights = self.exc_to_exc[i_con].get("weight", format="array")
                    num_active_connections = np.sum(
                        np.ceil(np.nansum(weights, axis=0) / self.p.plasticity.w_mature) > num_weights_thresh)

                    if soma_i[0] < dend_i[0] < soma_k[0] and \
                            num_active_connections >= ratio_fn_activation * self.p.network.pattern_size:
                        sym_pre = id_to_symbol(soma_i[1])
                        sym_post = id_to_symbol(soma_k[1])
                        if dend_i[2] >= self.p.network.pattern_size:
                            edge_activity[(sym_pre, sym_post)] = DendriteState.PREDICTIVE
                        elif dend_i[2] > 0:
                            edge_activity[(sym_pre, sym_post)] = DendriteState.WEAK

            node_activity[id_to_symbol(soma_times[-1][1])] = SomaState.ACTIVE

        return node_activity, edge_activity

    def getsize(self):
        """sum size of object & members."""
        if isinstance(self, BLACKLIST):
            raise TypeError('getsize() does not take argument of type: ' + str(type(self)))
        seen_ids = set()
        size = 0
        objects = [self]
        while objects:
            need_referents = []
            for obj in objects:
                if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                    seen_ids.add(id(obj))
                    size += sys.getsizeof(obj)
                    need_referents.append(obj)
            objects = get_referents(*need_referents)
        return size

    def calc_runtime_single(self):
        runtime_single = self.p.encoding.t_exc_start
        for seq in self.p.experiment.sequences:
            for element in seq[:-1]:
                runtime_single += self.p.encoding.dt_stm
            runtime_single += self.p.encoding.dt_seq
        return runtime_single

    def calc_num_correlated_events(self, neuron_type_a, neuron_type_b, symbol_id, t_max, t_min=0):
        """
        Calculate number of correlated events within neurons of symbol_id

        :param neuron_type_a:
        :type neuron_type_a:
        :param neuron_type_b:
        :type neuron_type_b:
        :param symbol_id:
        :type symbol_id:
        :param t_max:
        :type t_max:
        :param t_min:
        :type t_min:
        :return:
        :rtype:
        """

        events_a = self.neuron_events[neuron_type_a][symbol_id]
        events_b = self.neuron_events[neuron_type_b][symbol_id]

        num_events = np.zeros(self.p.network.num_neurons)
        for i_neuron, events_a_i in enumerate(events_a):
            events_b_i = events_b[i_neuron]
            for event_a_i in events_a_i:
                if event_a_i > self.p.plasticity.execution_start:
                    break
                diff_arr = events_b_i - event_a_i
                if len(np.where(np.logical_and(diff_arr > t_min, diff_arr < t_max))[0]) > 0:
                    num_events[i_neuron] += 1

        return num_events

    def print_thresholds(self, symbols=None):
        if symbols is None:
            symbols = list(range(self.p.network.num_symbols))
        print("Membrane thresholds:")
        for sym_i in symbols:
            if type(sym_i) is str:
                sym_i = SYMBOLS[sym_i]

            print(f"{id_to_symbol(sym_i)}: {self.neurons_exc[sym_i].get('V_th')}")

    def __str__(self):
        return type(self).__name__


class SHTMTotal(SHTMBase, ABC):
    def __init__(self, experiment_type=ExperimentType.EVAL_SINGLE, experiment_id=None, experiment_num=None,
                 experiment_subnum=None, plasticity_cls=None, instance_id=None, seed_offset=None, p=None, **kwargs):
        super().__init__(experiment_type=experiment_type, experiment_id=experiment_id, experiment_num=experiment_num,
                         experiment_subnum=experiment_subnum, instance_id=instance_id, seed_offset=seed_offset, p=p,
                         **kwargs)

        self.con_plastic = None
        self.trace_dendrites = self.trace_dendrites = np.zeros(shape=(self.p.network.num_symbols,
                                                                      self.p.network.num_neurons))

        if plasticity_cls is None:
            self.plasticity_cls = learning.Plasticity
        else:
            self.plasticity_cls = plasticity_cls

        if self.p.experiment.log_permanence is None or not self.p.experiment.log_permanence:
            self.log_permanence = list()
            self.p.experiment.log_permanence = False
        else:
            self.log_permanence = range(self.p.network.num_symbols ** 2 - self.p.network.num_symbols)

        if self.p.experiment.log_weights is None or not self.p.experiment.log_weights:
            self.log_weights = list()
            self.p.experiment.log_weights = None
        else:
            self.log_weights = range(self.p.network.num_symbols ** 2 - self.p.network.num_symbols)

    def init_connections(self, exc_to_exc=None, exc_to_inh=None, debug=False):
        super().init_connections(exc_to_exc=exc_to_exc, exc_to_inh=exc_to_inh)

        self.con_plastic = list()

        for i_plastic in range(len(self.exc_to_exc)):
            # Retrieve id (letter) of post synaptic neuron population
            symbol_post = self.exc_to_exc[i_plastic].label.split('_')[1].split('>')[1]
            # Create population view of all post synaptic somas
            post_somas = PopulationView(self.get_neurons(NeuronType.Soma, symbol_id=SYMBOLS[symbol_post]),
                                        list(range(self.p.network.num_neurons)))
            if self.p.synapses.dyn_inh_weights:
                proj_post_soma_inh = self.exc_to_inh[SYMBOLS[symbol_post]]
            else:
                proj_post_soma_inh = None

            self.con_plastic.append(self.plasticity_cls(self.exc_to_exc[i_plastic], post_somas=post_somas, shtm=self,
                                                        proj_post_soma_inh=proj_post_soma_inh, index=i_plastic,
                                                        debug=debug, **self.p.plasticity.dict()))

        for i_perm in self.log_permanence:
            self.con_plastic[i_perm].enable_permanence_logging()
        # ToDo: Find out why accessing the weights before first simulation causes switch in connections between symbols.
        # This seems to be a nest issue.
        # for i_perm in self.log_weights:
        #     self.con_plastic[i_perm].enable_weights_logging()

    def print_permanence_diff(self):
        for i_perm in self.log_permanence:
            permanence_diff = self.con_plastic[i_perm].permanences[-1] - self.con_plastic[i_perm].permanences[0]
            print(
                f"Permanence diff for {self.con_plastic[i_perm].projection.label} ({i_perm}): {list(permanence_diff)}")

    def plot_permanence_diff(self):
        fig, axs = plt.subplots(len(self.log_permanence), 1, sharex="all", figsize=(22, 7))

        all_connection_ids = set()
        for i_perm in self.log_permanence:
            connection_ids = self.con_plastic[i_perm].get_all_connection_ids()
            all_connection_ids.update(connection_ids)
        all_connection_ids = sorted(all_connection_ids)

        for i_perm in self.log_permanence:
            permanence_diff = self.con_plastic[i_perm].permanences[-1] - self.con_plastic[i_perm].permanences[0]
            connection_ids = self.con_plastic[i_perm].get_all_connection_ids()

            all_permanence_diff = np.zeros(len(all_connection_ids))
            for i_all_cons, con_ids in enumerate(all_connection_ids):
                if con_ids in connection_ids:
                    all_permanence_diff[i_all_cons] = permanence_diff[connection_ids.index(con_ids)]

            colors = ['C0' if p >= 0 else 'C1' for p in all_permanence_diff]

            axs[i_perm].bar(range(len(all_connection_ids)), all_permanence_diff, color=colors)
            axs[i_perm].set_ylabel(self.con_plastic[i_perm].projection.label.split('_')[1], weight='bold')
            y_min = round(min(all_permanence_diff), 1)
            y_max = round(max(all_permanence_diff), 1)
            if y_min == y_max == 0:
                axs[i_perm].yaxis.set_ticks([0.0])
            else:
                axs[i_perm].yaxis.set_ticks([y_min, y_max])
            axs[i_perm].xaxis.set_ticks(range(len(all_connection_ids)), all_connection_ids, rotation=45)
            axs[i_perm].grid(True, which='both', axis='both')

        axs[-1].set_xlabel("Connection [#]")
        fig.tight_layout(pad=1.0)
        fig.text(0.02, 0.5, "Permanence diff / connection direction", va="center", rotation="vertical")

    def plot_permanence_history(self, plot_con_ids='all'):
        if type(plot_con_ids) is str and plot_con_ids == 'all':
            plot_con_ids = self.log_permanence
        elif type(plot_con_ids) is list:
            pass
        else:
            return

        fig, axs = plt.subplots(len(plot_con_ids), 1, sharex="all", figsize=(14, len(plot_con_ids) * 4))

        for i_plot, i_con in enumerate(plot_con_ids):
            permanences = np.array(self.con_plastic[i_con].permanences)

            ind = np.where(np.sum(permanences == 0, axis=0) <= 1)[0]
            permanences = permanences[:, ind].tolist()

            permanences_plot = list()
            for i_perm in range(len(permanences)):
                if not np.equal(permanences[i_perm], 0).all():
                    permanences_plot.append(permanences[i_perm])

            # Plot all previous permanences as a line over time
            axs[i_plot].plot(range(len(permanences_plot)), permanences_plot)

            if '>' in self.con_plastic[i_con].projection.label:
                y_label = self.con_plastic[i_con].projection.label.split('_')[1]
            else:
                i_con_total = i_con + int(i_con > 3) + int(i_con > 7) + 1
                y_label = (f"{id_to_symbol(int(i_con_total / self.p.network.num_symbols))}>"
                           f"{id_to_symbol(int(i_con_total % self.p.network.num_symbols))}")
            axs[i_plot].set_ylabel(y_label, weight='bold')
            axs[i_plot].grid(True, which='both', axis='both')

        axs[-1].set_xlabel("Number of learning phases")
        fig.text(0.02, 0.5, "Permanence / connection direction", va="center", rotation="vertical")

        fig.show()

    def plot_weight_diff(self):
        fig, axs = plt.subplots(len(self.log_weights), 1, sharex="all", figsize=(10, 7))

        for i_perm in self.log_weights:
            weights_diff = self.con_plastic[i_perm].weights[-1] - self.con_plastic[i_perm].weights[0]
            num_connections = len(weights_diff)

            colors = ['C0' if p >= 0 else 'C1' for p in weights_diff]

            axs[i_perm].bar(range(num_connections), weights_diff, color=colors)
            axs[i_perm].set_ylabel(self.con_plastic[i_perm].projection.label.split('_')[1], weight='bold')
            axs[i_perm].yaxis.set_ticks([0])
            axs[i_perm].xaxis.set_ticks(range(0, num_connections))
            axs[i_perm].grid(True, which='both', axis='both')

        axs[-1].set_xlabel("Connection [#]")
        fig.text(0.02, 0.5, "Weights diff / connection direction", va="center", rotation="vertical")

    def _retrieve_neuron_data(self):
        self.neuron_events = dict()
        self.max_spike_time = 0

        for neuron_type in NeuronType.get_all_types():
            self.neuron_events[neuron_type] = list()
            for i_symbol in range(self.p.network.num_symbols):
                events = self.get_neuron_data(neuron_type, value_type=RecTypes.SPIKES, symbol_id=i_symbol,
                                              dtype=list)
                self.neuron_events[neuron_type].append(events)
                for neuron_events in events:
                    if len(neuron_events) <= 0:
                        continue
                    max_spike_time = max(neuron_events)
                    if max_spike_time > self.max_spike_time:
                        self.max_spike_time = float(max_spike_time)

    def get_spike_times(self, runtime, dt):
        log.detail("Calculating spike times")

        times = np.linspace(0., runtime, int(runtime / dt))

        spike_times_dendrite = np.zeros((self.p.network.num_symbols, self.p.network.num_neurons, len(times)),
                                        dtype=np.int8)
        spike_times_soma = np.zeros((self.p.network.num_symbols, self.p.network.num_neurons, len(times)), dtype=np.int8)

        for i_symbol in range(self.p.network.num_symbols):
            for i_dendrite, dendrite_spikes in enumerate(self.get_neuron_data(NeuronType.Dendrite, symbol_id=i_symbol,
                                                                              value_type=RecTypes.SPIKES, dtype=list)):
                for spike_time in dendrite_spikes:
                    spike_id = int(spike_time / times[1])
                    spike_times_dendrite[i_symbol, i_dendrite, spike_id] = 1

            for i_soma, soma_spikes in enumerate(self.get_neuron_data(NeuronType.Soma, symbol_id=i_symbol,
                                                                      value_type=RecTypes.SPIKES)):
                for spike_time in soma_spikes:
                    spike_id = int(spike_time / times[1])
                    spike_times_soma[i_symbol, i_soma, spike_id] = 1

        return spike_times_dendrite, spike_times_soma

    def __update_dendritic_trace(self):
        for i_symbol in range(self.p.network.num_symbols):
            for i_neuron in range(self.p.network.num_neurons):
                events = np.array(self.neuron_events[NeuronType.Dendrite][i_symbol][i_neuron])
                self.trace_dendrites[i_symbol, i_neuron] = calculate_trace(self.trace_dendrites[i_symbol, i_neuron],
                                                                           0, self.p.experiment.runtime,
                                                                           events, self.p.plasticity.tau_h)

    def set_weights_exc_exc(self, new_weight, con_id, post_ids=None, p_con=1.0):
        # ToDo: Find out why this is not working in Nest after one simulation, only before all simulations
        weights = self.con_plastic[con_id].projection.get("weight", format="array")

        if post_ids is None:
            post_ids = range(weights.shape[1])

        for i in post_ids:
            pre_ids = np.logical_not(np.isnan(weights[:, i]))
            pre_ids = pre_ids[:int(p_con * len(pre_ids))]
            weights[pre_ids, i] = new_weight

        self.con_plastic[con_id].projection.set(weight=weights)
        self.con_plastic[con_id].w_mature = new_weight

        return self.con_plastic[con_id].projection.get("weight", format="array")

    def run(self, runtime=None, steps=None, plasticity_enabled=True, store_to_cache=False, dyn_exc_inh=False,
            run_type=RunType.SINGLE, num_active_neuron_thresh=None):
        if runtime is None:
            runtime = self.p.experiment.runtime
        if steps is None:
            steps = self.p.experiment.episodes
        self.p.experiment.episodes = 0

        if type(runtime) is str:
            if str(runtime).lower() == 'max':
                runtime = self.last_ext_spike_time + (self.p.encoding.dt_seq - self.p.encoding.t_exc_start)
        elif type(runtime) is float or type(runtime) is int:
            pass
        elif runtime is None:
            log.debug("No runtime specified. Setting runtime to last spike time + 2xdt_stm")
            runtime = self.last_ext_spike_time + (self.p.encoding.dt_seq - self.p.encoding.t_exc_start)
        else:
            log.error("Error! Wrong runtime")

        self.p.experiment.runtime = runtime

        for t in range(steps):
            log.info(
                f'Running {self.network_mode} step {self.experiment_episodes + t + 1}/{self.experiment_episodes + steps}')

            # reset the simulator and the network state if not first run
            if self.run_state:
                self.reset(store_to_cache)

            # set start time to 0.0 because
            # - nest is reset and always starts with 0.0
            # - bss2 resets the time itself after each run to 0.0
            sim_start_time = 0.0
            log.detail(f"Current time: {sim_start_time}")

            self.run_sim(runtime)

            self._retrieve_neuron_data()

            if self.network_mode == NetworkState.REPLAY:
                self.neuron_events_hist.append(self.neuron_events)

            if self.p.performance.compute_performance and self.network_mode == NetworkState.PREDICTIVE:
                self.performance.compute(neuron_events=self.neuron_events,
                                         method=self.p.performance.method)

            # update graph representation
            new_node_activity, new_edge_activity = self.get_activity()
            self.map.update_graph(new_node_activity, new_edge_activity)

            if plasticity_enabled:
                if run_type == RunType.MULTI:
                    log.warn(
                        f"Multi-core version of plasticity calculation is currently not working. Please choose the "
                        f"single-core version. Not calculating plasticity.")
                    # self.__run_plasticity_parallel(runtime, sim_start_time, dyn_exc_inh=dyn_exc_inh)
                elif run_type == RunType.SINGLE:
                    self.__run_plasticity_singular(runtime, sim_start_time, dyn_exc_inh=dyn_exc_inh)

            if self.p.experiment.save_auto and self.p.experiment.save_auto_epoches > 0:
                if (t + 1) % self.p.experiment.save_auto_epoches == 0:
                    self.p.experiment.episodes = self.experiment_episodes + t + 1
                    self.save_full_state()

            if self.p.plasticity.learning_rate_decay is not None:
                self.p.plasticity.learning_factor *= self.p.plasticity.learning_rate_decay

            if self.network_mode == NetworkState.REPLAY:
                self.update_adapt_thresholds(num_active_neuron_thresh=num_active_neuron_thresh)
                neuron_thresholds = [self.neurons_exc[i_sym].get("V_th") for i_sym in range(self.p.network.num_symbols)]
                self.neuron_thresholds_hist.append(neuron_thresholds)

            # print performance results
            self.print_performance_results(final=False)

        self.experiment_episodes += steps
        self.p.experiment.episodes = self.experiment_episodes

        if self.p.experiment.save_final or self.p.experiment.save_auto:
            self.save_full_state()

    def print_performance_results(self, final=False):
        performance_results = self.performance.get_performance_dict(final_result=True,
                                                                    running_avgs=self.p.performance.running_avgs)
        if final:
            log.essens(f"Performance (0.5):  {performance_results['error_running-avg-0.5']}  |  "
                       f"Epochs:  {performance_results['num-epochs']}")
        else:
            log.essens(f"Performance:  {performance_results['error_last']}  |  "
                       f"Performance (0.5):  {performance_results['error_running-avg-0.5']}  |  "
                       f"Epochs:  {performance_results['num-epochs']}")

    def __run_plasticity_singular(self, runtime, sim_start_time, dyn_exc_inh=False):
        log.debug("Starting plasticity calculations")

        active_synapse_post = np.zeros((self.p.network.num_symbols, self.p.network.num_neurons))

        # Calculate plasticity for each synapse
        for i_plasticity, plasticity in enumerate(self.con_plastic):
            plasticity(runtime, sim_start_time=sim_start_time)
            log.debug(f"Finished plasticity calculation {i_plasticity + 1}/{len(self.con_plastic)}")

            if dyn_exc_inh:
                w = self.exc_to_exc[i_plasticity].get("weight", format="array")
                letter_id = SYMBOLS[plasticity.get_post_symbol()]
                active_synapse_post[letter_id, :] = np.logical_or(active_synapse_post[letter_id, :],
                                                                  np.any(w > 0, axis=0))

        if dyn_exc_inh and self.p.synapses.w_exc_inh_dyn is not None:
            for i_inh in range(self.p.network.num_symbols):
                w = self.exc_to_inh.get("weight", format="array")
                w[active_synapse_post[i_inh, :]] = self.p.synapses.w_exc_inh_dyn

        self.__update_dendritic_trace()

    def __run_plasticity_parallel(self, runtime, sim_start_time, dyn_exc_inh=False):
        log.debug("Starting plasticity calculations")

        active_synapse_post = np.zeros((self.p.network.num_symbols, self.p.network.num_neurons))

        q_plasticity = mp.Queue()

        # Calculate plasticity for each synapse
        processes = []
        for i_plasticity, plasticity in enumerate(self.con_plastic):
            log.debug(f'Starting plasticity calculation for {i_plasticity}')
            processes.append(Process(target=plasticity, args=(plasticity, runtime, sim_start_time, q_plasticity)))
            processes[i_plasticity].start()

        num_finished_plasticities = 0
        while num_finished_plasticities < len(self.con_plastic):
            log.debug(f'Waiting for plasticity calculation [{num_finished_plasticities + 1}/{len(self.con_plastic)}]')
            con_plastic = q_plasticity.get()
            self.__update_con_plastic(con_plastic)
            num_finished_plasticities += 1

        for i_plasticity, plasticity in enumerate(self.con_plastic):
            processes[i_plasticity].join()

            log.debug(f"Finished plasticity calculation {i_plasticity + 1}/{len(self.con_plastic)}")

            # Check if an exception occurred in the sub-process, then raise this exception
            if processes[i_plasticity].exception:
                exc, trc = processes[i_plasticity].exception
                print(trc)
                raise exc

            if dyn_exc_inh:
                w = self.exc_to_exc[i_plasticity].get("weight", format="array")
                letter_id = SYMBOLS[plasticity.get_post_symbol()]
                active_synapse_post[letter_id, :] = np.logical_or(active_synapse_post[letter_id, :],
                                                                  np.any(w > 0, axis=0))

        if dyn_exc_inh and self.p.synapses.w_exc_inh_dyn is not None:
            for i_inh in range(self.p.network.num_symbols):
                w = self.exc_to_inh.get("weight", format="array")
                w[active_synapse_post[i_inh, :]] = self.p.synapses.w_exc_inh_dyn

        self.__update_dendritic_trace()

    def __update_con_plastic(self, new_con_plastic):
        for obj_name in NON_PICKLE_OBJECTS:
            setattr(new_con_plastic, obj_name, getattr(self.con_plastic[new_con_plastic.id], obj_name))

        for obj_name, obj_value in new_con_plastic.__dict__.items():
            if not (obj_name.startswith('_') or callable(obj_value)):
                if obj_name == "proj_post_soma_inh":
                    if self.con_plastic[new_con_plastic.id].proj_post_soma_inh is not None:
                        self.con_plastic[new_con_plastic.id].proj_post_soma_inh.set(
                            weight=new_con_plastic.get("weight", format="array"))
                elif obj_name == "projection_weight":
                    if new_con_plastic.projection_weight is not None:
                        self.con_plastic[new_con_plastic.id].projection.set(weight=new_con_plastic.projection_weight)
                elif obj_name not in NON_PICKLE_OBJECTS:
                    setattr(self.con_plastic[new_con_plastic.id], obj_name, getattr(new_con_plastic, obj_name))

    def save_config(self):
        folder_path = get_experiment_folder(self.p.experiment.type, self.p.experiment.id, self.experiment_num,
                                            experiment_map=self.p.experiment.map_name,
                                            experiment_subnum=self.experiment_subnum, instance_id=self.instance_id)
        file_path = get_experiment_file(FileNames.CONFIG[ConfigType.NETWORK], experiment_path=folder_path)

        with open(file_path, 'w') as file:
            yaml.dump(self.p.dict(exclude_none=True), file)

    def save_performance_data(self):
        folder_path = get_experiment_folder(self.p.experiment.type, self.p.experiment.id, self.experiment_num,
                                            experiment_map=self.p.experiment.map_name,
                                            experiment_subnum=self.experiment_subnum, instance_id=self.instance_id)
        file_path = get_experiment_file(FileNames.PERFORMANCE, experiment_path=folder_path)

        np.savez(file_path, **self.performance.data)

    def save_network_data(self):
        # ToDo: Check if this works with bss2
        folder_path = get_experiment_folder(self.p.experiment.type, self.p.experiment.id, self.experiment_num,
                                            experiment_map=self.p.experiment.map_name,
                                            experiment_subnum=self.experiment_subnum, instance_id=self.instance_id)

        # Save weights
        file_path = get_experiment_file(FileNames.WEIGHTS, experiment_path=folder_path)

        weights_dict = {var_name: getattr(self, var_name) for var_name in RuntimeConfig.saved_weights}
        for con_name, connections in weights_dict.items():
            weights_all = list()
            for connection in connections:
                weights_all.append(connection.get("weight", format="array"))
            weights_dict[con_name] = np.array(weights_all)

        np.savez(file_path, **weights_dict)

        # Save events
        file_path = get_experiment_file(FileNames.EVENTS, experiment_path=folder_path)
        with open(file_path, 'wb') as f:
            pickle.dump(self.neuron_events, f)

        # Save network variables
        file_path = get_experiment_file(FileNames.NETWORK, experiment_path=folder_path)

        network_dict = {var_name: getattr(self, var_name) for var_name in RuntimeConfig.saved_network_vars}

        np.savez(file_path, **network_dict)

        # Save plasticity parameters
        file_path = get_experiment_file(FileNames.PLASTICITY, experiment_path=folder_path)

        plasticity_dict = {var_name: list() for var_name in RuntimeConfig.saved_plasticity_vars}
        for con_plastic in self.con_plastic:
            for var_name in plasticity_dict.keys():
                plasticity_dict[var_name].append(getattr(con_plastic, var_name))

        for var_name in plasticity_dict.keys():
            plasticity_dict[var_name] = np.array(plasticity_dict[var_name])

        np.savez(file_path, **plasticity_dict)

    def save_plot_events(self, neuron_types="all", symbols="all", size=None, x_lim_lower=None, x_lim_upper=None,
                         seq_start=0, seq_end=None, fig_title="", run_id=None, show_grid=False,
                         separate_seqs=False, replay_runtime=None, plot_dendritic_trace=True, enable_y_ticks=True,
                         x_tick_step=None, plot_thresholds=False):
        # set plot_name and retrieve experiment folder given the current config
        plot_name = get_experiment_file(FileNames.PLOT_EVENTS)
        exp_folder_path = get_experiment_folder(self.p.experiment.type, self.p.experiment.id, self.experiment_num,
                                                experiment_map=self.p.experiment.map_name,
                                                experiment_subnum=self.experiment_subnum, instance_id=self.instance_id)

        # load all file/folder names that match pattern of plot_name in exp_folder_path
        files = glob.glob(f"{exp_folder_path}/{plot_name}*")

        # find last plot file, i.e. with highest id in the files list
        last_plot_id = 0
        for file_i in files:
            try:
                plot_id_i = int(os.path.basename(file_i).split('_')[-1].split('.')[0])
            except ValueError:
                log.warning(
                    f"Plot id '{os.path.basename(file_i).split('_')[-1].split('.')[0]}' is not an int. Skipping file.")
                continue
            if plot_id_i > last_plot_id:
                last_plot_id = plot_id_i
        last_plot_id += 1

        # generate new plot name and path based on the calculated id
        plot_name = f"{plot_name}_{last_plot_id:02d}"
        plot_path = join(exp_folder_path, plot_name)

        # plot events and save the plot to the given path
        self.plot_events(neuron_types=neuron_types, symbols=symbols, size=size, x_lim_lower=x_lim_lower,
                         x_lim_upper=x_lim_upper, seq_start=seq_start, seq_end=seq_end, fig_title=fig_title,
                         run_id=run_id, show_grid=show_grid, separate_seqs=separate_seqs, replay_runtime=replay_runtime,
                         plot_dendritic_trace=plot_dendritic_trace, enable_y_ticks=enable_y_ticks,
                         x_tick_step=x_tick_step, file_path=plot_path, plot_thresholds=plot_thresholds
                         )

    def save_plot_graph(self, history_range=None, fps=1, only_traversable=True, arrows=True,
                        label_type=LabelTypes.LETTERS, empty_label="", title="Frame", show_plot=False):
        self.map.plot_graph_history(self, history_range=history_range, fps=fps, only_traversable=only_traversable,
                                    arrows=arrows, label_type=label_type, empty_label=empty_label, title=title,
                                    show_plot=show_plot, experiment_num=self.experiment_num, save_plot=True)

    def save_full_state(self, running_avg_perc=0.5, optimized_parameter_ranges=None, save_setup=False):
        log.debug("Saving full state of network and experiment.")

        if (self.p.experiment.type in
                [ExperimentType.EVAL_MULTI, ExperimentType.OPT_GRID, ExperimentType.OPT_GRID_MULTI]):
            if self.instance_id is not None and self.instance_id == 0 and save_setup:
                self.experiment_num = save_experimental_setup(net=self, experiment_num=self.experiment_num,
                                                              experiment_subnum=self.experiment_subnum,
                                                              instance_id=self.instance_id,
                                                              optimized_parameter_ranges=optimized_parameter_ranges,
                                                              experiment_map=self.p.experiment.map_name)
            save_instance_setup(net=self.__str__(), parameters=self.p,
                                performance=self.performance.get_performance_dict(final_result=True,
                                                                                  running_avgs=self.p.performance.running_avgs,
                                                                                  decimals=3),
                                experiment_num=self.experiment_num, experiment_subnum=self.experiment_subnum,
                                instance_id=self.instance_id,
                                optimized_parameters=self.optimized_parameters)
        else:
            self.experiment_num = save_experimental_setup(net=self, experiment_num=self.experiment_num,
                                                          experiment_subnum=self.experiment_subnum,
                                                          instance_id=self.instance_id,
                                                          experiment_map=self.p.experiment.map_name)

        self.save_config()
        self.save_performance_data()
        self.save_network_data()
        self.map.plot_graph_history(self, experiment_num=self.experiment_num, arrows=True, save_plot=True,
                                    show_plot=False)

    def load_network_data(self, experiment_type, experiment_num, experiment_map=None, experiment_subnum=None,
                          instance_id=None):
        sys.modules['shtmbss2'] = neuroseq

        # ToDo: Check if this works with bss2
        folder_path = get_experiment_folder(experiment_type, self.p.experiment.id, experiment_num,
                                            experiment_map=experiment_map, experiment_subnum=experiment_subnum,
                                            instance_id=instance_id)

        # Load weights
        file_path = get_experiment_file(FileNames.WEIGHTS, experiment_path=folder_path)
        data_weights = np.load(file_path)

        # Load events
        file_path = get_experiment_file(FileNames.EVENTS, experiment_path=folder_path)
        with open(file_path, 'rb') as f:
            self.neuron_events = pickle.load(f)

        # Load network variables
        file_path = get_experiment_file(FileNames.NETWORK, experiment_path=folder_path)
        data_network = np.load(file_path)
        for var_name, var_value in data_network.items():
            setattr(self, var_name, var_value)

        # Load plasticity parameters
        file_path = get_experiment_file(FileNames.PLASTICITY, experiment_path=folder_path)
        data_plasticity = np.load(file_path, allow_pickle=True)
        data_plasticity = dict(data_plasticity)
        for var_name in ["permanences", "weights"]:
            if var_name in data_plasticity.keys():
                data_plasticity[var_name] = data_plasticity[var_name].tolist()

        return data_weights, data_plasticity

    @staticmethod
    def load_full_state(network_type, experiment_id, experiment_num, experiment_map=None,
                        experiment_type=ExperimentType.EVAL_SINGLE, experiment_subnum=None,
                        network_mode=NetworkState.PREDICTIVE, instance_id=None, debug=False, custom_params=None):
        log.debug("Loading full state of network and experiment.")

        RuntimeConfig.file_prefix = network_mode

        p = NetworkParameters(network_type=network_type)
        p.load_experiment_params(experiment_type=experiment_type, experiment_id=experiment_id,
                                 experiment_map=experiment_map, experiment_num=experiment_num,
                                 experiment_subnum=experiment_subnum, instance_id=instance_id,
                                 custom_params=custom_params)

        shtm = network_type(p=p)
        shtm.network_mode = network_mode

        shtm.p_plot = PlottingParameters(network_type=network_type)
        shtm.p_plot.load_default_params(network_mode=shtm.network_mode, map_name=shtm.p.experiment.map_name)

        shtm.performance.load_data(shtm, experiment_type, experiment_id, experiment_num, experiment_map=experiment_map,
                                   experiment_subnum=experiment_subnum, instance_id=instance_id)
        data_weights, data_plasticity = shtm.load_network_data(experiment_type, experiment_num,
                                                               experiment_map=experiment_map,
                                                               experiment_subnum=experiment_subnum,
                                                               instance_id=instance_id)

        shtm.init_neurons()
        shtm.init_connections(exc_to_exc=data_weights["exc_to_exc"], exc_to_inh=data_weights["exc_to_inh"], debug=debug)

        for i_con_plastic in range(len(shtm.con_plastic)):
            for var_name, var_value in data_plasticity.items():
                setattr(shtm.con_plastic[i_con_plastic], var_name, var_value[i_con_plastic])

        shtm.init_external_input()

        # this network has been run before, so set run-state to true
        shtm.run_state = True

        # set experiment number to given number
        shtm.experiment_num = experiment_num

        return shtm
