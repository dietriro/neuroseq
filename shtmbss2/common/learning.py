import copy
from abc import ABC, abstractmethod

import numpy as np

from shtmbss2.common.config import SYMBOLS, NeuronType, RuntimeConfig, Backends
from shtmbss2.common import network
from shtmbss2.core.helpers import symbol_from_label, calculate_trace
from shtmbss2.core.logging import log

if RuntimeConfig.backend == Backends.BRAIN_SCALES_2:
    from pynn_brainscales.brainscales2 import Projection
elif RuntimeConfig.backend == Backends.NEST:
    from pyNN.nest.projections import Projection
else:
    raise Exception(f"Backend {RuntimeConfig.backend} not implemented yet. "
                    f"Please choose among [{Backends.BRAIN_SCALES_2}, {Backends.NEST}]")


class Plasticity(ABC):
    def __init__(self, projection: Projection, post_somas, shtm, index, proj_post_soma_inh=None, debug=False,
                 learning_factor=None, permanence_init_min=None, permanence_init_max=None, permanence_max=None,
                 permanence_threshold=None, w_mature=None, y=None, lambda_plus=None, weight_learning=None,
                 weight_learning_scale=None, lambda_minus=None, lambda_h=None, target_rate_h=None, tau_plus=None,
                 tau_h=None, delta_t_min=None, delta_t_max=None, dt=None, correlation_threshold=None,
                 homeostasis_depression_rate=None, type=None, enable_structured_stdp=None, **kwargs):
        # custom objects
        self.projection = projection
        self.proj_post_soma_inh = proj_post_soma_inh
        self.shtm: network.SHTMTotal = shtm
        self.post_somas = post_somas

        # editable/changing variables
        if not enable_structured_stdp:
            self.permanence_min = None
        elif permanence_init_min == permanence_init_max:
            self.permanence_min = np.ones(shape=(len(self.projection),), dtype=float) * permanence_init_min
        else:
            self.permanence_min = np.asarray(np.random.randint(permanence_init_min, permanence_init_max,
                                                               size=(len(self.projection),)), dtype=float)
        self.permanence = copy.copy(self.permanence_min)
        self.permanences = None
        self.weights = None
        self.weight_learning = None
        self.weight_learning_scale = None
        self.x = np.zeros((len(self.projection.pre)))
        self.z = np.zeros((len(self.projection.post)))

        self.type = type
        self.enable_structured_stdp = enable_structured_stdp
        self.debug = debug
        self.id = index
        self.projection_weight = None

        # parameters - loaded from file
        self.permanence_max = permanence_max
        self.w_mature = w_mature
        self.tau_plus = tau_plus
        self.tau_h = tau_h
        self.target_rate_h = target_rate_h
        self.y = y
        self.delta_t_min = delta_t_min
        self.delta_t_max = delta_t_max
        self.dt = dt
        self.permanence_threshold = np.ones((len(self.projection))) * permanence_threshold
        self.correlation_threshold = correlation_threshold
        self.lambda_plus = lambda_plus * learning_factor
        self.lambda_minus = lambda_minus * learning_factor
        self.lambda_h = lambda_h * learning_factor
        self.homeostasis_depression_rate = homeostasis_depression_rate

        self.learning_rules = {"original": self.rule, "bss2": self.rule_bss2}

        self.symbol_id_pre = SYMBOLS[symbol_from_label(self.projection.label, network.ID_PRE)]
        self.symbol_id_post = SYMBOLS[symbol_from_label(self.projection.label, network.ID_POST)]

        self.connections = list()

        self.int_conversion = "int" in self.shtm.p.plasticity.type
        if self.int_conversion:
            self.scale_permanence(min_value=0, max_value=256)

    def scale_permanence(self, min_value, max_value):
        factor = max_value / self.permanence_max
        self.permanence_min = np.asarray(self.permanence_min * factor, dtype=int)
        self.permanence = copy.copy(self.permanence_min)

        self.permanence_max = int(max_value)
        self.permanence_threshold = np.ones((len(self.projection)), dtype=int) * int(max_value / 2)

        # self.target_rate_h *= max_value
        # self.correlation_threshold *= max_value

    def rule(self, permanence, permanence_threshold, x, z, runtime, permanence_min,
             neuron_spikes_pre, neuron_spikes_post_soma, neuron_spikes_post_dendrite,
             delay, sim_start_time=0.0):
        last_spike_pre = 0

        neuron_spikes_pre = np.array(neuron_spikes_pre)
        neuron_spikes_post_soma = np.array(neuron_spikes_post_soma)
        neuron_spikes_post_dendrite = np.array(neuron_spikes_post_dendrite)

        permanence_before = permanence

        # log.debug(f"{self.id}  permanence before: {permanence}")

        # loop through pre-synaptic spikes
        for spike_pre in neuron_spikes_pre:
            # calculate temporary x/z value (pre-synaptic/post-dendritic decay)
            x = x * np.exp(-(spike_pre - last_spike_pre) / self.tau_plus) + 1.0

            # loop through post-synaptic spikes between pre-synaptic spikes
            for spike_post in neuron_spikes_post_soma:
                spike_dt = (spike_post + delay) - spike_pre

                # log.debug(f"{self.id}  spikes: {spike_pre}, {spike_post}, {spike_dt}")

                # check if spike-dif is in boundaries
                if self.delta_t_min < spike_dt < self.delta_t_max:
                    # calculate temporary x value (pre synaptic decay)
                    x_tmp = x * np.exp(-spike_dt / self.tau_plus)
                    z_tmp = calculate_trace(z, sim_start_time, spike_post, neuron_spikes_post_dendrite,
                                            self.tau_h)

                    # hebbian learning
                    permanence = self.__facilitate(permanence, x_tmp)
                    d_facilitate = permanence - permanence_before
                    # log.debug(f"{self.id}  d_permanence facilitate: {d_facilitate}")
                    permanence_before = permanence

                    permanence = self.__homeostasis_control(permanence, z_tmp, permanence_min)
                    # if permanence - permanence_before < 0:
                    #     log.debug(f"{self.id}  d_permanence homeostasis: {permanence - permanence_before}")
                    # if self.debug and permanence - permanence_before < 0:
                    # log.info(f"{self.id}  spikes: {spike_pre}, {spike_post}, {spike_dt}")
                    # log.info(f"{self.id}  d_permanence facilitate: {d_facilitate}")
                    # log.info(f"{self.id}  d_permanence homeostasis: {permanence - permanence_before}")
                    permanence_before = permanence

            permanence = self.__depress(permanence, permanence_min)
            last_spike_pre = spike_pre
            # log.debug(f"{self.id}  permanence depression: {permanence - permanence_before}")

        # log.debug(f"{self.id}  permanence after: {permanence}")

        # update x (kplus) and z
        x = x * np.exp(-(runtime - last_spike_pre) / self.tau_plus)

        if permanence >= permanence_threshold:
            mature = True
        else:
            mature = False

        return permanence, x, mature

    def __facilitate(self, permanence, x):
        mu = 0
        clip = np.power(1.0 - (permanence / self.permanence_max), mu)
        permanence_norm = (permanence / self.permanence_max) + (self.lambda_plus * x * clip)
        return min(permanence_norm * self.permanence_max, self.permanence_max)

    def __homeostasis_control(self, permanence, z, permanence_min):
        permanence = permanence + self.lambda_h * (self.target_rate_h - z) * self.permanence_max
        return max(min(permanence, self.permanence_max), permanence_min)

    def __depress(self, permanence, permanence_min):
        permanence = permanence - self.lambda_minus * self.permanence_max
        return max(permanence_min, permanence)

    def __to_int(self, value, precision=256, amplitude=1.5):
        return int(value * precision)

    def rule_bss2(self, permanence, permanence_threshold, x, z, runtime, permanence_min,
                  neuron_spikes_pre, neuron_spikes_post_soma, neuron_spikes_post_dendrite,
                  delay, sim_start_time=0.0):
        neuron_spikes_pre = np.array(neuron_spikes_pre)
        neuron_spikes_post_soma = np.array(neuron_spikes_post_soma)
        neuron_spikes_post_dendrite = np.array(neuron_spikes_post_dendrite)

        permanence_before = permanence

        # log.debug(f"{self.id}  permanence before: {permanence}")

        x = 0
        z_tmp = 0

        # Calculate accumulated x
        spike_pairs_soma_soma = 0
        for spike_pre in neuron_spikes_pre:
            for spike_post in neuron_spikes_post_soma:
                spike_dt = spike_post - spike_pre

                # ToDo: Update rule based on actual trace calculation from BSS-2
                if spike_dt >= 0:
                    spike_pairs_soma_soma += 1
                    # log.debug(f"{self.id}  spikes (ss): {spike_pre}, {spike_post}, {spike_dt}")
                    # if self.int_conversion:
                    #     x += self.__to_int(np.exp(-spike_dt / self.tau_plus))
                    # else:
                    x += np.exp(-spike_dt / self.tau_plus)

        # Calculate accumulated z
        spike_pairs_dend_soma = 0
        for spike_post_dendrite in neuron_spikes_post_dendrite:
            for spike_post in neuron_spikes_post_soma:
                spike_dt = spike_post - spike_post_dendrite

                # ToDo: Update rule based on actual trace calculation from BSS-2
                if spike_dt >= 0:
                    spike_pairs_dend_soma += 1
                    # log.debug(f"{self.id}  spikes (ds): {spike_post_dendrite}, {spike_post}, {spike_dt}")
                    # if self.int_conversion:
                    #     z_tmp += self.__to_int(np.exp(-spike_dt / self.tau_plus))
                    # else:
                    z_tmp += np.exp(-spike_dt / self.tau_plus)

        # if z_tmp > 0:
        #     log.debug(f"{self.id}  x: {x},  z: {z_tmp}")

        # Calculation of z based on x
        # z = np.exp(-(-self.tau_plus*z_mean)/self.tau_h) * spike_pairs_dend_soma
        # Calculation of z using only number of pre-post spike pairs
        # z = spike_pairs_dend_soma
        # Calculation of z according to current BSS-2 implementation
        z = z_tmp

        # trace_threshold = np.exp(-self.delta_t_max / self.tau_plus)

        # log.debug(f"{self.id} permanence_threshold: {trace_threshold}")
        # log.debug(f"{self.id} x: {x},   x_mean: {x_mean}")
        # log.debug(f"{self.id} z: {z},   z_mean: {z_mean}")

        # hebbian learning
        # Only run facilitate/homeostasis if a spike pair exists with a delta within boundaries,
        # i.e. x or z > 0
        if x > self.correlation_threshold:
            permanence = self.__facilitate_bss2(permanence, x)
        log.debug(f"{self.id}  d_permanence facilitate: {permanence - permanence_before}")
        permanence_before = permanence

        # if self.int_conversion:
        #     permanence = int(np.round(permanence))

        if x > self.correlation_threshold:
            permanence = self.__homeostasis_control_bss2(permanence, z, permanence_min)
        # if permanence - permanence_before < 0:
        # if z > 0:
        log.debug(f"{self.id}  d_permanence homeostasis: {permanence - permanence_before}")
        permanence_before = permanence

        # if self.int_conversion:
        #     permanence = int(np.round(permanence))

        permanence = self.__depress_bss2(permanence, permanence_min, num_spikes=len(neuron_spikes_pre))
        log.debug(f"{self.id}  permanence depression: {permanence - permanence_before}")

        if self.int_conversion:
            permanence = int(np.round(permanence))

        # log.debug(f"{self.id}  permanence after: {permanence}")

        return permanence, x, permanence >= permanence_threshold

    def __facilitate_bss2(self, permanence, x):
        d_permanence = self.lambda_plus * x
        # if not self.int_conversion:
        d_permanence *= self.permanence_max
        permanence += d_permanence
        return min(permanence, self.permanence_max)

    def __homeostasis_control_bss2(self, permanence, z, permanence_min):
        z_prime = self.target_rate_h - z
        # if self.int_conversion:
        #     z_prime /= 256
        if z_prime < 0:
            z_prime = -np.exp(-z_prime / self.homeostasis_depression_rate)

        permanence = permanence + self.lambda_h * z_prime * self.permanence_max
        return max(min(permanence, self.permanence_max), permanence_min)

    def __depress_bss2(self, permanence, permanence_min, num_spikes):
        permanence = permanence - self.lambda_minus * self.permanence_max * num_spikes
        return max(permanence_min, permanence)

    def enable_permanence_logging(self):
        self.permanences = [np.copy(self.permanence)]

    def enable_weights_logging(self):
        self.weights = [np.copy(self.projection.get("weight", format="array").flatten())]

    def get_pre_symbol(self):
        return symbol_from_label(self.projection.label, network.ID_PRE)

    def get_post_symbol(self):
        return symbol_from_label(self.projection.label, network.ID_POST)

    def get_connection_ids(self, connection_id):
        connection_ids = (f"{self.get_connection_id_pre(self.get_connections()[connection_id])}>"
                          f"{self.get_connection_id_post(self.get_connections()[connection_id])}")
        return connection_ids

    @abstractmethod
    def get_connection_id_pre(self, connection):
        pass

    @abstractmethod
    def get_connection_id_post(self, connection):
        pass

    def get_all_connection_ids(self):
        connection_ids = []
        for con in self.get_connections():
            connection_ids.append(f"{self.get_connection_id_pre(con)}>{self.get_connection_id_post(con)}")
        return connection_ids

    @abstractmethod
    def get_connections(self):
        pass

    def init_connections(self):
        for c, connection in enumerate(self.get_connections()):
            i = self.get_connection_id_post(connection)
            j = self.get_connection_id_pre(connection)
            self.connections.append([c, j, i])

    def __call__(self, runtime: float, sim_start_time=0.0, q_plasticity=None):
        if not self.enable_structured_stdp and self.permanence is None:
            self.permanence_min = np.array(self.projection.get("weight", format="list"))[:, 2] * 100
            self.permanence = copy.copy(self.permanence_min)
            if self.permanences is not None:
                self.enable_permanence_logging()

        if self.connections is None or len(self.connections) <= 0:
            self.init_connections()

        spikes_pre = self.shtm.neuron_events[NeuronType.Soma][self.symbol_id_pre]
        spikes_post_dendrite = self.shtm.neuron_events[NeuronType.Dendrite][self.symbol_id_post]
        spikes_post_soma = self.shtm.neuron_events[NeuronType.Soma][self.symbol_id_post]

        weight = self.projection.get("weight", format="array")
        weight_before = np.copy(weight)

        for c, j, i in self.connections:
            neuron_spikes_pre = spikes_pre[j]
            neuron_spikes_post_dendrite = spikes_post_dendrite[i]
            neuron_spikes_post_soma = spikes_post_soma[i]
            z = self.shtm.trace_dendrites[self.symbol_id_post, i]

            # if self.debug:
            #     log.debug(f"Permanence calculation for connection {c} [{i}, {j}]")
            #     log.debug(f"Spikes pre [soma]: {neuron_spikes_pre}")
            #     log.debug(f"Spikes post [dend]: {neuron_spikes_post_dendrite}")
            #     log.debug(f"Spikes post [soma]: {neuron_spikes_post_soma}")

            permanence, x, mature = (self.learning_rules[self.type.split("_")[0]]
                                     (permanence=self.permanence[c],
                                      permanence_threshold=self.permanence_threshold[c],
                                      runtime=runtime, x=self.x[j], z=z,
                                      permanence_min=self.permanence_min[c],
                                      neuron_spikes_pre=neuron_spikes_pre,
                                      neuron_spikes_post_soma=neuron_spikes_post_soma,
                                      neuron_spikes_post_dendrite=neuron_spikes_post_dendrite,
                                      delay=self.shtm.p.synapses.delay_exc_exc,
                                      sim_start_time=sim_start_time))

            self.permanence[c] = permanence
            self.x[j] = x

            if mature or not self.enable_structured_stdp:
                if not self.enable_structured_stdp:
                    weight[j, i] = permanence * 0.01
                else:
                    weight_offset = (
                                            permanence - self.permanence_threshold) * self.weight_learning_scale if self.weight_learning else 0
                    weight[j, i] = self.w_mature + weight_offset
                if self.proj_post_soma_inh is not None:
                    weight_inh = self.proj_post_soma_inh.get("weight", format="array")
                    weight_inh[i, :] = 250
                    # log.debug(f"+ | W_inh[{i}] = {weight_inh.flatten()}")
                    self.proj_post_soma_inh.set(weight=weight_inh)
            else:
                weight[j, i] = 0
                if self.proj_post_soma_inh is not None:
                    weight_inh = self.proj_post_soma_inh.get("weight", format="array")
                    weight_inh_old = np.copy(weight_inh)
                    weight_inh[i, :] = 0
                    # if np.sum(weight_inh_old.flatten() - weight_inh.flatten()) == 0:
                    #     log.debug(f"- | W_inh[{i}] = {weight_inh.flatten()}")
                    self.proj_post_soma_inh.set(weight=weight_inh)

        weight_diff = weight - weight_before
        if np.logical_and(weight_diff != 0, ~np.isnan(weight_diff)).any():
            self.projection.set(weight=weight)

        if self.permanences is not None:
            self.permanences.append(np.copy(np.round(self.permanence, 6)))
        if self.weights is not None:
            self.weights.append(
                np.copy(np.round(self.projection.get("weight", format="array").flatten(), 6)))

        log.debug(f'Finished execution of plasticity for {self.id}')
