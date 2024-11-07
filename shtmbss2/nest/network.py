import os
import numpy as np

os.environ["PYNEST_QUIET"] = "1"

from abc import ABC
from quantities import ms
from neo.core.spiketrainlist import SpikeTrain, SpikeTrainList

from shtmbss2.nest.config import *
from shtmbss2.core.logging import log
import shtmbss2.common.network as network
from shtmbss2.core.helpers import id_to_symbol
from shtmbss2.common.config import NeuronType, RecTypes

import nest
import pyNN.nest as pynn

pynn.setup(timestep=0.1, t_flush=500, spike_precision="on_grid")

RECORDING_VALUES = {
    NeuronType.Soma: {RecTypes.SPIKES: "spikes", RecTypes.V: "V_m"},
    NeuronType.Dendrite: {RecTypes.SPIKES: "dAP", RecTypes.V: "I_dend"},
    NeuronType.Inhibitory: {RecTypes.SPIKES: "spikes", RecTypes.V: "v"},
    NeuronType.InhibitoryGlobal: {RecTypes.SPIKES: "spikes", RecTypes.V: "v"}
}
MCNeuron = pynn.NativeCellType


class SHTMBase(network.SHTMBase, ABC):

    def __init__(self, experiment_type=ExperimentType.EVAL_SINGLE, experiment_id=None, experiment_num=None,
                 experiment_subnum=None, instance_id=None, seed_offset=None, p=None, **kwargs):
        global MCNeuron

        super().__init__(experiment_type=experiment_type, experiment_id=experiment_id, experiment_num=experiment_num,
                         experiment_subnum=experiment_subnum, instance_id=instance_id, seed_offset=seed_offset, p=p,
                         **kwargs)

        if not RuntimeConfig.backend_initialization:
            nest.Install(self.p.backend.module_name)
            RuntimeConfig.backend_initialization = True
        MCNeuron = pynn.native_cell_type(self.p.backend.neuron_name)

    def init_all_neurons_exc(self, num_neurons=None):
        neurons_exc = list()
        for i in range(self.p.network.num_symbols):
            # create all neurons for symbol i
            neurons_symbol = self.init_neurons_exc(num_neurons=num_neurons, symbol_id=i)

            # record voltage and spikes from all dendrites/somas
            neurons_symbol.record([RECORDING_VALUES[NeuronType.Dendrite][RecTypes.SPIKES],
                                   RECORDING_VALUES[NeuronType.Dendrite][RecTypes.V],
                                   RECORDING_VALUES[NeuronType.Soma][RecTypes.SPIKES],
                                   RECORDING_VALUES[NeuronType.Soma][RecTypes.V]])

            # set units for custom cell
            # ToDo: fix this in the nestml model
            neurons_symbol.celltype.units["I_dend"] = "pA"
            neurons_symbol.celltype.units["I_p"] = "pA"
            neurons_symbol.celltype.units["dAP"] = "ms"

            neurons_exc.append(neurons_symbol)

        return neurons_exc

    def init_neurons_exc(self, num_neurons=None, symbol_id=None):
        if num_neurons is None:
            num_neurons = self.p.network.num_neurons

        # ToDo: Replace MCNeuron parameters with actual parameters from file
        all_neurons = pynn.Population(num_neurons, MCNeuron(
            theta_dAP=self.p.neurons.dendrite.theta_dAP,
            I_p=self.p.neurons.dendrite.I_p,
            tau_dAP=self.p.neurons.dendrite.tau_dAP,
            C_m=self.p.neurons.excitatory.c_m,
            E_L=self.p.neurons.excitatory.v_rest,
            V_reset=self.p.neurons.excitatory.v_reset,
            V_th=self.p.neurons.excitatory.v_thresh,
            tau_m=self.p.neurons.excitatory.tau_m,
            tau_syn1=self.p.neurons.excitatory.tau_syn_ext,
            tau_syn2=self.p.neurons.excitatory.tau_syn_den,
            tau_syn3=self.p.neurons.excitatory.tau_syn_inh,
            t_ref=self.p.neurons.excitatory.tau_refrac
        ), initial_values={
            "V_m": self.p.neurons.excitatory.v_rest,
            # "I_dend": 0
        }, label=f"exc_{id_to_symbol(symbol_id)}")

        return all_neurons

    def init_neurons_inh(self, num_neurons=None, tau_refrac=None):
        if num_neurons is None:
            num_neurons = self.p.network.num_symbols

        if tau_refrac is None:
            tau_refrac = self.p.neurons.inhibitory.tau_refrac

        # cm, i_offset, tau_m, tau_refrac, tau_syn_E, tau_syn_I, v_reset, v_rest, v_thresh

        # ToDo: Replace LIF parameters with actual parameters from file
        pop = pynn.Population(num_neurons, pynn.IF_curr_exp(
            cm=self.p.neurons.inhibitory.c_m,
            v_rest=self.p.neurons.inhibitory.v_rest,
            v_reset=self.p.neurons.inhibitory.v_reset,
            v_thresh=self.p.neurons.inhibitory.v_thresh,
            tau_m=self.p.neurons.inhibitory.tau_m,
            tau_syn_I=self.p.neurons.inhibitory.tau_syn_I,
            tau_syn_E=self.p.neurons.inhibitory.tau_syn_E,
            tau_refrac=tau_refrac * ms,
        ), initial_values={
            "v": self.p.neurons.inhibitory.v_rest
        })

        pop.record([RECORDING_VALUES[NeuronType.Inhibitory][RecTypes.SPIKES],
                    RECORDING_VALUES[NeuronType.Inhibitory][RecTypes.V]])

        return pop

    def init_external_input(self, init_recorder=False, init_performance=False):
        network.SHTMBase.init_external_input(self, init_recorder=init_recorder, init_performance=init_performance)

        if init_recorder:
            self.neurons_ext.record(["spikes"])


    def reset(self, store_to_cache=False):
        # ToDo: Have a look if we can keep pynn from running 'store_to_cache' - this takes about a second for 5 epochs
        pynn.reset(store_to_cache=store_to_cache)
        # re-initialize external input, but not the recorders (doesn't work with nest)
        self.init_external_input(init_recorder=False, init_performance=False)

        self.run_state = False

    def get_neurons(self, neuron_type, symbol_id=None):
        neurons = None
        if neuron_type == NeuronType.Inhibitory:
            neurons = self.neurons_inh
        elif neuron_type == NeuronType.InhibitoryGlobal:
            neurons = self.neurons_inh_global
        elif neuron_type in [NeuronType.Dendrite, NeuronType.Soma]:
            neurons = self.neurons_exc

        if symbol_id is None:
            return neurons
        else:
            if neuron_type == NeuronType.Inhibitory or neuron_type == NeuronType.InhibitoryGlobal:
                return pynn.PopulationView(neurons, [symbol_id])
            else:
                return neurons[symbol_id]

    def get_neuron_data(self, neuron_type, neurons=None, value_type="spikes", symbol_id=None, neuron_id=None,
                        runtime=None, dtype=None):
        if neurons is None:
            neurons = self.get_neurons(neuron_type,
                                       symbol_id=symbol_id if neuron_type is not NeuronType.InhibitoryGlobal else None)

        if value_type == RecTypes.SPIKES:
            if neuron_type == NeuronType.Dendrite:
                spikes_binary = neurons.get_data(
                    RECORDING_VALUES[neuron_type][value_type]).segments[-1].analogsignals[0]
                spike_ids = np.asarray(np.argwhere(np.array(spikes_binary) > 0), dtype=float)

                if dtype is np.ndarray:
                    data = spike_ids[:, [1, 0]]
                    data[:, 1] *= neurons.recorder.sampling_interval
                else:
                    spike_list = spike_ids.tolist()
                    spikes = [[] for _ in range(self.p.network.num_neurons)]
                    for spike_time, spike_id in spike_list:
                        spikes[int(spike_id)].append(spike_time)
                    for i_spikes in range(len(spikes)):
                        spikes[i_spikes] = ((np.array(spikes[i_spikes], dtype=float) + 1) *
                                            neurons.recorder.sampling_interval)

                    if dtype is list:
                        data = spikes
                    else:
                        spike_trains = [SpikeTrain(spikes_i * ms, t_start=neurons.get_data().segments[-1].t_start,
                                                   t_stop=neurons.get_data().segments[-1].t_stop)
                                        for spikes_i in spikes]
                        data = SpikeTrainList(spike_trains)
            else:
                data = neurons.get_data(RECORDING_VALUES[neuron_type][value_type]).segments[-1].spiketrains
                if dtype is np.ndarray:
                    spike_times = data.multiplexed
                    if len(spike_times[0]) > 0:
                        data = np.array(spike_times).transpose()
                        data[:, 0] = neurons.id_to_index(data[:, 0])
                    else:
                        data = np.empty((0, 2))
                elif dtype is list:
                    data = [s.base for s in data]
        elif value_type == RecTypes.V:
            # Return the analogsignal of the last segment (only one analogsignal in the list because we specified
            # that we want to have the voltage in get_data())
            data = neurons.get_data(RECORDING_VALUES[neuron_type][value_type]).segments[-1].analogsignals[0]
            if neuron_id is not None and neuron_type in [NeuronType.Dendrite, NeuronType.Soma]:
                if neuron_id >= data.shape[1]:
                    log.warning(f"Neuron ID {neuron_id} out of bounds for data with shape {data.shape}. "
                                f"Returning full data range.")
                else:
                    data = data[:, neuron_id]
        else:
            log.error(f"Error retrieving neuron data! Unknown value_type: '{value_type}'.")
            return None
        return data

    # def plot_events(self, neuron_types="all", symbols="all", size=None, x_lim_lower=None,
    #                 x_lim_upper=None, seq_start=0,
    #                 seq_end=None, fig_title="", file_path=None, window="initial"):
    #     if size is None:
    #         size = (12, 10)
    #
    #     if type(neuron_types) is str and neuron_types == "all":
    #         neuron_types = [NeuronType.Dendrite, NeuronType.Soma, NeuronType.Inhibitory]
    #     elif type(neuron_types) is list:
    #         pass
    #     else:
    #         return
    #
    #     if window == "initial":
    #         max_time = self.p.experiment.runtime
    #     else:
    #         max_time = get_current_time()
    #
    #     if x_lim_lower is None:
    #         if window == "initial":
    #             x_lim_lower = 0.
    #         else:
    #             x_lim_lower = get_current_time() - self.p.experiment.runtime
    #     if x_lim_upper is None:
    #         x_lim_upper = max_time
    #
    #     if type(symbols) is str and symbols == "all":
    #         symbols = range(self.p.network.num_symbols)
    #     elif type(symbols) is list:
    #         pass
    #
    #     if len(symbols) == 1:
    #         fig, axs = plt.subplots(figsize=size)
    #     else:
    #         fig, axs = plt.subplots(self.p.network.num_symbols, 1, sharex="all", figsize=size)
    #
    #     if seq_end is None:
    #         seq_end = seq_start + self.p.experiment.runtime
    #
    #     ax = None
    #
    #     for i_symbol in symbols:
    #         if len(symbols) == 1:
    #             ax = axs
    #         else:
    #             ax = axs[i_symbol]
    #
    #         # neurons_all = dict()
    #         # neurons_all[NeuronType.Dendrite], neurons_all[NeuronType.Soma], = self.neurons_exc[i_symbol]
    #         # neurons_all[NeuronType.Inhibitory] = pynn.PopulationView(self.neurons_inh, [i_symbol])
    #
    #         for neurons_i in neuron_types:
    #             neurons = PopulationView(self.neurons_inh, [i_symbol]) if neurons_i ==
    #             NeuronType.Inhibitory else self.neurons_exc[i_symbol]
    #             # Retrieve and plot spikes from selected neurons
    #             spikes = [s.base for s in self.get_neuron_data(neuron_type=neurons_i,
    #                                                            neurons=neurons,
    #                                                            value_type=RecTypes.SPIKES)]
    #             if neurons_i == NeuronType.Inhibitory:
    #                 spikes.append([])
    #             else:
    #                 spikes.insert(0, [])
    #             if neurons_i == NeuronType.Dendrite:
    #                 spikes_post = [s.base for s in self.get_neuron_data(neuron_type=NeuronType.Soma,
    #                                                                     neurons=neurons,
    #                                                                     value_type=RecTypes.SPIKES)]
    #                 plot_dendritic_events(ax, spikes[1:], spikes_post, tau_dap=self.p.neurons.dendrite.tau_dAP,
    #                                       color=f"C{neurons_i.ID}", label=neurons_i.NAME.capitalize(),
    #                                       seq_start=seq_start, seq_end=seq_end)
    #             else:
    #                 line_widths = 1.5
    #                 line_lengths = 1
    #
    #                 ax.eventplot(spikes, linewidths=line_widths, linelengths=line_lengths,
    #                              label=neurons_i.NAME.capitalize(), color=f"C{neurons_i.ID}")
    #
    #         # Configure the plot layout
    #         ax.set_xlim(x_lim_lower, x_lim_upper)
    #         ax.set_ylim(-1, self.p.network.num_neurons + 1)
    #         ax.yaxis.set_ticks(range(self.p.network.num_neurons + 2))
    #         ax.set_ylabel(self.id_to_letter(i_symbol), weight='bold', fontsize=20)
    #         # ax.grid(True, which='both', axis='both')
    #
    #         # Generate y-tick-labels based on number of neurons per symbol
    #         y_tick_labels = ['Inh', '', '0'] + ['' for k in range(self.p.network.num_neurons - 2)] + [
    #             str(self.p.network.num_neurons - 1)]
    #         ax.set_yticklabels(y_tick_labels, rotation=45, fontsize=18)
    #
    #     # Create custom legend for all plots
    #     custom_lines = [Line2D([0], [0], color=f"C{n.ID}", label=n.NAME.capitalize(), lw=3) for n in neuron_types]
    #
    #     ax.set_xlabel("Time [ms]", fontsize=26, labelpad=14)
    #     ax.xaxis.set_ticks(np.arange(x_lim_lower, x_lim_upper, self.p.encoding.dt_stm/2))
    #     ax.tick_params(axis='x', labelsize=18)
    #
    #     plt.figlegend(handles=custom_lines, loc=(0.377, 0.885), ncol=3, labelspacing=0., fontsize=18, fancybox=True,
    #                   borderaxespad=4)
    #
    #     fig.text(0.01, 0.5, "Symbol & Neuron ID", va="center", rotation="vertical", fontsize=26)
    #
    #     fig.suptitle(fig_title, x=0.5, y=0.99, fontsize=26)
    #     fig.show()
    #
    #     if file_path is not None:
    #         plt.savefig(f"{file_path}.pdf")
    #
    #         pickle.dump(fig, open(f'{file_path}.fig.pickle',
    #                               'wb'))  # This is for Python 3 - py2 may need `file` instead of `open`
    #
    # def plot_v_exc(self, alphabet_range, neuron_range='all', size=None, neuron_type=NeuronType.Soma, runtime=0.1,
    #                show_legend=False, file_path=None):
    #     self.reset_rec_exc()
    #
    #     if size is None:
    #         size = (12, 10)
    #
    #     if type(neuron_range) is str and neuron_range == 'all':
    #         neuron_range = range(self.p.network.num_neurons)
    #     elif type(neuron_range) is list:
    #         pass
    #     else:
    #         return
    #
    #     spike_times = [[]]
    #     header_spikes = list()
    #
    #     fig, ax = plt.subplots(figsize=size)
    #
    #     for alphabet_id in alphabet_range:
    #         for neuron_id in neuron_range:
    #             self.init_rec_exc(alphabet_id=alphabet_id, neuron_id=neuron_id, neuron_type=neuron_type)
    #             run(runtime)
    #
    #             # Retrieve and save spike times
    #             # spikes = self.rec_neurons_exc.get_data("spikes").segments[-1].spiketrains
    #             spikes = self.get_neuron_data(neurons=self.rec_neurons_exc, value_type=RecTypes.SPIKES)
    #             spike_times[0].append(np.array(spikes.multiplexed[1]).round(5).tolist())
    #             header_spikes.append(f"{self.id_to_letter(alphabet_id)}[{neuron_id}]")
    #
    #             # plot_membrane(self.rec_neurons_exc, label=header_spikes[-1])
    #
    #             # membrane = self.rec_neurons_exc.get_data("v").segments[-1].irregularlysampledsignals[0]
    #             membrane = self.get_neuron_data(neurons=self.rec_neurons_exc, value_type=RecTypes.V)
    #             ax.plot(membrane.times, membrane, alpha=0.5, label=header_spikes[-1])
    #
    #             self.reset_rec_exc()
    #
    #     # ax.xaxis.set_ticks(np.arange(0.02, 0.06, 0.01))
    #     ax.tick_params(axis='x', labelsize=18)
    #     ax.tick_params(axis='y', labelsize=18)
    #
    #     ax.set_xlabel("Time [ms]", labelpad=14, fontsize=26)
    #     ax.set_ylabel("Membrane Voltage [a.u.]", labelpad=14, fontsize=26)
    #
    #     if show_legend:
    #         plt.legend()
    #
    #         # Print spike times
    #     print(tabulate(spike_times, headers=header_spikes) + '\n')
    #
    #     if file_path is not None:
    #         plt.savefig(f"{file_path}.pdf")
    #
    #         pickle.dump(fig, open(f'{file_path}.fig.pickle',
    #                               'wb'))  # This is for Python 3 - py2 may need `file` instead of `open`


class SHTMTotal(SHTMBase, network.SHTMTotal):
    def __init__(self, experiment_type=ExperimentType.EVAL_SINGLE, experiment_id=None, experiment_num=None,
                 experiment_subnum=None, instance_id=None, seed_offset=None, p=None, **kwargs):
        super().__init__(experiment_type=experiment_type, experiment_id=experiment_id, experiment_num=experiment_num,
                         experiment_subnum=experiment_subnum, plasticity_cls=Plasticity, instance_id=instance_id,
                         seed_offset=seed_offset, p=p, **kwargs)


class Plasticity(network.Plasticity):
    def __init__(self, projection: pynn.Projection, post_somas, shtm, index, **kwargs):
        super().__init__(projection, post_somas, shtm, index, **kwargs)

    def get_connection_id_pre(self, connection):
        return self.projection.pre.id_to_index(connection.source)

    def get_connection_id_post(self, connection):
        return self.projection.post.id_to_index(connection.target)

    def get_connections(self):
        return self.projection.nest_connections
