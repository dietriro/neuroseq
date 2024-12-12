import textwrap
import numpy as np
import pynn_brainscales.brainscales2 as pynn
import pygrenade_vx as grenade


class PlasticityOnChip(pynn.PlasticityRule):
    def __init__(self, timer: pynn.Timer, num_neurons: int, permanence_threshold: int, w_mature: int, target_rate_h,
                 lambda_plus, lambda_minus, lambda_h, learning_factor, delta_t_max, tau_plus, num_runs, random_seed,
                 p_exc_exc=0.2, correlation_threshold=0):
        # observables recorded for each invocation of rule during experiment
        # [weights, permanences, one correlation]
        obsv_data = pynn.PlasticityRule.ObservablePerSynapse()
        obsv_data.type = pynn.PlasticityRule.ObservablePerSynapse.Type.int8
        obsv_data.layout_per_row = pynn.PlasticityRule.ObservablePerSynapse \
            .LayoutPerRow.packed_active_columns

        pynn.PlasticityRule.__init__(self, timer, observables={
            "data": obsv_data,
            "correlation": obsv_data,
        })
        self.num_neurons = num_neurons
        self.permanence_threshold = permanence_threshold
        self.w_mature = w_mature
        self.target_rate_h = target_rate_h
        self.lambda_plus = lambda_plus * learning_factor
        self.lambda_minus = lambda_minus * learning_factor
        self.lambda_h = lambda_h * learning_factor
        self.p_exc_exc = p_exc_exc
        self.threshold = np.exp(-delta_t_max / tau_plus)
        self.correlation_threshold = correlation_threshold
        self.num_runs = num_runs
        self.random_seed = random_seed

    def get_placement(self):
        if self._simulator.state.grenade_network_graph is None or not \
        self._simulator.state.grenade_network.execution_instances[grenade.common.ExecutionInstanceID()].projections:
            return "TODO"
        synapse_rows = []
        for projection in self._projections:
            placed_connections = projection.placed_connections
            synapse_rows.append([])
            for placed_connection in placed_connections:
                assert len(placed_connection) == 1
                if (not synapse_rows[-1]) or (synapse_rows[-1][-1] != int(placed_connection[0].synapse_row)):
                    synapse_rows[-1].append(int(placed_connection[0].synapse_row))
        ret = []
        for projection in synapse_rows:
            assert len(projection) == 60
            ret.append("std::array<size_t, 60>{" + ", ".join([str(i) for i in projection]) + "}")
        return ",\n".join(ret)

    def generate_kernel(self) -> str:
        """
        Generate plasticity rule kernel to be compiled into PPU program.

        :return: PPU-code of plasticity-rule kernel as string.
        """
        return textwrap.dedent(f"""
        #include "grenade/vx/ppu/neuron_view_handle.h"
        #include "grenade/vx/ppu/synapse_array_view_handle.h"
        #include "grenade/vx/ppu/translate_columns.h"
        #include "libnux/vx/location.h"
        #include "libnux/vx/vector_row.h"
        #include "libnux/vx/vector_if.h"
        #include "libnux/system.h"
        #include "libnux/vx/helper.h"
        #include "libnux/vx/random.h"

        using namespace grenade::vx::ppu;
        using namespace libnux::vx;

        extern volatile PPUOnDLS ppu;
        extern volatile uint64_t time_origin;

        // TODO: remove once fixed in grenade
        // We need to read something from intmem because extmem expects a std::map entry to exist
        // which it should create.
        volatile uint32_t dummy;

        // fractional saturating arithmetic between [-128, 127] / 128
        int8_t permanences[{self.num_neurons}][256] __attribute__((section("ext.data")));

        /**
         * Baseline reads from correlation sensors, one per column.
         */
        VectorRowMod8 get_baselines()
        {{
            reset_all_correlations();
            VectorRowMod16 accumulator(0);
            for (size_t row = 0; row < 256; ++row) {{
                    VectorRowMod8 result;
                    get_causal_correlation(&result.even.data, &result.odd.data, row);
                    clobber();
                    accumulator += static_cast<VectorRowMod16>(result);
            }}
            return VectorRowMod8(accumulator >> 8);
        }}
        VectorRowFracSat8 correlation_baseline __attribute__((section("ext.data"))) = get_baselines().convert_contiguous();

        std::array<std::array<size_t, 60>, 3> synapse_rows = {{
        {self.get_placement()}
        }};

        void PLASTICITY_RULE_KERNEL(
            std::array<SynapseArrayViewHandle, 3>& synapses,
            std::array<NeuronViewHandle, 2>& neurons,
            Recording& recording)
        {{
            auto const& synapses_soma_to_dendrite = synapses[0];
            auto const& synapses_soma_to_soma = synapses[1];
            auto const& synapses_dendrite_to_soma = synapses[2];

            auto& somas = neurons[1];

            // both projections need to be placed on the same hemisphere
            if (synapses_soma_to_dendrite.hemisphere != synapses_soma_to_soma.hemisphere || synapses_soma_to_dendrite.hemisphere != synapses_dendrite_to_soma.hemisphere) {{
                exit(1);
            }}

            // check that we are executing on the correct hemisphere
            if (synapses_soma_to_dendrite.hemisphere != ppu) {{
                return;
            }}

            uint32_t seed = {self.random_seed};

            VectorRowFracSat8 causal_correlation_dendrite_to_soma;
            size_t synapse_column_dendrite_to_soma_index = 0;
            for (size_t used_synapse_row_index = 0; used_synapse_row_index < synapse_rows[0].size(); ++used_synapse_row_index) {{
                size_t const synapse_row_dendrite_to_soma_index = synapse_rows[2][used_synapse_row_index];

                while(!synapses_dendrite_to_soma.columns.test(synapse_column_dendrite_to_soma_index)) {{
                    synapse_column_dendrite_to_soma_index++;
                }}

                // get causal correlations and reset accumulated signals
                // in [-128, 127] integer
                VectorRowMod8 causal_correlation_dendrite_to_soma_raw;
                get_causal_correlation(&causal_correlation_dendrite_to_soma_raw.even.data, &causal_correlation_dendrite_to_soma_raw.odd.data, synapse_row_dendrite_to_soma_index);

                causal_correlation_dendrite_to_soma[synapse_column_dendrite_to_soma_index] = -(causal_correlation_dendrite_to_soma_raw.convert_contiguous() - correlation_baseline)[synapse_column_dendrite_to_soma_index];

                ++synapse_column_dendrite_to_soma_index;
            }}
            causal_correlation_dendrite_to_soma = translate_columns(causal_correlation_dendrite_to_soma, synapses_dendrite_to_soma, synapses_soma_to_dendrite);


            //size_t used_synapse_row_index = 0;
            //size_t synapse_row_dendrite_to_soma_index = 0;
            //size_t synapse_row_soma_to_soma_index = 0;
            size_t pre_neuron_soma_index = 0;
            for (size_t used_synapse_row_index = 0; used_synapse_row_index < synapse_rows[0].size(); ++used_synapse_row_index) {{
                size_t const synapse_row_soma_to_dendrite_index = synapse_rows[0][used_synapse_row_index];
                size_t const synapse_row_soma_to_soma_index = synapse_rows[1][used_synapse_row_index];
                size_t const synapse_row_dendrite_to_soma_index = synapse_rows[2][used_synapse_row_index];
            //for (size_t synapse_row_soma_to_dendrite_index = 0; synapse_row_soma_to_dendrite_index < synapses_soma_to_dendrite.rows.size; ++synapse_row_soma_to_dendrite_index) {{
                if (!synapses_soma_to_dendrite.rows.test(synapse_row_soma_to_dendrite_index)) {{
                    exit(1);
                    //continue;
                }}

                //while(!synapses_soma_to_soma.rows.test(synapse_row_soma_to_soma_index)) {{
                //   synapse_row_soma_to_soma_index++;
                //}}

                //while(!synapses_dendrite_to_soma.rows.test(synapse_row_dendrite_to_soma_index)) {{
                //   synapse_row_dendrite_to_soma_index++;
                //}}

                while(!somas.columns.test(pre_neuron_soma_index)) {{
                   pre_neuron_soma_index++;
                }}

                // calculate new column mask masking out recurrent connections
                // number of neurons per symbol times three
                size_t column_mask_begin = (used_synapse_row_index / 15) * 30;
                size_t column_mask_end = ((used_synapse_row_index / 15) + 1) * 30;
                VectorRowMod8 column_mask(1);
                for (size_t column_mask_index = column_mask_begin; column_mask_index < column_mask_end; ++column_mask_index) {{
                    column_mask[column_mask_index] = 0;
                }}

                // update column mask to incorporate sampling of connections,
                // -> only p percent of connections should be active
                for (size_t column_mask_index = 0; column_mask_index < synapses_soma_to_dendrite.columns.size; column_mask_index+=2) {{
                    //if ((column_mask_index/2) % {round(1 / self.p_exc_exc)} != 0) {{
                    //    column_mask[column_mask_index] = 0;
                    //}}
                    if (xorshift32(&seed) >= {round(self.p_exc_exc * 2 ** 32)}) {{
                        column_mask[column_mask_index] = 0;
                    }}
                }}


                // get causal correlations and reset accumulated signals
                // in [-128, 127] integer
                VectorRowMod8 causal_correlation_soma_to_soma_raw;
                get_causal_correlation(&causal_correlation_soma_to_soma_raw.even.data, &causal_correlation_soma_to_soma_raw.odd.data, synapse_row_soma_to_soma_index);

                auto const causal_correlation_soma_to_soma = -translate_columns(causal_correlation_soma_to_soma_raw.convert_contiguous() - correlation_baseline, synapses_soma_to_soma, synapses_soma_to_dendrite);

                // get number of spikes since last invokation of pre neuron
                uint8_t const pre_neuron_soma_num_spikes = somas.get_rate_counter(pre_neuron_soma_index, true);

                // get permanence values, [-128, 127] integer
                auto& permanence = reinterpret_cast<VectorRowFracSat8&>(permanences[used_synapse_row_index]);

                // update permanence values
                permanence += vector_if(
                    column_mask, VectorIfCondition::greater,
                    vector_if(
                        causal_correlation_soma_to_soma - {self.correlation_threshold},
                        VectorIfCondition::greater_equal,
                        causal_correlation_soma_to_soma * {self.lambda_plus},
                        VectorRowFracSat8(0)),
                    VectorRowFracSat8(0));

                auto homeostasis = vector_if(
                    column_mask, VectorIfCondition::greater,
                    vector_if(
                        causal_correlation_soma_to_soma - {self.correlation_threshold},
                        VectorIfCondition::greater_equal,
                        ({self.target_rate_h} - causal_correlation_dendrite_to_soma) * {self.lambda_h},
                        VectorRowFracSat8(0)),
                    VectorRowFracSat8(0));

                permanence += homeostasis;



                // TODO: * 255 seems wrong, since the result is required to be in [-128, 127)
                auto depression = vector_if(column_mask, VectorIfCondition::greater,
                    VectorRowFracSat8(std::min(static_cast<size_t>(127), static_cast<size_t>(({self.lambda_minus} * 255) * (pre_neuron_soma_num_spikes / {self.num_runs})))),
                                       VectorRowFracSat8(0));

                permanence -= depression;
                
                permanence = vector_if(
                        permanence,
                        VectorIfCondition::greater_equal,
                        permanence,
                        VectorRowFracSat8(0));
                
                //auto permanence_tmp = vector_if(
                //        permanence-60,
                //        VectorIfCondition::greater_equal,
                //        permanence-60,
                //        VectorRowFracSat8(0));
                        
                //permanence_tmp = vector_if(
                //        permanence_tmp-50,
                //        VectorIfCondition::greater_equal,
                //        VectorRowFracSat8(50),
                //        permanence_tmp);

                // update weights
                auto weights = synapses_soma_to_dendrite.get_weights(synapse_row_soma_to_dendrite_index);

                // performs: if (permanence >= {self.permanence_threshold}) {{ weight = {self.w_mature}; }} else {{ weight = 0; }}
                weights = vector_if(permanence - {self.permanence_threshold}, VectorIfCondition::greater_equal, VectorRowMod8({self.w_mature}), VectorRowMod8(0));

                // mask out recurrent connections
                // auto weights = VectorRowMod8(permanence);
                weights *= column_mask;

                synapses_soma_to_dendrite.set_weights(weights, synapse_row_soma_to_dendrite_index);

                // record observables
                for (size_t active_column = 0, column = 0; column < synapses_soma_to_dendrite.columns.size; ++column) {{
                    if (!synapses_soma_to_dendrite.columns.test(column)) {{
                        continue;
                    }}
                    std::get<0>(recording.data)[used_synapse_row_index][active_column] = weights[column];
                    std::get<1>(recording.data)[used_synapse_row_index][active_column] = pre_neuron_soma_num_spikes;
                    std::get<2>(recording.data)[used_synapse_row_index][active_column] = permanence[column];
                    std::get<1>(recording.correlation)[used_synapse_row_index][active_column] = causal_correlation_soma_to_soma[column];
                    std::get<2>(recording.correlation)[used_synapse_row_index][active_column] = causal_correlation_dendrite_to_soma[column];
                    active_column++;
                }}
                //++synapse_row_soma_to_soma_index;
                //++synapse_row_dendrite_to_soma_index;
                ++pre_neuron_soma_index;
                //++used_synapse_row_index;
            }}
            reset_all_correlations();

            // required to expose symbol, TODO: remove once fixed
            do_not_optimize_away(dummy);
            dummy += 1;
        }}
        """)