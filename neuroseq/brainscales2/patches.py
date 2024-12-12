import pickle
import numpy as np
import quantities as pq
import pynn_brainscales.brainscales2 as pynn
import calix.spiking
from calix import calibrate
from dlens_vx_v3 import sta


# TODO: remove once calib target and options can be injected
def patch_pynn_calibration(padi_bus_dacen_extension = 0, correlation_amplitude = 1.5, correlation_time_constant = 30):
    def pynn_simulator_state_configure_recorders_populations(self):        
        changed = set()
        for population in self.populations:
            if population.changed_since_last_run:
                changed.add(population)
        if not changed:
            return
        
        correlation_target = calix.spiking.correlation.CorrelationCalibTarget(
            amplitude=correlation_amplitude,
            time_constant=correlation_time_constant * pq.us
        )
        
        calibration_options = calix.spiking.SpikingCalibOptions(
            correlation_options=calix.spiking.correlation.CorrelationCalibOptions(
                calibrate_synapses=False,
                branches=calix.spiking.correlation.CorrelationBranches.CAUSAL,
                default_amp_calib=1, v_res_meas=0.95 * pq.V
            )
        )
        
        neuron_target = calix.spiking.neuron.NeuronCalibTarget().DenseDefault
        # initialize shared parameters between neurons with None to allow check
        # for different values in different populations
        neuron_target.synapse_dac_bias = None
        neuron_target.i_synin_gm = np.array([None, None])
        
        # gather calibration information
        execute_calib = False
        for population in changed:
            assert isinstance(population.celltype, pynn.standardmodels.cells_base.StandardCellType)
            if hasattr(population.celltype, 'add_calib_params'):
                population.celltype.add_calib_params(
                    neuron_target, population.all_cells)
                execute_calib = True
        
        if execute_calib:
            if self.initial_config is not None:
                self.log.WARN("Using automatically calibrating neurons with "
                              "initial_config. Initial configuration will be "
                              "overwritten")
            calib_target = calix.spiking.SpikingCalibTarget(
                neuron_target=neuron_target,
                correlation_target=correlation_target
            )
            # release JITGraphExecuter connection to establish a new one for
            # calibration (JITGraphExecuter conenctions can not be shared with
            # lower layers).
            if self.conn is not None and not self.conn_comes_from_outside:
                self.conn_manager.__exit__()
            result = calibrate(
                calib_target,
                calibration_options,
                self.calib_cache_dir)
            # load calibration
            #with open("correlation_calix-native.pkl", "rb") as calibfile:
            #    result = pickle.load(calibfile)
            if self.conn is not None and not self.conn_comes_from_outside:
                self.conn = self.conn_manager.__enter__()
            dumper = sta.PlaybackProgramBuilderDumper()
            result.apply(dumper)
            
            # The correlation voltages are set on the board and therefore
            # are not contained in lola.Chip(), so we inject them as a builder
            calib_builder = sta.PlaybackProgramBuilder()
            result.apply(calib_builder)
            self.injection_pre_static_config = calib_builder

            self.grenade_chip_config = sta.convert_to_chip(
                dumper.done(), self.grenade_chip_config)

            # disable neuron readout to CADC (observe correlation instead)
            for cadc_config in self.grenade_chip_config.cadc_readout_chains:
                for channels in [cadc_config.channels_causal,
                                 cadc_config.channels_acausal]:
                    for channel_config in channels:
                        channel_config.enable_connect_neuron = False

            # increase dacen pulse extension in synapse drivers for stronger synaptic events
            for synapse_driver_block in self.grenade_chip_config.synapse_driver_blocks:
                synapse_driver_block.padi_bus.dacen_pulse_extension.fill(synapse_driver_block.padi_bus.DacenPulseExtension(padi_bus_dacen_extension))
            for neuron in self.grenade_chip_config.neuron_block.atomic_neurons:
                neuron.reset.enable_multiplication = True

        for population in changed:
            if hasattr(population.celltype, 'add_to_chip'):
                population.celltype.add_to_chip(
                    population.all_cells, self.grenade_chip_config)

    pynn.simulator.State._configure_recorders_populations = pynn_simulator_state_configure_recorders_populations