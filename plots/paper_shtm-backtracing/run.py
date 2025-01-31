from plot_ambig_01 import *

shtm = SHTMTotal.load_full_state(network_type=SHTMTotal,
                                 experiment_id="eval",
                                 experiment_num=experiment_num,
                                 experiment_map=experiment_map,
                                 experiment_type=ExperimentType.EVAL_SINGLE,
                                 custom_params=custom_params
                                 )

shtm.set_state(NetworkMode.REPLAY,
               target=target
               )

shtm.run(steps=num_replay_steps, plasticity_enabled=False, runtime=1200)
shtm.print_thresholds()

shtm.p_plot.load_default_params(custom_path=path_config_plotting)

if save_plot:
    shtm.save_plot_events(neuron_types="all",
                          show_grid=False,
                          separate_seqs=True,
                          replay_runtime=replay_runtime,
                          plot_dendritic_trace=False,
                          enable_y_ticks=False,
                          x_tick_step=80,
                          fig_title=fig_title,
                          plot_thresholds=plot_thresholds,
                          custom_labels=custom_labels,
                          )
else:
    shtm.plot_events(neuron_types="all",
                     show_grid=False,
                     separate_seqs=True,
                     replay_runtime=replay_runtime,
                     plot_dendritic_trace=False,
                     enable_y_ticks=False,
                     x_tick_step=80,
                     fig_title=fig_title,
                     plot_thresholds=plot_thresholds,
                     custom_labels=custom_labels,
                     )
