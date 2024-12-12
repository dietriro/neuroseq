from base import *

# experiment
experiment_map = "ambiguous_03"
experiment_num = 1
target = None
num_replay_steps = 2
custom_params = {
    "replay.threshold_delta_t_up": 60,
}
threshold_delta_t_up = None
# plotting
path_config_plotting = "./config/config_plotting_ambig_02b.yaml"
custom_labels = [
    {"label": "STDTA", "color": "C4"},
    {"label": "ADTA", "color": "C6"},
]
replay_runtime = 200
fig_title = "Place Disambiguation 2b"
plot_thresholds = True
save_plot = False
