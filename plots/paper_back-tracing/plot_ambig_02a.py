from base import *

# experiment
experiment_map = "ambiguous_03"
experiment_num = 1
target = None
num_replay_steps = 2
custom_params = {}
threshold_delta_t_up = None
# plotting
path_config_plotting = "./config/config_plotting_ambig_02a.yaml"
custom_labels = [
    {"label": "STDTA", "color": "C4"},
    {"label": "ADTA", "color": "C6"},
]
replay_runtime = 200
fig_title = "Place Disambiguation 2a"
plot_thresholds = True
save_plot = False
