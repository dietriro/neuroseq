from base import *

# experiment
experiment_map = "path-planning_02"
experiment_num = 1
target = SYMBOLS['J']
num_replay_steps = 3
custom_params = {}
threshold_delta_t_up = None
# plotting
path_config_plotting = "./config/config_plotting_path_01.yaml"
custom_labels = [
    {"label": "STDTA", "color": "C4"},
]
replay_runtime = 340
fig_title = "Path Planning (A -> J)"
plot_thresholds = True
fig_size = (15, 12)
save_plot = False
