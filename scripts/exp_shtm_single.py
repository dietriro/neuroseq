import sys
import os 
import warnings
import numpy as np

pkg_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(pkg_path)

np.set_printoptions(threshold=np.inf, suppress=True, linewidth=np.inf)
warnings.filterwarnings(action='ignore', category=UserWarning)

import neuroseq.addsrc
from neuroseq.common.config import *
from neuroseq.core.logging import log

np.set_printoptions(threshold=np.inf, suppress=True, linewidth=np.inf)
warnings.filterwarnings(action='ignore', category=UserWarning)

# In[3]:


RuntimeConfig.backend = Backends.BRAIN_SCALES_2


# In[4]:


shtm = None
if RuntimeConfig.backend == Backends.BRAIN_SCALES_2:
    import neuroseq.brainscales2.patches
    from neuroseq.brainscales2.hardware import hardware_initialization
    from neuroseq.brainscales2.network import SHTMTotal
    # TODO: remove once grenade supports dense inter-population-view projections
    neuronPermutation = []
    shtm = SHTMTotal(use_on_chip_plasticity=True)
    alphabet_size = shtm.p.network.num_symbols
    num_neurons_per_symbol = shtm.p.network.num_neurons
    for a in range(alphabet_size):
        # dendrites
        for i in range(num_neurons_per_symbol):
            neuronPermutation.append((a * num_neurons_per_symbol + i) * 2)
    for a in range(alphabet_size):
        # somas
        for i in range(num_neurons_per_symbol):
            neuronPermutation.append((a * num_neurons_per_symbol + i) * 2 + 1)
    for i in range(alphabet_size * num_neurons_per_symbol * 2, 512):
        neuronPermutation.append(i)

    hardware_initialization(neuron_permutation=neuronPermutation)
elif RuntimeConfig.backend == Backends.NEST:
    from neuroseq.nest.network import SHTMTotal
    
from neuroseq.common.network import NeuronType, RecTypes


# ## Configuration

# In[5]:



#log.handlers[LogHandler.STREAM].setLevel(logging.DEBUG)

debug = False

experiment_num = None

### Initialize

# ## Network Initialization

# In[6]:


if shtm is None:
    shtm = SHTMTotal()

# FIXME: the timer only starts at start_time + period, fake calculation
# to get exactly one execution at the end of each runtime
runtime=0.
shtm.init_plasticity_rule(start_time=0., period=30., runtime=runtime)
shtm.init_neurons()
shtm.init_connections(debug=debug)
shtm.init_external_input()

if RuntimeConfig.backend == Backends.BRAIN_SCALES_2:
    shtm.init_rec_exc()
    
shtm.experiment_num = experiment_num

### Run

# In[7]:

# shtm.set_weights_exc_exc(10, 1, post_ids=[3,4,5], p_con=1.0)
shtm.run(steps=100, plasticity_enabled=True)

#shtm = SHTMTotal.load_full_state(SHTMTotal, "test", 12)


# ## Network Emulation & Plotting

# In[8]:


import pynn_brainscales.brainscales2 as pynn
pynn.logger.default_config(level=pynn.logger.LogLevel.TRACE)
shtm.run(steps=5, plasticity_enabled=False)


# In[9]:


from matplotlib import pyplot as plt

#print(shtm.con_plastic[0].permanences)
data = np.array(shtm.con_plastic[0].x[0]).reshape((60,60))
print(data.shape)
plt.imshow(data)
plt.colorbar()


# In[10]:


from matplotlib import pyplot as plt

#print(shtm.con_plastic[0].permanences)
data = np.array(shtm.con_plastic[0].z[0]).reshape((60,60))
print(data.shape)
plt.imshow(data)
plt.colorbar()


# In[11]:


from matplotlib import pyplot as plt

#print(shtm.con_plastic[0].permanences)
data = np.array(shtm.con_plastic[0].permanences[0]).reshape((60,60))
print(data.shape)
plt.imshow(data)
plt.colorbar()


# In[12]:


from matplotlib import pyplot as plt

#print(shtm.con_plastic[0].permanences)
data = np.array(shtm.con_plastic[0].weights[0]).reshape((60,60))
print(data.shape)
plt.imshow(data)
plt.colorbar()


# In[ ]:



shtm.plot_events(neuron_types="all", size=(24, 10))

# shtm.plot_v_exc(alphabet_range=[0], neuron_range=range(3), neuron_type=NeuronType.Inhibitory, show_legend=True)

# In[ ]:



shtm.plot_performance(StatisticalMetrics.MEAN)


# In[ ]:


shtm.save_full_state()


# In[ ]:


shtm.run(steps=10, plasticity_enabled=False)


# In[ ]:



shtm.plot_events(neuron_types="all", size=(24, 10))


# In[ ]:



shtm.plot_performance()


# In[ ]:


shtm.run(steps=10, plasticity_enabled=False)


# In[ ]:



shtm.plot_events(neuron_types="all", size=(24, 10))


# In[ ]:


shtm.set_weights_exc_exc(10, 0, post_ids=[3,4,5], p_con=1.0)
shtm.run(steps=1, plasticity_enabled=False)

# shtm.plot_events(neuron_types="all", size=(24, 10))


# ## Additional Plotting

# In[ ]:



shtm.plot_permanence_diff()


# In[ ]:



for i in [0, 1]:
    print(shtm.con_plastic[i].permanences)

shtm.plot_permanence_history(plot_con_ids=[0, 1])


# In[ ]:


from neuroseq.common.network import NeuronType, RecTypes

shtm.plot_v_exc(alphabet_range=[2], neuron_range="all", neuron_type=NeuronType.Soma)


# In[ ]:



shtm.plot_v_exc(alphabet_range=[0], neuron_range='all', neuron_type=1, show_legend=False)


# In[ ]:



shtm.plot_v_exc(alphabet_range=range(1, alphabet_size))


# ## Additional Analysis

# In[ ]:


for i in range(len(shtm.con_plastic)):
    shtm.con_plastic[i].mature_weight = 120
    print(i, shtm.con_plastic[i].projection.label.split('_')[1], shtm.con_plastic[i].get_all_connection_ids())
    


# In[ ]:


arr = np.array(shtm.con_plastic[1].permanences)


# In[ ]:


for c in shtm.con_plastic[1].projection.connections:
    print(f'C[{c.presynaptic_index}, {c.postsynaptic_index}].weight = {c.weight}')


# In[ ]:


np.set_printoptions(threshold=np.inf, suppress=True, linewidth=np.inf)
shtm.con_plastic[1].projection.get("weight", format="array")


# In[ ]:


shtm.con_plastic[7].projection.get("weight", format="array")


# In[ ]:


# Print spikes form spiketrain
for s in shtm.con_plastic[1].projection.post.get_data("spikes").segments[-1].spiketrains:
    print(s)
print(len(shtm.con_plastic[1].projection.post.get_data("spikes").segments[-1].spiketrains))
print(len(shtm.con_plastic[1].projection.connections))


# In[ ]:


for con in shtm.con_plastic:
    print(f"Sum(P[{con.projection.label}]) = {np.sum(con.permanences[-1] - con.permanences[0])}")


# In[ ]:


dendrites, somas = shtm.get_spike_times(0.44, 0.1e-2)

print(somas)


# In[ ]:


for i_plastic in range(len(shtm.con_plastic)):
    shtm.con_plastic[i_plastic].lambda_plus *= 2
    shtm.con_plastic[i_plastic].lambda_minus *= 2
    shtm.con_plastic[i_plastic].lambda_h *= 2
    
    # print(f"Sum(P[{con.projection.label}]) = {np.sum(con.permanences[-1] - con.permanences[0])}")


# ## Check indices

# In[ ]:


dendrites, somas = shtm.neurons_exc[0]

print(somas.all_cells)
print(somas.id_to_index(13))

print(somas.get_data("spikes").segments[-1].spiketrains[8])
print(shtm.con_plastic[0].projection.pre.get_data("spikes").segments[-1].spiketrains[8])


# ## Check spikes

# In[ ]:


dendrites, somas = shtm.neurons_exc[0]

spike_ids_a = list()
spike_ids_b = list()

# Print spikes form spiketrain
for s in somas.get_data("spikes").segments[-1].spiketrains:
    print(s)
    if len(s) > 0:
        print(s[0]/0.1e-3)
        spike_ids_a.append(int(s[0]/0.1e3))

    # for t in np.linspace(0., runtime, int(runtime / 0.1e-3)):
        
    
print(len(somas.get_data("spikes").segments[-1].spiketrains))


# In[ ]:


print(shtm.neurons_exc[0][0].get("tau_m"))
print(shtm.neurons_inh[0].tau_m)
w = shtm.exc_to_inh[1].get("weight", format="array")
print(w)
print(pynn.get_current_time())


# ## Save objects

# In[ ]:





# In[ ]:


import pickle

experiment_name = "shtm_off-chip_01"

with open(f'../evaluation/objects/{experiment_name}.pkl', 'wb') as out_file:
    pickle.dump(shtm, out_file)


# In[ ]:


with open(f'../evaluation/objects/{experiment_name}.pkl', 'rb') as in_file:
    obj = pickle.load(in_file)


# ## Plotting - Events - All Symbols

# In[ ]:



seq = 2

fig_title = "Neuronal Events for Sequence {D, C, B} - After Learning"

file_path = f"../evaluation/figures/shtm-bss2_eval_learning-off-chip_seq-0{seq}_before-learning"
# file_path = f"../evaluation/figures/shtm-bss2_eval_learning-off-chip_seq-0{seq}_after-learning"

# file_path += "_a"

if seq == 1:
    fig = shtm.plot_events(shtm, size=[12, 10], x_lim_lower=0, x_lim_upper=0.14, seq_start=0.0, seq_end=0.14, fig_title=fig_title, file_path=file_path)
elif seq == 2:
    fig = shtm.plot_events(shtm, size=[12, 10], x_lim_lower=0.22, x_lim_upper=0.36, seq_start=0.22, seq_end=0.36, fig_title=fig_title, file_path=file_path)
    


# ## Plotting - Events - One Symbol

# In[ ]:



fig_title = "Neuronal Events for Sequence {D, C, B} - After Learning"
file_path = f"../evaluation/figures/shtm-bss2_eval_limits-spikes_w-inh"

# file_path += "_a"

fig = shtm.plot_events(shtm, neuron_types="all", symbols=[2], size=[12, 10], x_lim_lower=0, x_lim_upper=0.14, seq_start=0.0, seq_end=0.14, fig_title=fig_title, file_path=file_path)


spikes = shtm.neurons_exc[2][1].get_data("spikes").segments[-1].spiketrains


# In[ ]:


# file_path_open = f"../evaluation/figures/shtm-bss2_eval_limits-volts_w-inh"
# file_path_open = f"../evaluation/figures/shtm-bss2_eval_limits-volts_wo-inh"
# file_path_open = f"../evaluation/figures/shtm-bss2_eval_limits-spikes_w-inh"
file_path_open = f"../evaluation/figures/shtm-bss2_eval_limits-spikes_wo-inh"

figx = pickle.load(open(f'{file_path_open}.fig.pickle', 'rb'))
figx.set_size_inches(12, 6)

figx.legends = []
figx.suptitle("")
figx.subplots_adjust(top=0.85)
neuron_types = [NeuronType.Dendrite, NeuronType.Soma, NeuronType.Inhibitory]
custom_lines = [Line2D([0], [0], color=f"C{n.ID}", label=n.NAME.capitalize(), lw=3) for n in neuron_types]
plt.figlegend(handles=custom_lines, loc=(0.402, 0.888), ncol=3, labelspacing=0., fontsize=18, fancybox=True, borderaxespad=1)


# figx.show()

figx.savefig(f"{file_path_open}.pdf", bbox_inches='tight')
# figx.savefig(f"{file_path_open}.png")


# ## Final Plots - Voltage

# In[ ]:



# plt.rcParams.update({'font.size': 12})

seq = 2

# file_path = f"../evaluation/figures/shtm-bss2_eval_learning-off-chip_seq-0{seq}_before-learning"
# file_path = f"../evaluation/figures/shtm-bss2_eval_learning-off-chip_seq-0{seq}_after-learning"
file_path = f"../evaluation/figures/shtm-bss2_eval_limits-volts_w-inh"

# file_path += "_a"

# if seq == 1:
    # fig = plot_v_exc(shtm, [0], neuron_range="all", size=[12, 10], x_lim_lower=0, x_lim_upper=0.14, seq_start=0.0, seq_end=0.14, file_path=file_path)
# elif seq == 2:
    # fig = plot_v_exc(shtm, [0], neuron_range="all", size=[12, 10], x_lim_lower=0.22, x_lim_upper=0.36, seq_start=0.22, seq_end=0.36, file_path=file_path)

shtm.plot_v_exc(shtm, [2], neuron_range="all", size=[12, 10], runtime=0.14, file_path=file_path)

    


# In[ ]:




