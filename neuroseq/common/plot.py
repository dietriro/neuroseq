import numpy as np

from matplotlib import pyplot as plt


def plot_membrane(membrane_recorded, label=None):
    membrane = membrane_recorded.get_data("v").segments[-1].irregularlysampledsignals[0]
    plt.plot(membrane.times, membrane, alpha=0.5, label=label)


def plot_dendritic_events(axis, spikes_dend, spikes_post, color, label, tau_dap, y_offset=1.0, seq_start=0,
                          seq_end=None, epoch_end=None, plot_trace=True):
    for i_neuron in range(len(spikes_dend)):
        if len(spikes_dend[i_neuron]) <= 0:
            continue
        spikes_dend_i = np.array(spikes_dend[i_neuron])
        spikes_dend_i = spikes_dend_i[np.where(np.logical_and(spikes_dend_i > seq_start, spikes_dend_i < seq_end))]
        if len(spikes_dend_i) <= 0:
            continue

        for spike_dend_i in spikes_dend_i:
            if len(spikes_post[i_neuron]) <= 0:
                spike_post_i = spike_dend_i + tau_dap
            else:
                spikes_post_i = np.array(spikes_post[i_neuron])
                spikes_post_i = spikes_post_i[
                    np.where(np.logical_and(spikes_post_i > seq_start, spikes_post_i < seq_end))]
                if len(spikes_post_i) <= 0:
                    spike_post_i = spike_dend_i + tau_dap
                else:
                    spikes_post_i = spikes_post_i[
                        np.where(np.logical_and(spikes_post_i > spike_dend_i, spikes_post_i < spike_dend_i + tau_dap))]
                    if len(spikes_post_i) <= 0:
                        spike_post_i = spike_dend_i + tau_dap
                    else:
                        spike_post_i = spikes_post_i[0]

            axis.plot([spike_dend_i, spike_post_i], [i_neuron + y_offset, i_neuron + y_offset], color=color,
                      label=label)

            if plot_trace and epoch_end is not None and spike_dend_i < epoch_end:
                axis.plot([spike_dend_i, epoch_end], [i_neuron + y_offset, i_neuron + y_offset], color=color,
                          label=label, linewidth=6, alpha=0.2)



def plot_panel_label(s, pos, ax, title='', size=10):
    """Creates a panel label (A,B,C,...) for the current axis object of a matplotlib figure.

    NetworkParameters
    ----------
    s:   str
        panel label
    pos: tuple
        x-/y- position of panel label (in units relative to the size of the current axis)

    title: str
        additional text describing the figure
    """

    plt.text(pos[0], pos[1], s, transform=ax.transAxes, horizontalalignment='center', verticalalignment='top', size=size,
             weight='bold')
    # plt.text(pos[0], pos[1], r'\bfseries{}%s %s' % (s, title), transform=ax.transAxes, horizontalalignment='center', verticalalignment='center',
    #         size=10)
    plt.text(pos[0] + 0.1, pos[1], title, transform=ax.transAxes, verticalalignment='top', size=size)

    return 0