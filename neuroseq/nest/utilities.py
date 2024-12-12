import matplotlib.pyplot as plt

import nest
from pynestml.codegeneration.nest_code_generator_utils import NESTCodeGeneratorUtils

from neuroseq.core.logging import log
from neuroseq.nest.config import DEFAULT_MC_MODEL_FILE


def install_mc_neuron(version, nestml_file=DEFAULT_MC_MODEL_FILE):
    """
    Builds and installs the multi-compartment neuron model by Bouhadjar et al. (2022) using nestml.

    :param version: The version label of the new mc neuron model.
    :type version: str
    :param nestml_file: The name of the nestml file to be built.
    :type nestml_file: str
    :return: The module and neuron name, respectively.
    :rtype: tuple
    """

    module_name, neuron_name = NESTCodeGeneratorUtils.generate_code_for(nestml_neuron_model=nestml_file,
                                                                        uniq_id=f"mc{version}", logging_level="ERROR")


    nest.Install(module_name)

    log.info(f"Installed nestml module '{module_name}' with neuron '{neuron_name}'.")


def evaluate_neuron(module_name, neuron_name, neuron_parms=None, t_sim=100., plot=True):
    """
    Run a simulation in NEST for the specified neuron. Inject a stepwise
    current and plot the membrane potential dynamics and action potentials generated.

    Returns the number of postsynaptic action potentials that occurred.
    """
    dt = .1   # [ms]

    nest.ResetKernel()
    try:
        nest.Install(module_name)
    except :
        pass
    neuron = nest.Create(neuron_name)
    if neuron_parms:
        for k, v in neuron_parms.items():
            nest.SetStatus(neuron, k, v)

    sg = nest.Create("spike_generator", params={"spike_times": [10., 20., 30.]})

    multimeter = nest.Create("multimeter")
    record_from_vars = ["V_m", "I_dend", "dAP"]
    multimeter.set({"record_from": record_from_vars,
                    "interval": dt})
    sr_pre = nest.Create("spike_recorder")
    sr = nest.Create("spike_recorder")

    nest.Connect(sg, neuron, syn_spec={"weight": 50., "delay": 1., "receptor_type": 2})

    nest.Connect(multimeter, neuron)
    nest.Connect(sg, sr_pre)
    nest.Connect(neuron, sr)

    nest.Simulate(t_sim)

    mm = nest.GetStatus(multimeter)[0]
    timevec = mm.get("events")["times"]
    I_dend = mm.get("events")["I_dend"]
    V_m = mm.get("events")["V_m"]

    if plot:
        fig, axs = plt.subplots(2, 1)

        axs[0].plot(timevec, I_dend, 'k', label='on-grid')
        axs[0].set_xlabel('time [ms]')
        axs[0].set_ylabel(f'I_dend [mV]')
        axs[1].plot(timevec, V_m, 'k', label='on-grid')
        axs[1].set_xlabel('time [ms]')
        axs[1].set_ylabel(f'V_m [mV]')

        fig.show()
