import numpy as np
import copy
import pynn_brainscales.brainscales2 as pynn

from _static.common.helpers import setup_hardware_client


def hardware_initialization(neuron_permutation):
    # Setup BrainScaleS-2
    setup_hardware_client()

    # setup PyNN and set calibration cache path
    from pathlib import Path
    pynn.setup(calibration_cache=[Path(".calix")], neuronPermutation=neuron_permutation)