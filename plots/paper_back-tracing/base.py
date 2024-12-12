import sys
import os
import warnings
import numpy as np

pkg_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(pkg_path)

np.set_printoptions(threshold=np.inf, suppress=True, linewidth=np.inf)
warnings.filterwarnings(action='ignore', category=UserWarning)

from shtmbss2.common.config import *

RuntimeConfig.backend = Backends.NEST
RuntimeConfig.plasticity_location = PlasticityLocation.OFF_CHIP

from shtmbss2.core.logging import log
from shtmbss2.nest.network import SHTMTotal
