from inspect import isclass
from abc import ABC

import neuroseq
import logging
import os
import string
import numpy as np

from os.path import join, dirname, split

# Workaround to remove "Invalid MIT-MAGIC-COOKIE-1 key" error message caused by import of mpi4py in NumpyRNG (pyNN)
os.environ["HWLOC_COMPONENTS"] = "-gl"


PY_PKG_PATH = split(dirname(neuroseq.__file__))[0]


class NeuronTypeABC(ABC):
    ID: int = None
    NAME: str = None
    NAME_PRINT: str = None
    COLOR_ID: int = None

    @classmethod
    def get_name_print(cls):
        return cls.NAME.replace("_", " ").title()


class NeuronType:
    class Dendrite(NeuronTypeABC):
        ID = 0
        NAME = "dendrite"
        NAME_PRINT = "Dendrite"
        COLOR_ID = 0

    class Soma(NeuronTypeABC):
        ID = 1
        NAME = "soma"
        NAME_PRINT = "Soma"
        COLOR_ID = 1

    class Inhibitory(NeuronTypeABC):
        ID = 2
        NAME = "inhibitory"
        NAME_PRINT = "Local Inh."
        COLOR_ID = 2

    class InhibitoryGlobal(NeuronTypeABC):
        ID = 3
        NAME = "inhibitory_global"
        NAME_PRINT = "Global Inh."
        COLOR_ID = 3

    @staticmethod
    def get_all_types():
        all_types = list()
        for n_type_name, n_type in NeuronType.__dict__.items():
            if isclass(n_type):
                all_types.append(n_type)
        return all_types


class RecTypes:
    SPIKES = "spikes"
    V = "v"


class NamedStorage:
    @classmethod
    def get_all(cls, case=str):
        return [case(v) for n, v in vars(cls).items() if not (n.startswith('_') or callable(v))]


class Backends(NamedStorage):
    BRAIN_SCALES_2 = 'bss2'
    NEST = 'nest'


class RunType(NamedStorage):
    MULTI = "multi"
    SINGLE = "single"


class NetworkState(NamedStorage):
    PREDICTIVE = "predictive"
    REPLAY = "replay"


class EncodingType(NamedStorage):
    DEFAULT = "default"
    PROBABILISTIC = "probabilistic"


class ReplayMode(NamedStorage):
    PARALLEL = "parallel"
    CONSECUTIVE = "consecutive"


class DendriteState(NamedStorage):
    INACTIVE = "inactive"
    WEAK = "weak"
    PREDICTIVE = "predictive"
    DUPLICATE = "duplicate"


class SomaState(NamedStorage):
    INACTIVE = "inactive"
    ACTIVE = "active"


class Colors(NamedStorage):
    BLACK = "black"
    GREY = "grey"
    BLUE = "blue"
    LIGHT_BLUE = "lightblue"
    DARK_BLUE = "darkblue"
    PURPLE = "purple"
    RED = "red"
    GREEN = "green"


class FileType(NamedStorage):
    DATA = 'data'
    FIGURE = 'figure'
    MODEL = 'model'
    OPTIMIZATION = 'optimization'


class PlotFileType(NamedStorage):
    PDF = "pdf"
    PNG = "png"
    JPG = "jpg"


class ExperimentType(NamedStorage):
    EVAL_SINGLE = 'eval-single'
    EVAL_MULTI = 'eval-multi'
    OPT_GRID = 'opt-grid'
    OPT_GRID_MULTI = 'opt-grid-multi'
    INSTANCE = 'instance'


class ConfigType(NamedStorage):
    NETWORK = 'network'
    PLOTTING = 'plotting'


class PlasticityLocation(NamedStorage):
    ON_CHIP = 'on-chip'
    OFF_CHIP = 'off-chip'


class ParameterMatchingType(NamedStorage):
    ALL = 'all'
    SINGLE = 'single'


class PerformanceType(NamedStorage):
    ALL_SYMBOLS = "all_symbols"
    LAST_SYMBOL = "last_symbol"


class PerformanceMetrics(NamedStorage):
    ERROR = 'error'
    FP = 'false_positive'
    FN = 'false_negative'
    ACTIVE_SOMAS = 'active_somas'
    # ACTIVE_DENDRITES = 'active_dendrite'
    DD = 'duplicate_dendrites'


class StatisticalMetrics(NamedStorage):
    MEAN = 'mean'
    STD = 'std'
    MEDIAN = 'median'
    PERCENTILE = 'percentile'


class LogHandler(NamedStorage):
    FILE = 0
    STREAM = 1


class RuntimeConfig(NamedStorage):
    backend = None
    plasticity_location = PlasticityLocation.OFF_CHIP
    backend_initialization = False
    config_prefix = "neuroseq_config"
    plot_file_types = [PlotFileType.PNG, PlotFileType.PDF]
    subnum_digits = 2
    instance_digits = 2
    saved_weights = ["exc_to_exc", "exc_to_inh"]
    saved_events = [NeuronType.Soma, NeuronType.Dendrite, NeuronType.Inhibitory]
    saved_network_vars = ["trace_dendrites"]
    saved_plasticity_vars = ["permanence", "permanence_min", "permanences", "weights", "x", "z"]
    saved_instance_params = []


# Logging
class Log(NamedStorage):
    FILE = join(PY_PKG_PATH, 'data/log/neuroseq.log')
    # FORMAT_FILE = "[%(asctime)s] [%(filename)s:%(lineno)s - %(funcName)20s() ] [%(levelname)-8s] %(message)s"
    FORMAT_FILE = "[%(asctime)s] [%(filename)-20s:%(lineno)-4s] [%(levelname)-8s] %(message)s"
    FORMAT_SCREEN_COLOR = "%(log_color)s%(message)s"
    FORMAT_SCREEN_NO_COLOR = "%(message)s"
    LEVEL_FILE = logging.INFO
    LEVEL_SCREEN = logging.INFO
    DATEFMT = '%d.%m.%Y %H:%M:%S'


PATH_CONFIG = join(PY_PKG_PATH, 'config')
PATH_MODELS = join(PY_PKG_PATH, 'models')
PATH_MAPS = join(PY_PKG_PATH, 'data', 'maps')

CONFIG_FOLDERS = {
    ConfigType.NETWORK: PATH_CONFIG,
    ConfigType.PLOTTING: join(PATH_CONFIG, ConfigType.PLOTTING)
}

EXPERIMENT_FOLDERS = {
    Backends.NEST: join(PY_PKG_PATH, 'data/evaluation/nest'),
    Backends.BRAIN_SCALES_2: join(PY_PKG_PATH, 'data/evaluation/bss2')
}
EXPERIMENT_SUBFOLDERS = {
    ExperimentType.EVAL_SINGLE: 'single',
    ExperimentType.EVAL_MULTI: 'multi',
    ExperimentType.OPT_GRID: 'grid',
    ExperimentType.OPT_GRID_MULTI: 'grid-multi'
}
EXPERIMENT_SETUP_FILE_NAME = {
    ExperimentType.EVAL_SINGLE: 'experiments_single.csv',
    ExperimentType.EVAL_MULTI: 'experiments_multi.csv',
    ExperimentType.OPT_GRID: 'experiments_grid.csv',
    ExperimentType.OPT_GRID_MULTI: 'experiments_grid-multi.csv',
    ExperimentType.INSTANCE: 'experimental_results.csv'
}

SYMBOLS = {symbol: index for index, symbol in enumerate(string.ascii_uppercase)}

NP_STATISTICS = {m: getattr(np, m) for m in StatisticalMetrics.get_all()}
