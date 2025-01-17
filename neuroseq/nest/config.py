from neuroseq.common.config import *

RuntimeConfig.backend = Backends.NEST

DEFAULT_MC_MODEL_FILE = join(RuntimeConfig.Paths.models, "mc", "iaf_psc_exp_nonlineardendrite.nestml")
