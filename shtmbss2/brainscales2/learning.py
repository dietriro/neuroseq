import copy
from abc import ABC

import numpy as np

import shtmbss2.common.learning
from pynn_brainscales import brainscales2 as pynn
from pynn_brainscales.brainscales2 import Projection
from shtmbss2.brainscales2.network import SHTMTotal
from shtmbss2.core.logging import log


class Plasticity(shtmbss2.common.learning.Plasticity):
    def __init__(self, projection: pynn.Projection, post_somas, shtm, index, **kwargs):
        super().__init__(projection, post_somas, shtm, index, **kwargs)

    def get_connection_id_pre(self, connection):
        return connection.presynaptic_index

    def get_connection_id_post(self, connection):
        return connection.postsynaptic_index

    def get_connections(self):
        return self.projection.connections


class PlasticitySingleNeuron:
    def __init__(self, projection: pynn.Projection, post_somas: pynn.PopulationView):
        self.projection = projection
        log.debug("inside constructor")

        self.permanence_min = np.asarray(np.random.randint(0, 8, size=(len(self.projection),)), dtype=float)
        self.permanence = copy.copy(self.permanence_min)
        self.permanence_max = 20.
        self.permanence_threshold = np.ones((len(self.projection))) * 20.
        self.lambda_plus = 0.08 * 1e3
        self.tau_plus = 20. / 1e3
        self.lambda_minus = 0.0015 * 1e3
        self.target_rate_h = 1.
        self.lambda_h = 0.014 * 1e3
        self.tau_h = 440. / 1e3
        self.y = 1.
        self.delta_t_min = 4e-3
        self.delta_t_max = 8e-3
        self.dt = 0.1e-3
        self.post_somas = post_somas
        self.mature_weight = 63
        self.debug = False


class OnChipPlasticityDummy(ABC):
    def __init__(self, projection: Projection, shtm, id=0):
        # custom objects
        self.projection = projection
        self.shtm: SHTMTotal = shtm

        self.permanences = []
        self.rates = []
        self.permanence = None
        self.permanence_min = 0
        self.weights = []
        self.x = []
        self.z = []

        self.id = id

    def update_values(self, weights, rates, permanences, x, z):
        # save weights
        self.weights.append(weights)

        # save rates
        self.rates.append(rates)

        # save permanences
        self.permanences.append(permanences)
        self.permanence = self.permanences[-1]

        # save correlations
        self.x.append(x)
        self.z.append(z)

    def enable_permanence_logging(self):
        pass

    def enable_weights_logging(self):
        pass

    def get_connection_ids(self, connection_id):
        connection_ids = (f"{self.get_connection_id_pre(self.get_connections()[connection_id])}>"
                          f"{self.get_connection_id_post(self.get_connections()[connection_id])}")
        return connection_ids

    def get_connection_id_pre(self, connection):
        return connection.presynaptic_index

    def get_connection_id_post(self, connection):
        return connection.postsynaptic_index

    def get_all_connection_ids(self):
        connection_ids = []
        for con in self.get_connections():
            connection_ids.append(f"{self.get_connection_id_pre(con)}>{self.get_connection_id_post(con)}")
        return connection_ids

    def get_connections(self):
        return self.projection.connections
