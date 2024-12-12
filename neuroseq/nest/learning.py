from pyNN import nest as pynn

import neuroseq.common.learning


class Plasticity(neuroseq.common.learning.Plasticity):
    def __init__(self, projection: pynn.Projection, post_somas, shtm, index, **kwargs):
        super().__init__(projection, post_somas, shtm, index, **kwargs)

    def get_connection_id_pre(self, connection):
        return self.projection.pre.id_to_index(connection.source)

    def get_connection_id_post(self, connection):
        return self.projection.post.id_to_index(connection.target)

    def get_connections(self):
        return self.projection.nest_connections
