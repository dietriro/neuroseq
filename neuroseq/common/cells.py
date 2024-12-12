from pyNN.standardmodels.cells import MultiCompartmentNeuron
from pyNN.nest.populations import PopulationView, Assembly
from pyNN.nest import simulator
from pyNN import common


class MCPopulationView(PopulationView):
    __doc__ = common.PopulationView.__doc__
    _simulator = simulator
    _assembly_class = Assembly

    def __init__(self, parent, selector, label=None):
        super().__init__(parent, selector, label=None)
