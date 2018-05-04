from . import Network

from .. import parser_tools as parset
from .. import parser

from .. parser_tools import main_register, ArgumentException
from ..variable import Variable


class KerasNetwork(Network):
    arguments = parset.ClassArguments('KerasNetwork', Network.arguments,
                                      ('model', False, None, str),
                                      order=["model", "load", "store",
                                             "formats", "variables", "id"]
                                      )
    def __init__(self, model, load=None, store=None, formats=None,
                 variables={}, id=None):
        Network.__init__(self, load, store, formats, variables, id)
        self.model = model

    def _initialize(self):
        pass

    def _finalize(self):
        pass

    def _store(self, path, formats):
        pass

    def train(self, format, data, epochs=1):
        pass

    def evaluate(self, format, data):
        pass

    @staticmethod
    def parse(tree, item_cache):
        obj = parser.try_lookup_obj(tree, item_cache, Network, None)
        if obj is not None:
            return obj
        else:
            raise ArgumentException("The definition of the keras network can "
                                    "only be used for look up of any previously"
                                    " defined schema via 'Sampler(id=ID)'")


main_register.append_register(KerasNetwork, "keras_network")


class MLPKeras(KerasNetwork):
    arguments = parset.ClassArguments('MLPKeras', KerasNetwork.arguments)

    def __init__(self, model, load=None, store=None, formats=None,
                 variables={}, id=None):
        KerasNetwork.__init__(self,model, load, store, formats, variables, id)

    @staticmethod
    def parse(tree, item_cache):
        obj = parser.try_lookup_obj(tree, item_cache, Network, None)
        if obj is not None:
            return obj
        else:
            raise ArgumentException("The definition of the MLPKeras network can "
                                    "only be used for look up of any previously"
                                    " defined schema via 'Sampler(id=ID)'")


main_register.append_register(KerasNetwork, "keras")