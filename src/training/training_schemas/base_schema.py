from .. import parser
from .. import parser_tools as parset
from .. import ABC

from ..networks import Network
from ..parser_tools import main_register, ArgumentException
from ..samplers import Sampler
from ..variable import Variable

import abc

class Schema(ABC):
    """
    Base class for Schemas. Schemas describe what shall be how run.
    E.g. A training schema calls the necessary methods to obtain data and train
    a network.
    """

    arguments = parset.ClassArguments('Schema', None,
                                      ('variables', True, {},
                                       main_register.get_register(Variable)),
                                      ('id', True, None, str))

    def __init__(self, variables={}, id=None):
        """
        CONSTRUCTOR.
        Creates the _last_value member variable.
        :param variables: Map of variables for this schema. The map key defines
                            the usage of the variable object given.
        :param id: id to identify this condition
        """
        if not isinstance(variables, dict):
            raise ArgumentException("The provided variables have to be a map. "
                                    "Please define them as {name=VARIABLE,...}.")
        self.variables = variables
        self.id = id

    @abc.abstractmethod
    def run(self):
        pass

    @staticmethod
    def parse(tree, item_cache):
        obj = parser.try_lookup_obj(tree, item_cache, Schema, None)
        if obj is not None:
            return obj
        else:
            raise ArgumentException("The definition of the base schema can "
                                    "only be used for look up of any previously"
                                    " defined schema via 'Schema(id=ID)'")


main_register.append_register(Schema, "schema")
scregister = main_register.get_register(Schema)


class SamplerNetworkSchema(Schema):

    arguments = parset.ClassArguments('SamplerNetworkSchema', Schema.arguments,
        ('sampler', True, None, main_register.get_register(Sampler)),
        ('network', True, None, main_register.get_register(Network)),
        order=["sampler", "network", "variables", "id"]
    )

    def __init__(self, sampler=None, network=None, variables={}, id=None):
        Schema.__init__(self, variables, id)
        self.sampler = sampler
        self.network = network

    @staticmethod
    def parse(tree, item_cache):
        obj = parser.try_lookup_obj(tree, item_cache, SamplerNetworkSchema, None)
        if obj is not None:
            return obj
        else:
            raise ArgumentException("The definition of the SamplerNetworkSchema"
                                    " can  only be used for look up of any "
                                    "previously defined schema via "
                                    "'SamplerNetworkSchema(id=ID)'")


main_register.append_register(SamplerNetworkSchema, "samplernetworkschema")


