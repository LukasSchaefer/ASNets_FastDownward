from . import Network

from .. import parser_tools as parset
from .. import parser
from .. import main_register

from ..variable import Variable


class KerasNetwork(Network):
    arguments = parset.ClassArguments('KerasNetwork', Network.arguments,
                                      ('path', False, None, str),
                                      order=["path", "store",
                                             "variables", "id"]
                                      )

    def __init__(self, load=None, store=None, variables={}, id=None):
        Network.__init__(self, store, variables, id)
        self.path = path

    def _initialize(self):
        pass

    def _finalize(self):
        pass

    def _store(self):
        pass

    def train(self, msgs, data):
        pass

    def evaluate(self):
        pass


    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  KerasNetwork)

main_register.append_register(KerasNetwork, "keras")