from . import Network

from .. import parser_tools as parset
from .. import parser
from .. import main_register

from ..variable import Variable


class KerasNetwork(Network):
    arguments = parset.ClassArguments('KerasNetwork', Network.arguments,
                                      ('path', False, None, str),
                                      order=["path", "do_store",
                                             "variables", "id"]
                                      )

    def __init__(self, path, do_store=False, variables={}, id=None):
        Network.__init__(self, do_store, variables, id)
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


    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  KerasNetwork)

main_register.append_register(KerasNetwork, "keras")