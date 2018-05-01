from . import SamplerNetworkSchema

from .. import parser
from .. import parser_tools as parset
from .. import main_register
from .. import Message

from ..conditions import Condition
from ..samplers import Sampler
from ..variable import Variable


class AlternatingSchema(SamplerNetworkSchema):
    arguments = parset.ClassArguments("AlternatingSchema",
        SamplerNetworkSchema.arguments,
        ('condition', False, None, main_register.get_register(Condition)),
        ('sampler', False, None, main_register.get_register(Sampler)),
        variables=[('rounds', 0, int)],
        order=['condition', 'sampler', 'network', 'variables', 'id']
    )

    def __init__(self, condition, sampler, network=None, variables={}, id=None):
        SamplerNetworkSchema.__init__(self, sampler, network, variables, id)
        self.condition = condition

        self.var_rounds, = AlternatingSchema.arguments.validate_and_return_variables(variables)
        print(self.var_rounds)

    def run(self):
        self.sampler.initialize()
        if self.network is not None:
            self.network.initialize()

        msgs = {self.sampler: Message(),
                self.network: Message()}

        while self.condition.satisfied():
            if self.var_rounds is not None:
                self.var_rounds.value += 1

            data = self.sampler.sample(msgs)

            if self.network is not None:
                self.network.train(msgs, data)
                self.network.store()

        self.sampler.finalize()
        if self.network is not None:
            self.network.finalize()
            self.network.store()

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  AlternatingSchema)

main_register.append_register(AlternatingSchema, "alternating", "alt")