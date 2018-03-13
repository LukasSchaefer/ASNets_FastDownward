from . import Schema, sregister

from .. import main_register
from .. import parser
from .. import parser_tools as parset
from .. import Message


class AlternatingSchema(Schema):
    def __init__(self, condition, sampler, network=None):
        self.condition = condition
        self.sampler = sampler
        self.network = network

    def train(self):
        self.sampler.initialize()
        if self.network is not None:
            self.network.initialize()

        msg_sampler = Message()
        msg_network = Message()
        while self.condition.satisfied():
            data, msg = self.sampler.sample(msg_sampler, msg_network)
            #preprocess data

            if network is not None:
                msg = self.network.train(msg_sampler, msg_network, data)
                self.network.store()

        self.sampler.finalize()
        if self.network is not None:
            self.network.finalize()


main_register.append_register(AlternatingSchema, "alternating", "alt")