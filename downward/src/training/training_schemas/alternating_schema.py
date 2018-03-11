from . import register

from .. import append_register
from .. import Message

def train(condition, sampler, network=None):

    sampler.initialize()
    if network is not None:
        network.initialize()

    msg_sampler = Message()
    msg_network = Message()
    while condition.satisfied():
        data, msg = sampler.sample(msg_sampler, msg_network)
        #preprocess data

        if network is not None:
            msg = network.train(msg_sampler, msg_network, data)
            network.store()

    sampler.finalize()
    network.finalize()

append_register(register, train, "alternating", "alt")