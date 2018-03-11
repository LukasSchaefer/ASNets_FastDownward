import abc
from future.utils import with_metaclass

class Network(with_metaclass(abc.ABCMeta, object)):
    """
    Base class for all neural networks.
    Do not forget to register your network subclass in this packages 'register'
    dictionary via 'append_register' of the main package.
    """
    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluate(self):
        pass

    @abc.abstractmethod
    def store(self):
        pass
