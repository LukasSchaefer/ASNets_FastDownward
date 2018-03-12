import abc
from future.utils import with_metaclass

class Schema(with_metaclass(abc.ABCMeta, object)):

    @abc.abstractmethod
    def train(self):
        pass