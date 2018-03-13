from ..parser_tools import main_register

import abc
from future.utils import with_metaclass


class Schema(with_metaclass(abc.ABCMeta, object)):

    @abc.abstractmethod
    def train(self):
        pass

main_register.append_register(Schema, "schema")
sregister = main_register.get_register(Schema)