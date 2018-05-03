#!/usr/bin/env python

import abc
import sys

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})


class Abstract(ABC):
    @abc.abstractmethod
    def do(self):
        pass


class Sub(Abstract):
    def do(self):
        print("DONE")
a = Sub()
a.do()