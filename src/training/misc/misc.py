import sys
import abc

if sys.version_info >= (3, 4):
    AbstractBaseClass = abc.ABC
else:
    AbstractBaseClass = abc.ABCMeta('ABC', (object,), {'__slots__': ()})


class InvalidModuleImplementation(Exception):
    pass