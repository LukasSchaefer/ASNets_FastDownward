import abc
import sys
import random

if sys.version_info >= (3, 4):
    AbstractBaseClass = abc.ABC
else:
    AbstractBaseClass = abc.ABCMeta('ABC', (object,), {'__slots__': ()})


class InvalidModuleImplementation(Exception):
    pass

# If using random numbers to distinguish files which might collide, use numbers
# of this length
RND_SUFFIX_LEN = 10
def get_rnd_suffix(size=RND_SUFFIX_LEN):
    FMT = "%0"+str(size) + "d"
    return FMT % int(random.random() * (10**size))
