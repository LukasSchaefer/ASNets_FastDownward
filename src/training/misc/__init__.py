"""
Miscelaneous things of more or less usefullness
"""


from . import canonicalization
from . import data_similarities as similarities
from . import hasher
from .domain_properties import DomainProperties
from .misc import AbstractBaseClass, InvalidModuleImplementation, get_rnd_suffix, RND_SUFFIX_LEN
from . stream_contexts import StreamDefinition, OpenStreamDefinition, GzipStreamDefinition
from .stream_contexts import StreamContext