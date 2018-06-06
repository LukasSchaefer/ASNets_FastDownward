from .. import parser
from .. import parser_tools as parset
from .. parser_tools import ArgumentException, main_register

import abc
import sys
import random
import re


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


class MatchRegex(object):
    arguments = parset.ClassArguments("MatchRegex", None,
                                      ("regex", False, None, str, "Regex to match"),
                                      ("invert", True, False, parser.convert_bool, "Invert matching result")
                                      )
    def __init__(self, regex, invert=False):
        self._regex = re.compile(regex)
        self._invert = invert

    def check(self, s):
        r = self._regex.match(s)
        return (not r) if self._invert else r

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  MatchRegex)

main_register.append_register(MatchRegex, "regex")