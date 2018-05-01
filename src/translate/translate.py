#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import logging
import traceback
import os

sys.path.insert(0,os.path.dirname(os.path.dirname(__file__)))
from translate import translator

def main(argv=None, log=None):
    translator.main(argv=argv, log=log)
if __name__ == "__main__":
    log = logging.root
    try:
        translator.main(log=log)
    except MemoryError:
        log.critical("Translator ran out of memory, traceback:")
        log.critical("=" * 79)
        log.critical(traceback.format_exc())
        log.critical("=" * 79)
        sys.exit(EXIT_MEMORY_ERROR)
