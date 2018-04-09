from __future__ import print_function

# -*- coding: utf-8 -*-

import contextlib
import os
import logging
import time


class Timer(object):
    def __init__(self):
        self.start_time = time.time()
        self.start_clock = self._clock()

    def _clock(self):
        times = os.times()
        return times[0] + times[1]

    def __str__(self):
        return "[%.3fs CPU, %.3fs wall-clock]" % (
            self._clock() - self.start_clock,
            time.time() - self.start_time)


@contextlib.contextmanager
def timing(text, block=False, log=logging.root):
    timer = Timer()
    log.info("%s..." % text)
    yield
    log.info("%s: %s" % (text, timer))
