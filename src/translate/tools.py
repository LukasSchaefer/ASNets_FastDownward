import logging
import sys

def cartesian_product(sequences):
    # TODO: Rename this. It's not good that we have two functions
    # called "product" and "cartesian_product", of which "product"
    # computes cartesian products, while "cartesian_product" does not.

    # This isn't actually a proper cartesian product because we
    # concatenate lists, rather than forming sequences of atomic elements.
    # We could probably also use something like
    # map(itertools.chain, product(*sequences))
    # but that does not produce the same results
    if not sequences:
        yield []
    else:
        temp = list(cartesian_product(sequences[1:]))
        for item in sequences[0]:
            for sequence in temp:
                yield item + sequence


def get_peak_memory_in_kb():
    try:
        # This will only work on Linux systems.
        with open("/proc/self/status") as status_file:
            for line in status_file:
                parts = line.split()
                if parts[0] == "VmPeak:":
                    return int(parts[1])
    except IOError:
        pass
    raise Warning("warning: could not determine peak memory")


class SkipHigherFilter(logging.Filter):
    def __init__(self, name="", level=None):
        logging.Filter.__init__(self, name)
        self.level = level

    def filter(self, record):
        if logging.Filter.filter(self, record):
            return record.levelno <= self.level
        return False


def setup_logger(log=None, level=logging.INFO, handlers=None):
    """
    Set up the given logger.
    :param log: Logger to set up (default root logger)
    :param level: Miminum message level to process (default logger.INFO)
    :param handlers: Handlers to register (default Stdout and Stderr handlers)
    :return: None
    """

    log = logging.root if log is None else log
    log.setLevel(level)
    if handlers is None:
        handlers = []
        if level <= logging.INFO:
            hdlr_out = logging.StreamHandler(sys.stdout)
            hdlr_out.setLevel(logging.NOTSET)
            hdlr_out.addFilter(SkipHigherFilter(level=logging.INFO))
            handlers.append(hdlr_out)

        hdlr_err = logging.StreamHandler(sys.stderr)
        hdlr_err.setLevel(logging.WARNING)
        handlers.append(hdlr_err)

    log.handlers = handlers

