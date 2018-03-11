from .base_mutators import Mutator



class MGroup(Mutator):
    def __init__(self, mutators, condition_mutate=None,
                 condition_signal=None, condition_reset=None):
        Mutator.__init__(self, condition_mutate, condition_signal,
                         condition_reset)
        self.mutators = mutators

    def _mutate(self):
        for m in self._mutators:
            m.next()

    def _reset(self):
        for m in self._mutate():
            m._reset()


class MRoundRobin(Mutator):
    def __init__(self, mutators, condition_mutate=None,
                 condition_signal=None, condition_reset=None):
        Mutator.__init__(self, condition_mutate, condition_signal,
                         condition_reset)
        self._mutators = mutators
        self._next_mutator = 0

    def _mutate(self):
        self._mutators[self._next_mutator].next()
        self._next_mutator += 1

    def _reset(self):
        self._next_mutator = 0
        for m in self._mutate():
            m._reset()


class MLeft2Right(Mutator):
    def __init__(self, mutators, condition_mutate=None,
                 condition_signal=None, condition_reset=None):
        Mutator.__init__(self, condition_mutate, condition_signal,
                         condition_reset)
        self._mutators = mutators
        self._next_mutator = 0

    def _mutate(self):
        for m in self._mutators:
            sig = m.next()
            if not sig:
                break

    def _reset(self):
        self._next_mutator = 0
        for m in self._mutate():
            m._reset()
