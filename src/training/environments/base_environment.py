from .task import Task

from .. import parser
from .. import parser_tools as parset

from ..parser_tools import main_register

import multiprocessing
import threading

try:
    import queue
except ImportError:
    import Queue as queue


class NoFreeSlotError(Exception):
    pass


class Environment(object):
    """
    Base class for all environments.
    Do not forget to register your environment subclass.
    """

    arguments = parset.ClassArguments("Environment", None,
                                      ("max_active", True, 1, int),
                                      ('id', True, None, str),
                                      )

    def __init__(self, max_active=1, id=None, clock=None):
        self._clock = threading.Event() if clock is None else clock
        self._lock = threading.RLock() #Improve later the locking if needed

        self._started = False
        self._idle = True

        self._queue = []
        self._done = set()
        self._max_active = max_active
        self._active = set()

        self.id = id

    def queue_empty(self):
        return len(self._queue) == 0

    def queue_count(self, task):
        with self._lock:
            c = 0
            for item in self._queue:
                if item == task:
                    c += 1
            return c

    def queue_push(self, task, count=1):
        with self._lock:
            for i in range(count):
                self._queue.append(task)

    def queue_remove(self, task, count=1):
        with self._lock:
            idx = 0
            while idx < len(self._queue):
                if self._queue[idx] == task:
                    del self._queue[idx]
                    idx -= 1
                    count -= 1
                    if count == 0:
                        return True
        return False

    def queue_clear(self):
        with self._lock:
            self._queue = []

    def _queue_pop(self):
        with self._lock:
            if self.queue_empty():
                return None
            obj = self._queue[0]
            del self._queue[0]
            return obj

    def _done_add(self, task):
        with self._lock:
            self._done.add(task)

    def done_has(self, task):
        return task in self._done

    def active_size(self):
        return len(self._active)

    def _active_add(self, task):
        with self._lock:
            self._active.add(task)

    def _start_next_task(self):
        with self._lock:
            if self.active_size() >= self._max_active:
                raise NoFreeSlotError("No task slot available for the task "
                                      "to start.")
            obj = self.queue_pop()
            obj.add_alarm(self._clock)
            thread = threading.Thread(target=obj.run)
            self._active_add(obj)
            thread.start()



    def _run(self):
        while self._started:
            self._clock.wait()

            while self._clock.is_set():
                self._clock.clear()

                with self._lock:
                    freshly_done = set()
                    for task in self._active:
                        #Task has finished
                        if task.status >= 0:
                            freshly_done.add(task)
                    for task in freshly_done:
                        task.del_alarm(self._clock)
                        self._done_add(task)
                        self._active.remove(task)

                    while (not self.queue_empty()
                           and self.active_size() < self._max_active):
                        self._start_next_task()



    def start(self):
        with self._lock:
            if not self._started:
                t = threading.Thread(target=self._run)
                t.start()
                self._started = True

    # TODO def stop(self, wait=False):

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache, Environment)


main_register.append_register(Environment, "environment", "env")
envregister = main_register.get_register(Environment)


class NbCoresEnvironment(Environment):

    arguments = parset.ClassArguments("NbCoresEnvironment", None,
                                      ('id', True, None, str),
                                      )

    def __init__(self, id=None, clock=None):
        Environment.__init__(max_active=multiprocessing.cpu_count(),
                             id=id, clock=clock)

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  NbCoresEnvironment)


main_register.append_register(NbCoresEnvironment, "cores_env")
