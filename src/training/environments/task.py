import abc
import subprocess


class InvalidMethodCallException(Exception):
    pass

class Task(object):
    UNINITIALIZED = -3
    INITIALIZED   = -2
    RUNNING       = -1

    def __init__(self, name, *alarms, **kwargs):
        """
        Base Task
        :param name: name of the task
        :param alarms: SEQUENCE of threading events to wake up on status changes
        :param alarm_set: KWARGUMENT. OPTIONAL
                          set of threading events to wake up on status changes
                          (similar to alarms, but this time a set is expected)
        """
        self.alarms = kwargs.pop("alarm_set", set())
        if len(kwargs) > 0:
            raise ValueError("Unknown arguments:" + str(kwargs))

        self._name = name
        self._status = Task.UNINITIALIZED
        self._result = None
        for alarm in alarms:
            self.add_alarm(alarm)

    def _get_name(self):
        return self._name

    def _get_status(self):
        return self._status

    def _get_result(self):
        return self._result

    name = property(_get_name)
    status = property(_get_status)
    result = property(_get_result)

    def add_alarm(self, event):
        self.alarms.add(event)

    def del_alarm(self, event):
        self.alarms.remove(event)

    def reset(self):
        self._status = Task.UNINITIALIZED
        self._result = None

    def _change_status(self, new):
        self._status = new
        if self._status >= 0:
            for alarm in self.alarms:
                alarm.set()

    @abc.abstractmethod
    def _initialize(self):
        pass

    def initialize(self):
        if not self._status == Task.UNINITIALIZED:
            raise InvalidMethodCallException("Cannot initialize an already initialized task: ", self._name)
        else:
            self._initialize()
            self._change_status(Task.INITIALIZED)

    @abc.abstractmethod
    def _execute(self):
        """
        Should set self._result if it has produced results
        :return: status code of executed task (0 = OK, 1+ = Error)
        """
        pass

    def execute(self):
        if not self._status == Task.INITIALIZED:
            raise InvalidMethodCallException("Cannot execute an uninitialized or already executed task: ", self._name)
        else:
            self._change_status(Task.RUNNING)
            self._change_status(self._execute())

    def run(self):
        if self._status == Task.UNINITIALIZED:
            self.initialize()

        if self._status == Task.INITIALIZED:
            self.execute()


class DelegationTask(Task):
    def __init__(self, name, func_initialize, func_execute, *alarms, **kwargs):
        """
        DelegationTask.
        :param name:
        :param alarms: SEQUENCE of threading events to wake up on status changes
        :param alarm_set: KWARGUMENT. OPTIONAL
                          set of threading events to wake up on status changes
                          (similar to alarms, but this time a set is expected)
        """
        Task.__init__(self, name, *alarms, **kwargs)
        self._func_initialize = func_initialize
        self._func_execute = func_execute

    def _initialize(self):
        self._func_initialize(self)

    def _execute(self):
        return self._func_execute(self)


class SubprocessTask(Task):
    """
    DelegationTask.
    :param name:
    :param alarms: SEQUENCE of threading events to wake up on status changes
    :param alarm_set: KWARGUMENT. OPTIONAL
                      set of threading events to wake up on status changes
                      (similar to alarms, but this time a set is expected)
    """

    def __init__(self, name, command, *alarms, **kwargs):

        Task.__init__(self, name, *alarms, **kwargs)
        self._command = command

    def _initialize(self):
        pass

    def _execute(self):
        x =  subprocess.call(self._command, shell=False)
        print(self._command)
        return x

