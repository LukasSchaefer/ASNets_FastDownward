"""
Configure which dependencies are available in your environment. Load this
module first and modify the dependencies as required, before loading any other
module of this framework. Otherwise your dependency changes will be ignored.
Some modules are only loaded if their dependencies are met. Some will fail if
they shall be loaded, but their depenencies are not met.
Each module/subpackage has to check itself for dependencies and if it shall load

There are two types of dependencies currently:
1. Dependencies within the framework: Please sort them in their directory structure
2. Dependencies to modules outside of this framework.

THIS IS NOT FOR CHECKING ON EVERY MODULE THAT THE MODULE IT LOADS ARE PRESENT.
PYTHON WILL HANDLE MISSING MODULES. THIS IS WHEN YOU DO NOT WANT TO LOAD SOME
FEATURES IN YOUR CODE. FOR EXAMPLE, SOME MODULES REQUIRE EXTERNAL LIBRARIES
(LIKE TENSORFLOW) WHICH ARE NOT EVERYWHERE PRESENT AND NOT EVERY SCRIPT NEEDS
THE MODULES WHICH REQUIRE THOSE EXTERNAL LIBRARIES (e.g. keras_networks require
keras and tensorflow, whereas, if I want to use only fast-sample, i have no need
of install those dependencies)
"""
import sys

print("LOAAAAAAAAAAAAAAAAAAAAAAAAAAADDDDD")

# If this module is loaded such that it is already registered within sys.modules,
# then the dependency flags will be directly registered. If the module is not
# registered in sys.modules, then 'setup()' has to be manually called after it
# was registered.
do_setup = __name__ in sys.modules
done_setup = False

this = None  # Reference to this module
ALL = {}  # {Name : Description}
REQ = {}  # {Name : Other dependencies which require this}


def add(name, value=True, sub=None, requires=None, external=False, description=None):
    """
    Adds a new dependency
    :param name: name of the dependency
    :param value: Value of the dependency (most times True or False)
    :param sub: subordinate dependency names (they have toz be created separatly)
    :param requires: iterable of dependencies required for this dependency
    :param external: dependency to external library
    :param description: Description of the flag
    :return:
    """
    if name in ALL:
        raise ValueError("Dependency exists already: " + str(name))
    ALL[name] = (sub, requires, external, description)
    if requires is not None:
        for dep in requires:
            if dep not in REQ:
                REQ[dep] = set()
            REQ[dep].add(name)

    setattr(this, name, value)


def set_flag(name, value):
    if name not in ALL:
        raise ValueError("Unknown dependency: " + name)
    setattr(this, name, value)


def set_tree(name, value):
    set_flag(name, value)
    subs = [] if ALL[name][0] is None else ALL[name][0]
    for sub in subs:
        set_tree(sub, value)


def set_reqr(name, value):
    set_flag(name, value)
    if name in REQ:
        for dep in REQ[name]:
            set_reqr(dep, value)


def set_all(value):
    for name in ALL:
        setattr(this, name, value)


def set_external(value, cascade_required=False):
    for name in ALL:
        external = ALL[name][2]
        if external:
            if cascade_required:
                set_reqr(name, value)
            else:
                set_flag(name, value)


def setup():
    global this, done_setup
    if done_setup:
        return
    else:
        done_setup = True
    print("SETIPP")
    this = sys.modules[__name__]
    """External Dependencies"""
    add("keras", True, external=True)
    add("tensorflow", True, external=True)


    """Internal Dependencies"""
    add("bridges",True)

    add("conditions", True)

    add("environments", True)

    # Disabling misc makes no sense, because by design misc contains modules
    # which cannot be grouped otherwise. Only disabling Them one by one should
    # be allowed
    #--a-d-d-(-"-m-i-s-c-"-,- -T-r-u-e-)--

    add("networks", True)
    add("keras_networks", True, requires=["keras", "tensorflow"])

    add("problem_sorter", True)

    add("samplers", True)

    add("training_schemas", True)

    undefined_required_dependencies = ""
    for dep in REQ:
        if dep not in ALL:
            undefined_required_dependencies += str(dep) + ", "
    if undefined_required_dependencies != "":
        undefined_required_dependencies = undefined_required_dependencies[:-2]
        print("Dependencies where defined as required by other dependencies,"
              "although they are unknown: " + undefined_required_dependencies)

if do_setup:
    setup()
"""Dependency Checking"""


class FailedDependency(Exception):
    pass


class DependencyChecker():
    def __init__(self, name, required, optional=None, optional_flags=None,
                 verbosity=0):
        """
        Checks Dependencies
        :param name: Name of the tool for which dependencies are managed
        :param required: Iterable of dependencies definitely required
        :param optional: Iterable of optional dependencies
        :param optional_flags: {Name : Iterable of dependencies for this name}
        :param verbosity: 0 = no auxiliar output, 1 = unmet dependencies,
                          --2-=-all-dependencies-status--
        """
        self._name = str(name)
        self._required = required
        self._optional = set() if optional is None else set(optional)
        self._optional_flags = {} if optional_flags is None else optional_flags
        self._optional_flag_values = {}
        for flag in self._optional_flags:
            for dep in self._optional_flags[flag]:
                self._optional.add(dep)

        self._verbosity = verbosity

    def _check_dependency(self, dependency):
        if not hasattr(this, dependency):
            raise ValueError("The dependency flag requested by "
                             + self._name + " does not exist: " + str(dependency))
        return getattr(this, dependency)

    def _check_flag(self, flag):
        valid = True
        for dep in self._optional_flags[flag]:
            if not self._check_dependency(dep):
                valid = False
                break
        self._optional_flag_values[flag] = valid

    def check_dependencies(self):
        unmet = ""
        for dependency in self._required:
            if not self._check_dependency(dependency):
                unmet += str(dependency) + ", "
        if unmet != "":
            unmet = unmet[:-2]
            raise FailedDependency("One or multiple required dependencies of "
            + self._name +" are not met: " + unmet)

        unmet_optional = ""
        for dep in self._optional:
            if not self._check_dependency(dep):
                unmet_optional += str(dep) + ", "
        if unmet_optional != "" and self._verbosity > 0:
            unmet_optional = unmet_optional[:-2]
            print("Unmet optional dependencies of " + self._name + ": "
                  + str(unmet_optional))

        for flag in self._optional_flags:
            self._check_flag(flag)

        if self._verbosity > 0:
            print("Unmet flags: ")
            for flag in self._optional_flag_values:
                if not self._optional_flag_values[flag]:
                    print(flag)

    def valid(self, flag):
        if flag not in self._optional_flags:
            raise ValueError("Unknown flag for " + self._name + ": " + str(flag))

        if flag not in self._optional_flag_values:
            self._check_flag(flag)

        return self._optional_flag_values[flag]




