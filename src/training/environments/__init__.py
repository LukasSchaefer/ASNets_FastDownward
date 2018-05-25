"""
Environments manage the execution of task. You should use this to execute
resource intensive task. Small things can also be run in this tool. The
advantage of environments is that they can (depending on which one you are
using) execute tasks in different threads or on a computer cluster.

The problem is how does every object which wants to get access to the
environment get access. Our current workaround is:
Every object which wants to use environments has to:
    1. be registered (either via id or global name) in the item_cache (we are
       using the item_cache of a run to access all relevant objects)
    1. have an attribute _environment. If it defines its environment
       itself or via its parameter, it shall store the reference in
       _environment. It it wants to get access to the default environment,
       it shall set _environment = None. After the
       parsing all registered objects (all objects  with an id or with a global
       name) are checked for having the attribute _environment=None and if yes,
       the default environment is placed there.
    2. (optionally, but desired) be able to parse them via command line.
       In this case you can predefine an environment and provide it via its id
       to the object and you can use multiple different environments.


"""
from .. import dependencies

if dependencies.environments:
    from .task import Task, DelegationTask, SubprocessTask

    from .base_environment import Environment, NbCoresEnvironment
