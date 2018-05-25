from .. import dependencies

if dependencies.samplers:

    from .base_sampler import saregister, Sampler
    from .iterable_sampler import IterableFileSampler, DirectorySampler