from .. import dependencies

if dependencies.bridges:

    from .base_bridges import Bridge, SamplerBridge
    from .sampling_bridges import FastDownwardSamplerBridge, LoadSampleBridge

    from .sampling_bridges import StateFormat