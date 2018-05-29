"""Remember to register your schemas via 'append_register' of the main package.
"""
from .. import dependencies

if dependencies.training_schemas:
    from .base_schema import Schema
    from .base_schema import SamplerNetworkSchema
    from .base_schema import scregister

    from .alternating_schema import AlternatingSchema




