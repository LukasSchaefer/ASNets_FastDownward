from .. import dependencies

named_networks = {}

if dependencies.networks:
    from .base_network import Network, nregister, NetworkFormat

    from . import keras_networks