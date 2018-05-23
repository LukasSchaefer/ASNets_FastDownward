from .. import DEPENDENCIES
from .base_network import Network, nregister, NetworkFormat


named_networks = {}
if DEPENDENCIES.keras and DEPENDENCIES.tensorflow:
    from . import keras_networks
    from .keras_networks import KerasDataGenerator, store_keras_model_as_protobuf
    from .keras_networks import KerasNetwork, KerasDomainPropertiesNetwork, KerasDynamicMLP
