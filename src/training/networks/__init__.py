from .. import DEPENDENCIES
from .base_network import Network, nregister

if DEPENDENCIES.keras and DEPENDENCIES.tensorflow:
    from .keras_network import KerasNetwork