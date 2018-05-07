from .. import DEPENDENCIES
from .base_network import Network, nregister


named_networks = {}
if DEPENDENCIES.keras and DEPENDENCIES.tensorflow:
    from .keras_network import KerasNetwork, MLPDynamicKeras
    from .keras_tools import KerasDataGenerator