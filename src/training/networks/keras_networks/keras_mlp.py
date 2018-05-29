from .keras_network import KerasNetwork, KerasDomainPropertiesNetwork

from ... import parser_tools as parset
from ... import parser
from ... import main_register

from ...misc import similarities

import keras
import numpy as np


class KerasDynamicMLP(KerasDomainPropertiesNetwork):
    arguments = parset.ClassArguments('MLPDynamicKeras',
                                      KerasDomainPropertiesNetwork.arguments,
                                      ("hidden", False, None, int, "Number of hidden layers"),
                                      ("output_units", False, True, int,
                                       "Classification network with output_units output units or regression network if -1"),
                                      ("activation", True, "sigmoid", str, "Activation function of hidden layers"),
                                      ("dropout", True, None, int, "Dropout probability or None if no dropout"),
                                      ("optimizer", True, "adam", str, "Optimization algorithm"),
                                      ("loss", True, "mean_squared_error", str, "Loss function"),
                                      order=["hidden", "output_units",
                                             "activation", "dropout",
                                             "optimizer", "loss",
                                             "load", "store", "formats", "out",
                                             "epochs",
                                             "count_samples", "test_similarity",
                                             "graphdef", "callbacks", "variables", "id"]
                                      )

    def __init__(self, hidden, output_units=-1, activation="sigmoid",
                 dropout=None, optimizer="adam", loss="mean_squared_error",
                 load=None, store=None, formats=None, out=".", epochs=1000,
                 count_samples=False, test_similarity=None, graphdef=None,
                 callbacks=None,
                 variables=None, id=None,
                 domain_properties=None):
        KerasDomainPropertiesNetwork.__init__(
            self, load, store, formats, out, epochs, count_samples,
            test_similarity, graphdef, callbacks, variables, id, domain_properties=domain_properties)
        self._hidden = hidden
        self._output_units = output_units
        self._activation = activation
        self._dropout = dropout
        self._optimizer = optimizer
        self._loss = loss

        self._x_fields_extractor = lambda ds: [ds.field_current_state,
                                               ds.field_goal_state]
        self._y_fields_extractor = None#lambda ds: ds.field_heuristic
        self._x_converter = lambda x: [np.stack(x[:, 0], axis=0),
                                       np.stack(x[:, 1], axis=0)]
        self._x_fields_comparators = [
            similarities.hamming_measure_cmp_iterable_equal,
            similarities.hamming_measure_cmp_iterable_equal
        ]
        self._count_samples_hasher = lambda x, y: hash(str((x, y)))
        # Either self._domain_properties will be used to determine the state
        # size or on initialization the state size has to be given
        # If both is given, the DomainProperties will be prefered
        self._state_size = None

    def _initialize_general(self, *args, **kwargs):
        state_size = kwargs.pop("state_size", None)
        if state_size is not None:
            self._state_size = state_size


    def _initialize_model(self, *args, **kwargs):
        if self._domain_properties is None and self._state_size is None:
            raise ValueError("This network either needs the state size "
                             "information or preferably a DomainProperties"
                             "object,")

        input_units = (len(self._domain_properties.gnd_flexible)
                       if self._domain_properties is not None
                       else self._state_size)
        regression = self._output_units == -1
        output_units = 1 if regression else self._output_units

        in_state = keras.layers.Input(shape=(input_units,))
        in_goal = keras.layers.Input(shape=(input_units,))
        next = keras.layers.concatenate([in_state, in_goal], axis=-1)

        unit_diff = input_units * 2 - output_units
        step = int(unit_diff/(self._hidden + 1))
        units = input_units * 2
        for i in range(self._hidden):
            units -= step
            next = KerasNetwork.next_dense(next, units,
                                           self._activation, self._dropout)
        next = KerasNetwork.next_dense(next, output_units,
                                       "relu" if regression else "softmax", None)


        self._model = keras.Model(inputs=[in_state, in_goal], outputs=next)
        self._compile(self._optimizer, self._loss, ["accuracy",
                                                    "mean_absolute_error"])

    def reinitialize(self,*args, **kwargs):
        if self.path_load is not None:
            self.load(**kwargs)
        else:
            self._initialize_model(*args, **kwargs)

    def _finalize(self, *args, **kwargs):
        pass

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  KerasDynamicMLP)


main_register.append_register(KerasDynamicMLP, "keras_dyn_mlp")
