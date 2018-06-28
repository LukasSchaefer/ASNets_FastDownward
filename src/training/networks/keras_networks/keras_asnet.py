from .keras_network import KerasNetwork, KerasNetwork

from ... import parser_tools as parset
from ... import parser
from ... import main_register

import keras
import numpy as np


class KerasASNet(KerasNetwork):
    arguments = parset.ClassArguments('ASNetKeras',
                                      KerasNetwork.arguments,
                                      ("extra_input_size", False, 0, int, "Size of additional input features per action"),
                                      ("model", True, None, None, "Keras model for ASNet"),
                                      order=["extra_input_size", "model", "load", "store",
                                             "formats", "out", "epochs", "count_samples",
                                             "test_similarity", "graphdef", "callbacks",
                                             "variables", "id"]
                                      )

    def __init__(self, extra_input_size=0, model=None, load=None, store=None, formats=None,
                 out=".", epochs=1000, count_samples=False, test_similarity=None,
                 graphdef=None, callbacks=None, variables=None, id=None):
        KerasNetwork.__init__(
            self, load, store, formats, out, epochs, count_samples,
            test_similarity, graphdef, callbacks, variables, id)
        self._extra_input_size = extra_input_size
        self._model = model
        if self._model is None:
            raise ValueError("No ASNet model given to KerasASNet.")

        # entries in ds data is in following format:
        # [<PROBLEM_HASH>, <GOAL_VALUES>, <STATE_VALUES>, <APPLICABLE_VALUES>,
        #  <OPT_VALUES>(, <ADDITIONAL_INPUT_FEATURES>)]
        # input features for network:
        # [<STATE_VALUES>, <GOAL_VALUES>, <APPLICABLE_VALUES>(, <ADDITIONAL_INPUT_FEATURES>)]
        # y values for loss computation: [<OPT_VALUES>]

        # TODO check if indeces here are correct (assumed based on usage)
        if self._extra_input_size > 0:
            self._x_fields_extractor = lambda ds: [2, 1, 3, 5]
        else:
            self._x_fields_extractor = lambda ds: [2, 1, 3]

        self._y_fields_extractor = lambda ds: [4]

        if self._extra_input_size > 0:
            self._x_converter = lambda x: [np.stack(x[:, 0], axis=0),
                                           np.stack(x[:, 1], axis=0),
                                           np.stack(x[:, 2], axis=0),
                                           np.stack(x[:, 3], axis=0)]
        else:
            self._x_converter = lambda x: [np.stack(x[:, 0], axis=0),
                                           np.stack(x[:, 1], axis=0),
                                           np.stack(x[:, 2], axis=0)]

        self._y_converter = lambda y: [np.stack(y[:,0], axis=0)]

        self._count_samples_hasher = lambda x, y: hash(str((x, y)))


    def _initialize_general(self, *args, **kwargs):
        pass

    def _initialize_model(self, *args, **kwargs):
        pass

    def reinitialize(self,*args, **kwargs):
        pass

    def _finalize(self, *args, **kwargs):
        pass


    """----------------------DATA PARSING METHODS----------------------------"""

    def _convert_data(self, data):
        """
        The given data is first converted into the format needed for this
        network and then the SizeBatchData objects are finalized.

        :param data: List of SizeBatchData
        :return: Converted list of finalized SizeBatchData
        """
        data = data if isinstance(data, list) else [data]
        for data_set in data:
            if data_set.is_finalized:
                print("Warning: Data set previously finalized. Skipping now.")
                continue

            # convert lists from input to numpy arrays for network input
            # (all but first hash entry are lists)
            for type in data_set.data:
                for batch in data_set.data[type]:
                    for entry in batch:
                        for idx_input in range(1, len(entry)):
                            entry[idx_input] = np.array(entry[idx_input], dtype=np.int32)

            data_set.finalize()
        return data

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  KerasASNet)


main_register.append_register(KerasASNet, "keras_asnet")