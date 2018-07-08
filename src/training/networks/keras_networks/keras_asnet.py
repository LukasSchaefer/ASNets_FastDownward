from .keras_network import KerasNetwork
from .keras_tools import KerasDataGenerator

from ... import parser_tools as parset
from ... import parser
from ... import main_register

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


    def print_data(self, dtrain):
        """
        Check training data and print them all
        """
        dtrain = self._convert_data(dtrain)
        kdg_train = KerasDataGenerator(
            dtrain,
            x_fields=None if self._x_fields_extractor is None else self._x_fields_extractor(dtrain[0]),
            y_fields=None if self._y_fields_extractor is None else self._y_fields_extractor(dtrain[0]),
            x_converter=self._x_converter,
            y_converter=self._y_converter,
            shuffle=True,
            count_diff_samples=self._count_samples_hasher if self._count_samples else None)

        for batch in range(kdg_train.__len__()):
            x, y = kdg_train[batch]
            assert len(x) == 3, "Not 3 arrays in x train data!"
            print()
            print("Batch %d:" % batch)
            for sample_idx in range(len(x[0])):
                print("Sample %d:" % sample_idx)
                print("X:")
                print("Proposition truth values:")
                print(x[0][sample_idx])
                print("Proposition goal values:")
                print(x[1][sample_idx])
                print("Applicable values:")
                print(x[2][sample_idx])
                print("Y:")
                print(y[0][sample_idx])

            print()
            print("Loss for unifrom probs:")
            uniform_probs = None
            for arr in x[2]:
                sum = np.sum(arr)
                p = arr / sum
                if uniform_probs is None:
                    uniform_probs = [p]
                else:
                    uniform_probs = np.append(uniform_probs, [p], axis=0)
                assert np.sum(uniform_probs[-1]) == 1, "Sum of prediction is not 1"
            y = y[0]
            uniform_probs = np.clip(uniform_probs, 1e-8, 1 - 1e-8)
            ones = np.ones(y.shape)
            out = -(y * np.log(uniform_probs) + (ones - y) * np.log(ones - uniform_probs))
            loss = np.sum(out, axis=-1)
            for sample_idx in range(len(uniform_probs)):
                print()
                print("Sample %d:" % sample_idx)
                print("Uniform Action Probabilities (among all appicable actions):")
                print(uniform_probs[sample_idx])
                print("Y:")
                print(y[sample_idx])
                # print("Out:")
                # print(out[sample_idx])
                print("Loss: %d" % loss[sample_idx])


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
