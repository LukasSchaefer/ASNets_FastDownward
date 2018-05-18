from . import Network, NetworkFormat
from .keras_tools import KerasDataGenerator

from .. import parser_tools as parset
from .. import parser
from .. import main_register
from ..misc import similarities, InvalidModuleImplementation
from ..parser_tools import ArgumentException

from ..bridges import StateFormat

import json
import keras
from keras import layers
import keras.backend as K
import matplotlib as mpl
import math
mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

# List of formats in which the matplotlib figures shall be stored
MATPLOTLIB_OUTPUT_FORMATS=["png"]


class KerasNetwork(Network):
    arguments = parset.ClassArguments('KerasNetwork', Network.arguments,
        ("count_samples", True, False, parser.convert_bool,
         "Counts how many and which samples where used during training"
         " (This increases the runtime)."),
        ("test_similarity", True, None, similarities.get_similarity,
         "Estimates for every sample in the evaluation data how close it is"
         " to the trainings data. For this to work ALL training data is kept in"
         "memory, this could need a lot of memory"
         " (provide name of a similarity measure)"),
        order=["load", "store", "formats", "out", "count_samples",
               "test_similarity", "variables", "id"])

    def __init__(self, load=None, store=None, formats=None, out=".",
                 count_samples=False, test_similarity=None,
                 variables=None, id=None):
        Network.__init__(self, load, store, formats, out, variables, id)
        self._epochs = 20
        self._model = None


        """Analysis data"""
        self._history = None
        self._histories = []
        self._evaluation = None
        self._evaluations = []
        self._count_samples = count_samples
        self._count_samples_hashes = set()  # Used if _count_samples is True
        test_similarity = similarities.get_similarity(test_similarity) if isinstance(test_similarity, str) else test_similarity
        self._test_similarity = test_similarity
        self._training_data = set()  # If _test_similarity is not None, add training data

        """Concrete networks might need to set those values"""
        """For examples of the following values check existing implementations"""
        self._x_fields_extractor = None  # callable which gets a SampleBatchData
        self._y_fields_extractor = None  # object and returns an ordered list of
                                        # fields to use for the X/Y data
        self._x_converter = None  # Callable which converts extracted batch of
                                 # X fields to the format needed by the network
        self._y_converter = None  # Similar to x_converter

        # Needed if _count_samples is True. Defines how to calculate a hash from
        # a sample given as x and y value (where X and Y are the fields extracted
        # by the KerasDataGenerator). Example: lambda x,y: str((x,y))
        self._count_samples_hasher =None
        # Needed for test_similarity (if test_similarity is never used, this is
        # never used). Define for every field extracted from _x_fields_extractor
        # how the this field of two instances shall be compared. Take a look at
        # misc.data_similarities. E.g. provide for every field a callable or for
        # example the Hamming Measure has already two comparators defined which
        # can be used via providing their string name ("equal", "iterable")
        # Providing here None means the measure uses its default comparator for
        # all fields
        self._x_fields_comparators = None
        # Callback functions for the keras fitting
        self._callbacks = None

    def initialize(self, *args, **kwargs):
        Network.initialize(self, *args, **kwargs)
        # Check for a valid initialization of all required fields
        if self._count_samples:
            if self._count_samples_hasher is None:
                raise InvalidModuleImplementation(
                    "If allowing the 'count_samples' option in a KerasNetwork, "
                    "the network implementation has to set the parameter"
                    "'_count_samples_hasher'")

    def _compile(self, optimizer, loss, metrics):
        self._model.compile(optimizer=optimizer,
                            loss=loss,
                            metrics=metrics)

    def _get_default_network_format(self):
        return NetworkFormat.hdf5

    def _get_store_formats(self):
        return set([NetworkFormat.hdf5, NetworkFormat.protobuf])

    def _get_load_formats(self):
        return set([NetworkFormat.hdf5])

    def _load(self, path, format):
        if format == NetworkFormat.hdf5:
            self._model = keras.models.load_model(path)
        else:
            raise ValueError("Keras cannot load a network model from the "
                             "format: " + str(format))

    def _store(self, path, formats):
        for format in formats:
            if format is None:
                format = self._get_default_network_format()

            if format == NetworkFormat.hdf5:
                print("STPRE HDF5")
                self._model.save(path + "." + format.suffix)
            elif format == NetworkFormat.protobuf:
                print("STORE PB")
                print("TODO")
                # TODO


    def train(self, dtrain, dtest=None):
        """
        The given data is first converted into the format needed for this
        network and then the SampleBatchData objects are finalized. If your
        KerasNetwork subclass needs a different conversion than the default
        given by this class, define in your subclass a staticmethod
        _convert_data(DATA).
        :param dtrain: List of SampleBatchData for training
        :param dtest: List of SampleBatchData for testing
        :return:
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

        kdg_test = None
        if dtest is not None:
            dtest = self._convert_data(dtest)
            kdg_test = KerasDataGenerator(
                dtest,
                x_fields=None if self._x_fields_extractor is None else self._x_fields_extractor(dtest[0]),
                y_fields=None if self._y_fields_extractor is None else self._y_fields_extractor(dtest[0]),
                x_converter=self._x_converter,
                y_converter=self._y_converter,
                shuffle=True)

        history = self._model.fit_generator(
            kdg_train,
            epochs=self._epochs,
            verbose=1, callbacks=self._callbacks,
            validation_data=kdg_test,
            validation_steps=None, class_weight=None,
            max_queue_size=10, workers=1,
            use_multiprocessing=False,
            shuffle=True, initial_epoch=0)

        c = 0
        for ds in dtrain:
            c += ds.size()
        print("SAMPLE COUNT TOTAL", c)
        self._count_samples_hashes.update(kdg_train.generated_sample_hashes)

        self._history = history
        self._histories.append(history)
        if self._test_similarity is not None:
            self._training_data.update(set(dtrain))
        return history

    def evaluate(self, data):
        data = self._convert_data(data)
        # List in which the original y values for the predications will be added
        y_labels = []
        # Triple for comparing the similarities between a sample to predict and
        # the used training data or None if no similarity shall be measured
        sample_similarities = None
        if self._test_similarity is not None:
            for data_set_example in self._training_data: break
            wrapped_set_similarity = similarities.get_wrapper_similarity_on_set(
                self._test_similarity,
                None if self._x_fields_extractor is None else self._x_fields_extractor(data_set_example),
                self._x_fields_comparators,
                merge=max, init_measure_value=0, early_stopping=lambda x: x==1
            )
            sample_similarities = ([], self._training_data, wrapped_set_similarity)

        kdg_eval = KerasDataGenerator(
            data,
            x_fields=None if self._x_fields_extractor is None else self._x_fields_extractor(data[0]),
            y_fields=None if self._y_fields_extractor is None else self._y_fields_extractor(data[0]),
            x_converter=self._x_converter,
            y_converter=self._y_converter,
            y_remember=y_labels,
            similarity=sample_similarities)

        result = self._model.predict_generator(
            kdg_eval, max_queue_size=10, workers=1, use_multiprocessing=False)
        y_labels = np.concatenate(y_labels)
        if sample_similarities is not None:
            sample_similarities = np.concatenate(sample_similarities[0])

        result = (result.squeeze(axis=1), y_labels, sample_similarities)
        self._evaluation = result
        self._evaluations.append(result)
        return result

    """----------------------DATA PARSING METHODS----------------------------"""

    def _convert_state(self, state, format):
        """
        TODO: Add conversion for more formats, esp. StateFormat.Objects
        :param state: state description
        :param format: StateFormat in which the state is given
        :return:
        """
        if format != StateFormat.Full:
            raise ValueError(
                "Unable to convert data of the given format into the"
                "internal representation:" + str(format))
        parts = state.split("\t")
        for idx in range(len(parts)):
            parts[idx] = 0 if parts[idx][-1] == "-" else 1
        return np.array(parts)

    def _convert_data(self, data):
        """
        The given data is first converted into the format needed for this
        network and then the SampleBatchData objects are finalized. If your
        KerasNetwork subclass needs a different conversion than the default
        given by this class, define in your subclass a staticmethod
        _convert_data(DATA).
        :param data:
        :return:
        """
        print("KERAS CONVERT")
        data = data if isinstance(data, list) else [data]
        for data_set in data:
            if data_set.is_finalized:
                print("Warning: Data set previously finalized. Skipping now.")
                continue
            idx_states = [data_set.field_current_state,
                          data_set.field_goal_state,
                          data_set.field_other_state]
            idx_states = [idx for idx in idx_states if idx is not None]

            for type in data_set.data:
                for batch in data_set.data[type]:
                    for entry in batch:
                        for idx_state in idx_states:
                            if idx_state is None:
                                continue
                            entry[idx_state] = self._convert_state(
                                entry[idx_state],
                                data_set.field_descriptions[idx_state])
            data_set.finalize()
        return data

    """-------------------------ANALYSE PREDICATIONS-------------------------"""

    def _analyse(self, directory):
        KerasNetwork._analyse_from_history_plot(
            self._history, ['acc', 'val_acc'], "Model Accuracy", "accuracy",
            "epoch", ['train', 'test'], "evolution_accuracy", directory)

        KerasNetwork._analyse_from_history_plot(
            self._history, ['loss', 'val_loss'], "Model Loss", "loss", "epoch",
            ['train', 'val'], "evolution_loss", directory)

        KerasNetwork._analyse_from_predictions_scatter(
            self._evaluation[0],  self._evaluation[1], "Predictions",
            "original h", "predicted h", "predictions_scatter", directory)

        KerasNetwork._analyse_from_predictions_scatter_tiles(
            self._evaluation[0], self._evaluation[1],
            "Prediction Probabilities with resp. to the correct prediction",
            "original h", "predicted h", "predictions_tiles", directory)

        KerasNetwork._analyse_from_predictions_deviation(
            self._evaluation[0], self._evaluation[1], "Prediction Deviations",
            "deviation", "count", "deviations", directory)

        KerasNetwork._analyse_from_predictions_deviation_dep_on_h(
            self._evaluation[0], self._evaluation[1],
            "Prediction Deviations depending on original", "deviation",
            "original", "deviations_dep_h", directory)

        KerasNetwork._analyse_from_predictions_deviation_dep_on_similarity(
            self._evaluation[0], self._evaluation[1], self._evaluation[2],
            "Prediction Deviations depending on the similarity to training samples",
            "deviation",
            "similarity", "deviations_dep_sim", directory)

        analysis_data = {"histories": [h.history for h in self._histories],
                         "evaluations": [(e[0].tolist(), e[1].tolist()) for e in self._evaluations]}

        with open(os.path.join(directory, "analysis.meta"), "w") as f:
            json.dump(analysis_data, f)
        print("SEEN SAMPLES", len(self._count_samples_hashes))

    """----------------------ANALYSIS METHODS--------------------------------"""

    @staticmethod
    def _analyse_from_history_plot(history, measures, title, ylabel, xlabel,
                                   legend, file, directory=".",
                                   formats=MATPLOTLIB_OUTPUT_FORMATS):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for m in measures:
            ax.plot(history.history[m])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.legend(legend, loc='upper left')
        fig.tight_layout()
        for format in formats:
            fig.savefig(os.path.join(directory, file + "." + format))
        plt.close(fig)

    @staticmethod
    def _analyse_from_predictions_scatter(predicted, original, title,
                                          pred_label, orig_label, file,
                                          directory=".",
                                          formats=MATPLOTLIB_OUTPUT_FORMATS):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(original, predicted, s=80, c='maroon', alpha=0.1)

        unique = np.unique(predicted)
        if len(unique) == 1:
            yticks = ax.get_yticks().tolist()
            pivot = int(len(yticks)/2)
            mid = yticks[pivot]
            yticks = ["" for _ in yticks]
            yticks[pivot] = mid
            ax.set_yticklabels(yticks)

        ax.set_xlabel(orig_label)
        ax.set_ylabel(pred_label)
        ax.set_title(title)
        fig.tight_layout()
        for format in formats:
            fig.savefig(os.path.join(directory, file + "." + format))
        plt.close(fig)

    @staticmethod
    def _analyse_from_predictions_scatter_tiles(predicted, original, title,
                                                pred_label, orig_label, file,
                                                directory=".",
                                                formats=MATPLOTLIB_OUTPUT_FORMATS):
        max_o_h = max(original)
        min_o_h = min(original)
        max_p_h = math.ceil(max(predicted))
        min_p_h = math.floor(min(predicted))

        by_h = {}
        for idx in range(len(predicted)):
            h = original[idx]
            p = predicted[idx]
            if h not in by_h:
                by_h[h] = []
            by_h[h].append(p)

        EXPONENT = 1
        POWER = 2**EXPONENT
        h_p_bins = []
        for i in range(min_p_h*POWER, max_p_h*POWER + 1):
            h_p_bins.append(float(i)/POWER)

        tiles = np.ndarray(shape=(max_o_h - min_o_h + 1, len(h_p_bins)), dtype=float)
        for h_o in range(min_o_h, max_o_h + 1):
            if h_o not in by_h:
                for idx_p in range(len(h_p_bins)):
                    tiles[h_o - min_o_h, idx_p] = float("NaN")
            else:
                ary = np.around(np.array(by_h[h_o]) * POWER) / POWER
                count = float(len(ary))
                unique, counts = np.unique(ary, return_counts=True)
                # Otherwise they are numpy values and do not hash as needed
                unique = [float(i) for i in unique]
                occurrences = dict(zip(unique, counts))
                for idx_p in range(len(h_p_bins)):
                    h_p = h_p_bins[idx_p]
                    if h_p not in occurrences:
                        tiles[h_o - min_o_h, idx_p] = 0
                    else:
                        tiles[h_o - min_o_h, idx_p] = occurrences[h_p] / count

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        handle = ax.matshow(tiles, cmap="jet",
                           aspect=tiles.shape[1] / float(tiles.shape[0]))
        ax.set_xlabel(orig_label)
        ax.set_ylabel(pred_label)
        ax.set_title(title)
        cax = fig.add_axes([0.9, 0.1, 0.02, 0.8])
        fig.colorbar(handle, cax, orientation='vertical')
        xticks = ax.get_xticks().tolist()
        ax.set_xticklabels([''] + [h_p_bins[int(i)] for i in xticks[1:-1]] + [''])
        fig.tight_layout()
        for format in formats:
            fig.savefig(os.path.join(directory, file + "." + format))
        plt.close(fig)

    @staticmethod
    def _analyse_from_predictions_deviation(predicted, original, title,
                                            pred_label, orig_label, file,
                                            directory=".",
                                            formats=MATPLOTLIB_OUTPUT_FORMATS):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        dev = predicted - original
        dev = np.round(dev.astype(np.float))
        unique, counts = np.unique(dev, return_counts=True)
        # Otherwise they are numpy values and do not hash as needed
        unique = [float(i) for i in unique]
        occurrences = dict(zip(unique, counts))
        min_d, max_d = min(unique), max(unique)

        bars = np.arange(min_d, max_d + 1)
        heights = [0 if not i in occurrences else occurrences[i]
                   for i in range(math.floor(min_d), math.ceil(max_d) + 1)]

        ax.bar(bars, heights, align='center')
        ax.set_xlabel(orig_label)
        ax.set_ylabel(pred_label)
        ax.set_title(title)
        fig.tight_layout()
        for format in formats:
            fig.savefig(os.path.join(directory, file + "." + format))
        plt.close(fig)

    @staticmethod
    def _analyse_from_predictions_deviation_dep_on_h(predicted, original, title,
                                                     pred_label, orig_label,
                                                     file, directory=".",
                                                     formats=MATPLOTLIB_OUTPUT_FORMATS):
        by_h = {}
        min_h = min(original)
        max_h = max(original)
        for idx in range(len(original)):
            h = original[idx]
            p = predicted[idx]
            if h not in by_h:
                by_h[h] = []
            by_h[h].append(p - h)

        mean_dev = []
        median_dev = []
        std_dev = []
        for h in range(min_h, max_h + 1):
            if h in by_h:
                ary = np.array(by_h[h])
                by_h[h] = ary
                mean_dev.append(ary.mean())
                median_dev.append(np.median(ary))
                std_dev.append(np.std(ary))
            else:
                mean_dev.append(float('NaN'))
                median_dev.append(float('NaN'))
                std_dev.append(float('NaN'))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        new_x_ticks = []
        data = []
        for i in range(min_h, max_h + 1):
            if i in by_h:
                new_x_ticks.append("%d\n$n=%d$" % (i, len(by_h[i])))
                data.append(by_h[i])
            else:
                new_x_ticks.append("%d" % i)
                data.append([float('nan')])
        ax.boxplot(data)
        ax.set_xticklabels(new_x_ticks)

        ax.set_xlabel(orig_label)
        ax.set_ylabel(pred_label)
        ax.set_title(title)
        fig.tight_layout()
        for format in formats:
            fig.savefig(os.path.join(directory, file + "." + format))
        plt.close(fig)

    @staticmethod
    def _analyse_from_predictions_deviation_dep_on_similarity(
            predicted, original, similarity,
            title, pred_label, orig_label,
            file, directory=".", formats=MATPLOTLIB_OUTPUT_FORMATS):
        print("SIMILARITIES", sorted(similarity))
        return
        by_h = {}
        min_h = min(original)
        max_h = max(original)
        for idx in range(len(original)):
            h = original[idx]
            p = predicted[idx]
            if h not in by_h:
                by_h[h] = []
            by_h[h].append(p - h)

        mean_dev = []
        median_dev = []
        std_dev = []
        for h in range(min_h, max_h + 1):
            if h in by_h:
                ary = np.array(by_h[h])
                by_h[h] = ary
                mean_dev.append(ary.mean())
                median_dev.append(np.median(ary))
                std_dev.append(np.std(ary))
            else:
                mean_dev.append(float('NaN'))
                median_dev.append(float('NaN'))
                std_dev.append(float('NaN'))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        new_x_ticks = []
        data = []
        for i in range(min_h, max_h + 1):
            if i in by_h:
                new_x_ticks.append("%d\n$n=%d$" % (i, len(by_h[i])))
                data.append(by_h[i])
            else:
                new_x_ticks.append("%d" % i)
                data.append([float('nan')])
        ax.boxplot(data)
        ax.set_xticklabels(new_x_ticks)

        ax.set_xlabel(orig_label)
        ax.set_ylabel(pred_label)
        ax.set_title(title)
        fig.tight_layout()
        for format in formats:
            fig.savefig(os.path.join(directory, file + "." + format))
        plt.close(fig)

    """----------------------LAYER HELP METHODS------------------------------"""

    @staticmethod
    def next_dense(prev, neurons, activation=None, dropout=None):
        next = layers.Dense(neurons, activation=activation)(prev)
        if dropout is not None:
            next = layers.Dropout(dropout)(next)
        return next

    """-------------------------OTHER METHODS--------------------------------"""

    @staticmethod
    def parse(tree, item_cache):
        obj = parser.try_lookup_obj(tree, item_cache, KerasNetwork, None)
        if obj is not None:
            return obj
        else:
            raise ArgumentException("The definition of KerasNetwork can "
                                    "only be used for look up of any previously"
                                    " defined KerasNetwork via "
                                    "'keras(id=ID)'")


main_register.append_register(KerasNetwork, "keras")


class DomainPropertiesKerasNetwork(KerasNetwork):
    arguments = parset.ClassArguments('DomainPropertiesKerasNetwork',
                                      KerasNetwork.arguments)

    def __init__(self, load=None, store=None, formats=None, out=".",
                 count_samples=False, test_similarity=None, variables=None, id=None,
                 domain_properties=None):
        KerasNetwork.__init__(self, load, store, formats, out, count_samples,
                              test_similarity,
                              variables, id)

        self._domain_properties = None
        self._set_domain_properties(domain_properties)

    def _set_domain_properties(self, dp):
        self._domain_properties = dp
        self._gnd_static_str = (set() if dp is None else
                                set([str(item) for item in
                                     self._domain_properties.gnd_static]))
        print(self._gnd_static_str)

    def initialize(self, msgs, *args, domain_properties=None, **kwargs):
        if domain_properties is not None:
            self._set_domain_properties(domain_properties)
        KerasNetwork.initialize(self, msgs, *args, **kwargs)

    def _convert_state(self, state, format):
        """
        TODO: Add conversion for more formats, esp. StateFormat.Objects
        TODO: Add static pruning
        :param state: state description
        :param format: StateFormat in which the state is given
        :return:
        """
        if self._domain_properties is None:
            return KerasNetwork._convert_state(self, state, format)

        else:
            if format != StateFormat.Full:
                raise ValueError(
                    "Unable to convert data of the given format into the"
                    "internal representation:" + str(format))
            parts = state.split("\t")
            data = []
            for idx in range(len(parts)):
                if parts[idx][:-1] in self._gnd_static_str:
                    continue
                data.append(0 if parts[idx][-1] == "-" else 1)
            return np.array(data)

    @staticmethod
    def parse(tree, item_cache):
        obj = parser.try_lookup_obj(tree, item_cache,
                                    DomainPropertiesKerasNetwork, None)
        if obj is not None:
            return obj
        else:
            raise ArgumentException("The definition of "
                                    "DomainPropertyKerasNetwork can "
                                    "only be used for look up of any previously"
                                    " defined KerasNetwork via 'dp_keras(id=ID)'")


main_register.append_register(DomainPropertiesKerasNetwork, "dp_keras")


class MLPDynamicKeras(DomainPropertiesKerasNetwork):
    arguments = parset.ClassArguments('MLPDynamicKeras',
                                      DomainPropertiesKerasNetwork.arguments,
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
                                             "count_samples", "test_similarity",
                                             "variables", "id"]
                                      )

    def __init__(self, hidden, output_units=-1, activation="sigmoid",
                 dropout=None, optimizer="adam", loss="mean_squared_error",
                 load=None, store=None, formats=None, out=".",
                 count_samples=False, test_similarity=None, variables=None, id=None,
                 domain_properties=None):
        DomainPropertiesKerasNetwork.__init__(
            self, load, store, formats, out, count_samples, test_similarity,
            variables, id, domain_properties=domain_properties)
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
        self._count_samples_hasher = lambda x, y: hash(str((x,y)))
        # Either self._domain_properties will be used to determine the state
        # size or on initialization the state size has to be given
        # If both is given, the DomainProperties will be prefered
        self._state_size = None

    def _initialize_general(self, *args, state_size=None, **kwargs):
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

        in_state = layers.Input(shape=(input_units,))
        in_goal = layers.Input(shape=(input_units,))
        next = layers.concatenate([in_state, in_goal], axis=-1)

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

    def _finalize(self):
        pass

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  KerasNetwork)

main_register.append_register(MLPDynamicKeras, "mlp_dyn_keras")