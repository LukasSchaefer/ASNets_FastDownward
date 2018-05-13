from . import Network, NetworkFormat
from .keras_tools import KerasDataGenerator

from .. import parser_tools as parset
from .. import parser
from .. import main_register
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
        order=["load", "store", "formats", "out", "count_samples",
               "variables", "id"])

    def __init__(self, load=None, store=None, formats=None, out=".",
                 count_samples=False,
                 variables=None, id=None):
        Network.__init__(self, load, store, formats, out, variables, id)
        self._epochs = 750
        self._model = None
        self._history = None
        self._histories = []
        self._evaluation = None
        self._evaluations = []
        self._training_samples = set()
        self._count_samples = count_samples

        """Concrete networks might need to set those values"""
        """For examples of the following values check existing implementations"""
        self._x_fields_extractor = None  # callable which gets a SampleBatchData
        self._y_fields_extractor = None  # object and returns an ordered list of
                                        # fields to use for the X/Y data
        self._x_converter = None  # Callable which converts extracted batch of
                                 # X fields to the format needed by the network
        self._y_converter = None  # Similar to x_converter
        self._callbacks = None

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
            shuffle=True, count_diff_samples=True)

        kdg_test = None
        if dtest is not None:
            dtest = self._convert_data(dtest)
            kdg_test = KerasDataGenerator(
                dtest,
                x_fields=None if self._x_fields_extractor is None else self._x_fields_extractor(dtest[0]),
                y_fields=None if self._y_fields_extractor is None else self._y_fields_extractor(dtest[0]),
                x_converter=self._x_converter,
                y_converter=self._y_converter,
                shuffle=True, count_diff_samples=True)

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
        self._training_samples.update(kdg_train.generated_samples)
        self._history = history
        self._histories.append(history)
        return history

    def evaluate(self, data):
        data = self._convert_data(data)
        y_labels = []

        kdg_eval = KerasDataGenerator(
            data,
            x_fields=None if self._x_fields_extractor is None else self._x_fields_extractor(data[0]),
            y_fields=None if self._y_fields_extractor is None else self._y_fields_extractor(data[0]),
            x_converter=self._x_converter,
            y_converter=self._y_converter,
            y_remember=y_labels)

        result = self._model.predict_generator(
            kdg_eval, max_queue_size=10, workers=1, use_multiprocessing=False)

        y_labels = np.concatenate(y_labels)
        result = (result.squeeze(axis=1), y_labels)
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
            "Prediction Probabilities with resp.\nto the correct prediction",
            "original h", "predicted h", "predictions_tiles", directory)

        KerasNetwork._analyse_from_predictions_deviation(
            self._evaluation[0], self._evaluation[1], "Prediction Deviations",
            "deviation", "count", "deviations", directory)

        KerasNetwork._analyse_from_predictions_deviation_dep_on_h(
            self._evaluation[0], self._evaluation[1],
            "Prediction Deviations depending on original", "deviation",
            "original", "deviations_dep_h", directory)


        analysis_data = {"histories": [h.history for h in self._histories],
                         "evaluations": [(e[0].tolist(), e[1].tolist()) for e in self._evaluations]}

        with open(os.path.join(directory, "analysis.meta"), "w") as f:
            json.dump(analysis_data, f)
        print("SEEN SAMPLES", len(self._training_samples))

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
        ax.scatter(original, predicted, s=80, c='r', alpha=0.2)
        ax.set_xlabel(orig_label)
        ax.set_ylabel(pred_label)
        ax.set_title(title)
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

        #max_h = max(max_o_h, max_p_h)
        #min_h = min(min_o_h, min_p_h)

        #range_h = math.ceil(max_h - min_h + 1)

        by_h = {}
        for idx in range(len(predicted)):
            h = original[idx]
            p = predicted[idx]
            if h not in by_h:
                by_h[h] = []
            by_h[h].append(p)

        DECIMALS = 1
        POWER = 10**DECIMALS
        h_p_bins = []
        for i in range(min_p_h*POWER, max_p_h*POWER + 1):
            h_p_bins.append(float(i)/POWER)

        tiles = np.ndarray(shape=(max_o_h - min_o_h + 1, len(h_p_bins)), dtype=float)
        for h_o in range(min_o_h, max_o_h + 1):
            if h_o not in by_h:
                for idx_p in range(len(h_p_bins)):
                    tiles[h_o - min_o_h, idx_p] = float("NaN")
            else:
                ary = np.around(np.array(by_h[h_o]), decimals=DECIMALS)
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
        bins = math.ceil(max(dev) - min(dev) + 1)
        ax.hist([dev], bins=bins)
        ax.set_xlabel(orig_label)
        ax.set_ylabel(pred_label)
        ax.set_title(title)
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
        ax.bar(np.arange(min_h, max_h + 1), std_dev, color="orange")
        c = 0
        for i in ax.patches:
            h = min_h + c
            c += 1
            s = "0" if h not in by_h else str(len(by_h[h]))
            # get_x pulls left or right; get_height pushes up or down
            ax.text(i.get_x() + i.get_width()*0.25, -0.25, \
                    s, fontsize=11,
                    color='dimgrey')
        c = 0
        for i in ax.patches:
            h = min_h + c
            s = str(round(std_dev[c], 2))
            c += 1
            # get_x pulls left or right; get_height pushes up or down
            ax.text(i.get_x() + i.get_width()*0.25, i.get_height(), \
                    s, fontsize=11,
                    color='dimgrey')


        ax.plot(np.arange(min_h, max_h + 1), mean_dev, color="blue")
        ax.scatter(np.arange(min_h, max_h + 1), median_dev, color="blue")

        ax.set_xlabel(orig_label)
        ax.set_ylabel(pred_label)
        ax.set_title(title)
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
                 count_samples=False, variables=None, id=None,
                 domain_properties=None):
        KerasNetwork.__init__(self, load, store, formats, out, count_samples,
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
                                             "count_samples",
                                             "variables", "id"]
                                      )

    def __init__(self, hidden, output_units=-1, activation="sigmoid",
                 dropout=None, optimizer="adam", loss="mean_squared_error",
                 load=None, store=None, formats=None, out=".",
                 count_samples=False, variables=None, id=None,
                 domain_properties=None):
        DomainPropertiesKerasNetwork.__init__(
            self, load, store, formats, out, count_samples,
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