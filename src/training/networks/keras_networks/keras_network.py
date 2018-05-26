from . import KerasDataGenerator, store_keras_model_as_protobuf

from .. import Network, NetworkFormat

from ... import parser_tools as parset
from ... import parser
from ... import main_register

from ...bridges import StateFormat
from ...misc import similarities, InvalidModuleImplementation
from ...parser_tools import ArgumentException

import json
import keras
from keras import layers
import matplotlib as mpl
import math
mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# List of formats in which the matplotlib figures shall be stored
MATPLOTLIB_OUTPUT_FORMATS=["png"]
COLOR_DATA_MEAN = "g"
ALPHA_DATA_MEAN = 0.7

class KerasNetwork(Network):
    arguments = parset.ClassArguments('KerasNetwork', Network.arguments,
        ("epochs", True, 1, int, "Number of training epochs per training call"),
        ("count_samples", True, False, parser.convert_bool,
         "Counts how many and which samples where used during training"
         " (This increases the runtime)."),
        ("test_similarity", True, None, similarities.get_similarity,
         "Estimates for every sample in the evaluation data how close it is"
         " to the trainings data. For this to work ALL training data is kept in"
         "memory, this could need a lot of memory"
         " (provide name of a similarity measure)"),
        ("graphdef", "True", None, str,
         "Name for an ASCII GraphDef file of the stored Protobuf model (only "
         "applicable if Protobuf model is stored)"),
        order=["load", "store", "formats", "out", "epochs", "count_samples",
               "test_similarity", "graphdef", "variables", "id"])

    def __init__(self, load=None, store=None, formats=None, out=".",
                 epochs=1000, count_samples=False, test_similarity=None,
                 graphdef=None,
                 variables=None, id=None):
        Network.__init__(self, load, store, formats, out, variables, id)
        self._epochs = epochs
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
        # Callback functions for the keras_networks fitting
        self._callbacks = None

        # Variables for storing Protobuf (ignore if you do not support storing
        # Protobuf files, but in general keras networks can be converted)
        self._quantize = False  # Tensorflow quantize feature
        self._theano = False  # The model used theano as backend
        self._num_outputs = 1  # Number of output PATHS of the network (not output nodes)
        self._prefix_outputs = ""  # Prefix before every output path
        self._store_graphdef = graphdef  # Store GraphDef of Tensorflow Graph if file name is given


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

    def get_preferred_state_formats(self):
        return [StateFormat.Full]

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

            path_format = path + "." + format.suffix
            if format == NetworkFormat.hdf5:
                self._model.save(path_format)
            elif format == NetworkFormat.protobuf:
                #graphdef = (None if self._store_graphdef is None else
                #            os.path.join(self.path_out, self._store_graphdef))
                store_keras_model_as_protobuf(
                    self._model, os.path.dirname(path), os.path.basename(path_format),
                    quantize=self._quantize, theano=self._theano,
                    num_outputs=self._num_outputs, prefix_outputs=self._prefix_outputs,
                    store_graphdef=self._store_graphdef
                )


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

    def _analyse(self, directory, prefix):
        mean_prediction = round(self._evaluation[1].mean())
        predicted = np.round(self._evaluation[0])
        predicted_correct = (mean_prediction == predicted).sum()

        KerasNetwork._analyse_from_history_plot(
            self._history, ['acc', 'val_acc'], "Model Accuracy", "accuracy",
            "epoch", ['train', 'test'], prefix + "evolution_accuracy", directory,
            hline=(float(predicted_correct)/predicted.size))

        KerasNetwork._analyse_from_history_plot(
            self._history, ['loss', 'val_loss'], "Model Loss", "loss", "epoch",
            ['train', 'val'], prefix + "evolution_loss", directory)

        KerasNetwork._analyse_from_predictions_scatter(
            self._evaluation[0],  self._evaluation[1], "Predictions",
            "original h", "predicted h", "predictions_scatter", directory)

        KerasNetwork._analyse_from_predictions_scatter_tiles(
            self._evaluation[0], self._evaluation[1],
            "Prediction Probabilities with resp. to the correct prediction",
            "original h", "predicted h", prefix + "predictions_tiles", directory)

        KerasNetwork._analyse_from_predictions_deviation(
            self._evaluation[0], self._evaluation[1], "Prediction Deviations",
            "count", "deviation", prefix + "deviations", directory,
            diff_mean_to_prediction=True)

        KerasNetwork._analyse_from_predictions_deviation_dep_on_h(
            self._evaluation[0], self._evaluation[1],
            "Prediction Deviations depending on original", "deviation",
            "original", prefix + "deviations_dep_h", directory,
            diff_mean_to_prediction=True)

        KerasNetwork._analyse_from_predictions_deviation_dep_on_similarity(
            self._evaluation[0], self._evaluation[1], self._evaluation[2],
            "Prediction Deviations depending on the similarity to training samples",
            "deviation",
            "similarity", prefix + "deviations_dep_sim", directory)

        state_space_size = None if (not hasattr(self,"_domain_properties") or self._domain_properties is None) else self._domain_properties.state_space_size
        reachable_sss = None if (not hasattr(self,"_domain_properties") or self._domain_properties is None) else self._domain_properties.upper_bound_reachable_state_space_size
        KerasNetwork._analyse_misc(len(self._count_samples_hashes), state_space_size, reachable_sss,
                                   prefix + "misc", directory)

        analysis_data = {"histories": [h.history for h in self._histories],
                         "evaluations": [(e[0].tolist(), e[1].tolist()) for e in self._evaluations],
                         "count_samples": len(self._count_samples_hashes) if self._count_samples else "NA",
                         "state_space_size" : state_space_size,
                         "upper_reachable_state_space_bound" : reachable_sss }

        analysis_data["model"] = ""
        def add_model_summary(x):
            nonlocal analysis_data
            analysis_data["model"] += x + "\n"
        self._model.summary(print_fn=add_model_summary)

        with open(os.path.join(directory, "analysis.meta"), "w") as f:
            json.dump(analysis_data, f)

    """----------------------ANALYSIS METHODS--------------------------------"""
    @staticmethod
    def _analyse_misc(samples_seen, state_space_size, reachable_sss,
                      file, directory=".", formats=MATPLOTLIB_OUTPUT_FORMATS):
        fig = plt.figure()
        if state_space_size is not None:
            sss_bar = [state_space_size]
            new_xticks = [None, "Full State Space"]
            if reachable_sss is not None:
                sss_bar.append(reachable_sss)
                new_xticks.append("Reachable State Space")
            ax = fig.add_subplot(1, 1, 1)
            ax.bar(np.arange(len(sss_bar)), sss_bar, width=1,
                   color='g', align='center') #, label="State Space Size")
            ax.bar(np.arange(len(sss_bar)), [samples_seen] * len(sss_bar),
                   color='r', align='center', label="Training Samples Count")
            ax.set_title("Seen Parts of State Space")
            ax.set_ylabel("samples")
            ax.set_yscale("log")
            ax.set_xticklabels(new_xticks)

            #ax.set_xlabel(xlabel)

        fig.tight_layout()
        for format in formats:
            fig.savefig(os.path.join(directory, file + "." + format))
        plt.close(fig)
    @staticmethod
    def _analyse_from_history_plot(history, measures, title, ylabel, xlabel,
                                   legend, file, directory=".",
                                   formats=MATPLOTLIB_OUTPUT_FORMATS,
                                   hline=None):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for m in measures:
            ax.plot(history.history[m])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        if hline is not None:
            ax.axhline(hline, color=COLOR_DATA_MEAN, alpha=ALPHA_DATA_MEAN)
            legend.append("Predicting Data Mean")
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
                                            ylabel, xlabel, file,
                                            directory=".",
                                            formats=MATPLOTLIB_OUTPUT_FORMATS,
                                            diff_mean_to_prediction=False):
        # (Legend, Color, Alpha, Data)
        deviations = [('Deviation of Predictions','b', 1, predicted - original)]
        if diff_mean_to_prediction:
            mean = original.mean()
            deviations.insert(0, ("Deviation of Data Mean", COLOR_DATA_MEAN,
                                  ALPHA_DATA_MEAN, mean - original))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        round = -1
        for (label, color, alpha, data) in deviations:
            round += 1
            width = 0.4 #0.9 - 0.2 * round
            dev = np.round(data.astype(np.float))
            unique, counts = np.unique(dev, return_counts=True)
            # Otherwise they are numpy values and do not hash as needed
            unique = [float(i) for i in unique]
            occurrences = dict(zip(unique, counts))
            min_d, max_d = min(unique), max(unique)

            bars = np.arange(min_d, max_d + 1) -width*len(deviations)/2 + round * width + width/2
            heights = [0 if not i in occurrences else occurrences[i]
                       for i in range(math.floor(min_d), math.ceil(max_d) + 1)]
            ax.bar(bars, heights, width=width,
                   color=color, alpha=alpha, align='center', label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if len(deviations) > 1:
            ax.legend()
        fig.tight_layout()
        for format in formats:
            fig.savefig(os.path.join(directory, file + "." + format))
        plt.close(fig)

    @staticmethod
    def _analyse_from_predictions_deviation_dep_on_h(predicted, original, title,
                                                     pred_label, orig_label,
                                                     file, directory=".",
                                                     formats=MATPLOTLIB_OUTPUT_FORMATS,
                                                     diff_mean_to_prediction=False):

        by_h = {}
        min_h = min(original)
        max_h = max(original)
        for idx in range(len(original)):
            h = original[idx]
            p = predicted[idx]
            if h not in by_h:
                by_h[h] = []
            by_h[h].append(p - h)

        new_x_ticks = []
        data = []
        means = []
        mean = original.mean() if diff_mean_to_prediction else None
        for i in range(min_h, max_h + 1):
            if diff_mean_to_prediction:
                means.append(mean - i)
            if i in by_h:
                new_x_ticks.append("%d\n$n=%d$" % (i, len(by_h[i])))
                data.append(by_h[i])
            else:
                new_x_ticks.append("%d" % i)
                data.append([float('nan')])
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.boxplot(data)
        ax.set_xticklabels(new_x_ticks)
        if diff_mean_to_prediction:
            ax.scatter(np.arange(len(data)) + 1, means, color='r', alpha=0.6,
                       label="Deviation to Data Mean")
        ax.set_xlabel(orig_label)
        ax.set_ylabel(pred_label)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        for format in formats:
            fig.savefig(os.path.join(directory, file + "." + format))
        plt.close(fig)

    @staticmethod
    def _analyse_from_predictions_deviation_dep_on_similarity(
            predicted, original, similarity,
            title, pred_label, orig_label,
            file, directory=".", formats=MATPLOTLIB_OUTPUT_FORMATS,
            steps=10, precision="%.2f"):

        min_sim, max_sim = min(similarity), max(similarity)
        step_sim = (max_sim-min_sim)/float(steps)
        def get_bin(x):
            return max(0, min(steps - 1, int((x - min_sim) / step_sim)))
        by_sim = {}
        for idx in range(len(original)):
            d = original[idx] - predicted[idx]
            s = similarity[idx]
            b = get_bin(s)
            if b not in by_sim:
                by_sim[b] = []
            by_sim[b].append(d)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        new_x_ticks = []
        data = []
        for i in range(steps):
            if i in by_sim:
                sim_range = ("[%s,%s" % (precision, precision)
                             + ("]" if i == steps -1 else "["))
                sim_range = sim_range % (min_sim + i * step_sim,
                                         min_sim + (i + 1) * step_sim)

                new_x_ticks.append("%s\n$n=%d$" % (sim_range, len(by_sim[i])))
                data.append(by_sim[i])

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
                                    "'keras_networks(id=ID)'")


main_register.append_register(KerasNetwork, "keras_networks")


class KerasDomainPropertiesNetwork(KerasNetwork):
    arguments = parset.ClassArguments('DomainPropertiesKerasNetwork',
                                      KerasNetwork.arguments)

    def __init__(self, load=None, store=None, formats=None, out=".", epochs=1000,
                 count_samples=False, test_similarity=None, graphdef=None,
                 variables=None, id=None,
                 domain_properties=None):
        KerasNetwork.__init__(self, load, store, formats, out, epochs,
                              count_samples, test_similarity, graphdef,
                              variables, id)

        self._domain_properties = None
        self._gnd_static_str = None
        self._set_domain_properties(domain_properties)

    def _set_domain_properties(self, dp):
        """
        You might you this method from outside of this object, but pay attention
        or confusion will arise (e.g. use it before initializing should be safe)
        :param dp:
        :return:
        """
        self._domain_properties = dp
        self._gnd_static_str = (set() if dp is None else
                                set([str(item) for item in
                                     self._domain_properties.gnd_static]))

    def initialize(self, msgs, *args, domain_properties=None, **kwargs):
        if domain_properties is not None:
            self._set_domain_properties(domain_properties)
        KerasNetwork.initialize(self, msgs, *args, **kwargs)

    def _convert_state(self, state, format):
        """
        TODO: Add conversion for more formats, esp. StateFormat.Objects
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
                                    KerasDomainPropertiesNetwork, None)
        if obj is not None:
            return obj
        else:
            raise ArgumentException("The definition of "
                                    "KerasDomainPropertiesNetwork can "
                                    "only be used for look up of any previously"
                                    " defined KerasNetwork via 'dp_keras(id=ID)'")


main_register.append_register(KerasDomainPropertiesNetwork, "keras_dp")
