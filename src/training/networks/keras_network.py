from . import Network

from .keras_tools import KerasDataGenerator

from .. import parser_tools as parset
from .. import parser
from .. import main_register

import keras
from keras import layers
import matplotlib as mpl
import math
mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# TODO delete pass or replace with relevant code
class KerasNetwork(Network):
    arguments = parset.ClassArguments('KerasNetwork', Network.arguments)

    def __init__(self, load=None, store=None, formats=None, variables=None, id=None):
        Network.__init__(self, load, store , formats, variables, id)

    def _initialize(self, **kwargs):
        pass

    def _finalize(self):
        pass

    def _store(self):
        pass

    def train(self, data):
        pass

    def evaluate(self, data):
        pass

    @staticmethod
    def next_dense(prev, neurons, activation=None, dropout=None):
        next = layers.Dense(neurons, activation=activation)(prev)
        if dropout is not None:
            next = layers.Dropout(dropout)(next)
        return next

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  KerasNetwork)

main_register.append_register(KerasNetwork, "keras")



class MLPDynamicKeras(KerasNetwork):
    arguments = parset.ClassArguments('KerasNetwork', KerasNetwork.arguments,
                                      ("hidden", False, None, int, "Number of hidden layers"),
                                      ("output_units", False, True, int,
                                       "Classification network with output_units output units or regression network if -1"),
                                      ("activation", True, "sigmoid", str, "Activation function of hidden layers"),
                                      ("dropout", True, None, int, "Dropout probability or None if no dropout"),
                                      ("optimizer", True, "adam", str, "Optimization algorithm"),
                                      ("loss", True, "mean_squared_error", str, "Loss function"),
                                      ("out", True, ".", str, "Path to output directory"),
                                      order=["hidden", "output_units",
                                             "activation", "dropout",
                                             "optimizer", "loss",
                                             "load", "store", "out", "formats",
                                             "variables", "id"]
                                      )

    def __init__(self, hidden, output_units=-1, activation="sigmoid",
                 dropout=None, optimizer="adam", loss="mean_squared_error",
                 load=None, store=None, out=".", formats=None, variables=None, id=None):
        KerasNetwork.__init__(self, load, store, formats, variables, id)
        self._hidden = hidden
        self._output_units = output_units
        self._activation = activation
        self._dropout = dropout
        self._optimizer = optimizer
        self._loss = loss
        self._out = out
        self._model = None
        self._history = None
        self._histories = []
        self._evaluation = None
        self._evaluations = []

    def _compile(self, optimizer, loss, metrics):
        self._model.compile(optimizer=optimizer,
                            loss=loss,
                            metrics=metrics)

    def _initialize(self, **kwargs):
        input_units = kwargs["state_size"]
        print("IU", input_units)
        regression = self._output_units == -1
        output_units = 1 if regression else self._output_units

        in_state = layers.Input(shape=(input_units,))
        in_goal = layers.Input(shape=(input_units,))
        next = layers.concatenate([in_state, in_goal], axis=-1)

        unit_diff = input_units*2 - output_units
        step = int(unit_diff/(self._hidden + 1))
        units = input_units * 2
        for i in range(self._hidden):
            units -= step
            next = KerasNetwork.next_dense(next, units, self._activation, self._dropout)
        next = KerasNetwork.next_dense(next, output_units,
                                       "relu" if regression else "softmax", None)


        self._model = keras.Model(inputs=[in_state, in_goal], outputs=next)
        self._compile(self._optimizer, self._loss, ["accuracy"])


    def _finalize(self):
        pass

    def _store(self):
        pass

    def _convert_state(self, state):
        parts = state.split("\t")
        for idx in range(len(parts)):
            parts[idx] = 0 if parts[idx][-1] == "-" else 1
        return np.array(parts)

    def _convert_data(self, data):
        data = data if isinstance(data, list) else [data]
        for data_set in data:
            if data_set.is_finalized:
                print("Warning: Dataset previously finalized. Skipping now.")
                continue

            for type in data_set.data:
                for batch in data_set.data[type]:
                    for entry in batch:
                        entry[data_set.field_current_state] = (
                            self._convert_state(entry[data_set.field_current_state]))
                        entry[data_set.field_goal_state] = (
                            self._convert_state(
                                entry[data_set.field_goal_state]))
                        entry[data_set.field_other_state] = (
                            self._convert_state(
                                entry[data_set.field_other_state]))
            data_set.finalize()
        return data

    def train(self, dtrain, dtest=None):
        dtrain = self._convert_data(dtrain)


        kdg_train = KerasDataGenerator(dtrain,
                                       x_fields=[dtrain[0].field_current_state,
                                                 dtrain[0].field_goal_state],
                                       x_converter=lambda x: [np.stack(x[:,0], axis=0), np.stack(x[:,1], axis=0)])
        kdg_test = None
        if dtest is not None:
            dtest = self._convert_data(dtest)
            kdg_test = KerasDataGenerator(dtrain,
                                          x_fields=[dtest[0].field_current_state,
                                                     dtest[0].field_goal_state],
                                          x_converter=lambda x: [
                                              np.stack(x[:, 0], axis=0),
                                              np.stack(x[:, 1], axis=0)])

        history = self._model.fit_generator(kdg_train,
            epochs=1,
            verbose=1, callbacks=None,
            validation_data=kdg_test,
            validation_steps=None, class_weight=None,
            max_queue_size=10, workers=1,
            use_multiprocessing=False,
            shuffle=True, initial_epoch=0)

        self._history = history
        self._histories.append(history)
        return history


    def evaluate(self, data):
        data = self._convert_data(data)
        y_labels = []

        kdg_eval = KerasDataGenerator(data,
                                       x_fields=[data[0].field_current_state,
                                                 data[0].field_goal_state],
                                       x_converter=lambda x: [np.stack(x[:,0], axis=0), np.stack(x[:,1], axis=0)],
                                      y_remember=y_labels)


        result = self._model.predict_generator(kdg_eval,
                                                max_queue_size=10, workers=1,
                                                use_multiprocessing=False)
        y_labels = np.concatenate(y_labels)
        result = (result.squeeze(axis=1), y_labels)
        self._evaluation = result
        self._evaluations.append(result)
        return result


    def _analyse_from_history_plot(self, measures, title, ylabel, xlabel, legend, path):
        # Summarize history for accuracy
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for m in measures:
            ax.plot(self._history.history[m])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.legend(legend, loc='upper left')

        fig.savefig(os.path.join(self._out, path))
        plt.close(fig)

    def _analyse_from_predictions_scatter(self, predicted, original, title, pred_label, orig_label, path):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.scatter(original, predicted, s=80, c='r', alpha=0.2)
        #ax.scatter(y_test, y_test)
        ax.set_xlabel(orig_label)
        ax.set_ylabel(pred_label)
        ax.set_title(title)

        fig.savefig(os.path.join(self._out, path))
        plt.close(fig)

    def _analyse_from_predictions_deviation(self, predicted, original, title, pred_label, orig_label, path):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        dev = predicted - original
        bins = math.ceil(max(dev) - min(dev) + 1)
        ax.hist([dev],
                bins=bins)
        ax.set_xlabel(orig_label)
        ax.set_ylabel(pred_label)
        ax.set_title(title)
        fig.savefig(os.path.join(self._out, path))

    def _analyse_from_predictions_deviation_dep_on_h(self, predicted, original, title, pred_label, orig_label, path):

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
        ax.plot(np.arange(min_h, max_h + 1), mean_dev)
        ax.scatter(np.arange(min_h, max_h + 1), median_dev)
        ax.bar(np.arange(min_h, max_h + 1), std_dev)
        ax.set_xlabel(orig_label)
        ax.set_ylabel(pred_label)
        ax.set_title(title)
        fig.savefig(os.path.join(self._out, path))

    def analyse(self):
        # Summarize history for accuracy
        self._analyse_from_history_plot(['acc', 'val_acc'], "Model Accuracy", "accuracy", "epoch", ['train', 'val'], "evolution_accuracy.png")
        self._analyse_from_history_plot(['loss', 'val_loss'], "Model Loss", "loss", "epoch", ['train', 'val'], "evolution_loss.png")

        # scatter plot
        self._analyse_from_predictions_scatter(self._evaluation[0], self._evaluation[1], "Predictions", "original h", "predicted h", "predictions.png")

        self._analyse_from_predictions_deviation(self._evaluation[0],
                                               self._evaluation[1],
                                               "Prediction Deviations", "deviation",
                                               "count", "deviations.png")
        self._analyse_from_predictions_deviation_dep_on_h(self._evaluation[0],
                                                 self._evaluation[1],
                                                 "Prediction Deviations depending on original",
                                                 "deviation",
                                                 "original", "deviations_dep_h.png")

        """

        # create compare test and original
        # original= np.loadtxt('../../training_data_tests/combined/12loc_test_label.txt')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.hist((y_test, predicted_test),
                bins=max(max(y_test), max(predicted_test)) + 1)
        ax.set_xlabel('label');
        ax.set_ylabel('number of appearance');
        ax.legend(['original', 'predictions'], loc='upper left')

        fig.savefig(os.path.join(directory, 'NumberOfAppearanceInTest.png'))
        plt.close(fig)

        # create compare train and original
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.hist((y_train, predicted_train),
                bins=max(max(y_train), max(predicted_train)) + 1)
        ax.set_xlabel('label');
        ax.set_ylabel('number of appearance');
        ax.legend(['original', 'predictions'], loc='upper left')

        fig.savefig(os.path.join(directory, 'NumberOfAppearanceInTrain.png'))
        plt.close(fig)

        # scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.scatter(y_test, predicted_test, s=80, c='r', alpha=0.2)
        ax.scatter(y_test, y_test)
        ax.set_xlabel('original h*')
        ax.set_ylabel('predicted h*')
        ax.set_title('Predictions')

        fig.savefig(os.path.join(directory, 'ScatterPredictionsTest.png'))
        plt.close(fig)

        # scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.scatter(y_train, predicted_train, s=80, c='r', alpha=0.2)
        ax.scatter(y_train, y_train)
        ax.set_xlabel('original h*')
        ax.set_ylabel('predicted h*')
        ax.set_title('Predictions')

        fig.savefig(os.path.join(directory, 'ScatterPredictionsTrains.png'))
        plt.close(fig)

        # box plot of difference test data
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        bp_dict = ax.boxplot(diffTest)
        for line in bp_dict['medians']:
            x, y = line.get_xydata()[1]  # top of median line
            ax.annotate("%d" % y, xy=(x + 0.05, y), xytext=(x + 0.05, y))
        for line in bp_dict['boxes']:
            x, y = line.get_xydata()[1]  # top of median line
            ax.annotate("%d" % y, xy=(x + 0.05, y), xytext=(x + 0.05, y))
            x, y = line.get_xydata()[2]  # top of median line
            ax.annotate("%d" % y, xy=(x + 0.05, y), xytext=(x + 0.05, y))

        ax.set_ylabel('Difference')
        ax.set_title('Difference of test data prediction to correct h* value')

        fig.savefig(os.path.join(directory, 'BoxplotTest.png'))
        plt.close(fig)

        # box plot of difference test data
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        diffTrain = np.absolute(y_train - predicted_train)
        bp_dict = ax.boxplot(diffTrain)
        for line in bp_dict['medians']:
            x, y = line.get_xydata()[1]  # top of median line
            ax.annotate("%d" % y, xy=(x + 0.05, y), xytext=(x + 0.05, y))
        for line in bp_dict['boxes']:
            x, y = line.get_xydata()[1]  # top of median line
            ax.annotate("%d" % y, xy=(x + 0.05, y), xytext=(x + 0.05, y))
            x, y = line.get_xydata()[2]  # top of median line
            ax.annotate("%d" % y, xy=(x + 0.05, y), xytext=(x + 0.05, y))
        ax.set_ylabel('Difference')
        ax.set_title(
            'Difference of training data prediction to correct h* value')

        fig.savefig(os.path.join(directory, 'BoxplotTrain.png'))
        plt.close(fig)
    """
    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  KerasNetwork)

main_register.append_register(MLPDynamicKeras, "mlp_dyn_keras")