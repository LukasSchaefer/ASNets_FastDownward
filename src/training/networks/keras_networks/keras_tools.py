from __future__ import print_function

import keras
import math
import numpy as np
import random

class ProgressCheckingCallback(keras.callbacks.Callback):
    def __init__(self, monitor, epochs, threshold=None, minimize=True):
        keras.callbacks.Callback.__init__(self)
        self._monitor = monitor
        self._epochs = epochs
        self._threshold = threshold
        self._minimize = minimize
        self._active = True
        self.failed = False

    def on_train_begin(self, logs={}):
        self._active = True

    def on_epoch_end(self, epoch, logs={}):
        if not self._active:
            return

        if epoch >= self._epochs:  # Fire once after self._epochs epochs
            self._active = False
            current = logs.get(self._monitor)
            if current is None:
                raise RuntimeError("ProgressCheckingCallback requires %s"
                                   " available!" % self._monitor)


            if ((self._minimize and current >= self._threshold)
                or (not self._minimize and current <= self._threshold)):
                self.model.stop_training = True
                self.failed = True


class KerasDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, x_fields=None, y_fields=None, types=["O"], batch_size=100,
                 x_converter=None, y_converter=None, y_remember=None,
                 shuffle=True, count_diff_samples=None,
                 similarity=None):
        """

        :param data: Single or list of SizeBatchData object(s)
        :param x_fields: Fields in SizeBatchData entries to use as x data
                         if not given, uses all fields which are not used
                         for y values
        :param y_fields: fields in SizeBatchData entries to use as y data
                         if not given, uses the last field as y value
        :param types: types from the SizeBatchData to use
        :param batch_size:
        :param x_converter:
        :param y_converter:
        :param y_remember: List object. If given, all y values for which data
                           was generated is appended (in the batches they
                           are generated, thus, at the end this is a list of
                           numpy arrays)(any object with an
                           append method can be used). Example use case is, if
                           a network evaluation only returns predications, but
                           not the original values
        :param shuffle: shuffle data after each epoche
        :param count_diff_samples: If None it does not keep track of the
                                   different samples generated. Otherwise this
                                   is a callable which expects two arguments
                                   (X and Y data of sample) and generates a
                                   hash for them.
        :param similarity: If None it does not measure the similarity between an
                           generated value X and all entries of an iterable of
                           SizeBatchData (which would be given in similarity).
                           Otherwise similarity is a triple:
                               -list to store the similarities in batches,
                               -iterable of SizeBatchData to compare to,
                               -method to calculate the similarity with the
                                interface (sample, iterable of SizeBatchData)
                           Using this feature will cost a lot of performance!
        """
        self.data = data if isinstance(data, list) else [data]
        self.y_fields = (data[0].nb_fields - 1) if y_fields is None else y_fields
        self.x_fields = x_fields
        if self.x_fields is None:
            # Used to test if integer in y_fields
            y_tester = lambda x: x in self.y_fields
            try:
                y_tester(3)
            except TypeError:
                y_tester = lambda x: x == self.y_fields

            self.x_fields = []
            for i in range(data[0].nb_fields):
                if not y_tester(i):
                    self.x_fields.append(i)
        self.x_converter = x_converter
        self.y_converter = y_converter
        self.y_remember = y_remember
        self.similarity = similarity

        self.types = types
        self.batch_size = batch_size
        self.batch_order = []
        for idx_ds in range(len(self.data)):
            ds = self.data[idx_ds]
            for type in types:
                for idx_batch in range(len(ds.data[type])):
                    batch = ds.data[type][idx_batch]
                    count = int(math.ceil(len(batch)/batch_size))
                    step = int(len(batch)/count)
                    start = 0
                    for i in range(count):
                        if i == count - 1:
                            self.batch_order.append((idx_ds, type, idx_batch,
                                                     start, len(batch)))
                        else:
                            self.batch_order.append((idx_ds, type, idx_batch,
                                                     start, start + step))
                        start += step
        self._next = -1

        self.shuffle = shuffle
        self.count_diff_samples = count_diff_samples
        self.generated_sample_hashes = set()

        """
        self.precaching = precaching
        self.cache_x = None
        self.cache_y = None
        if self.precaching:
            self.__compute_caching(types)
        
    def __compute_caching(self, types):
        self.cache_x = []
        self.cache_y = []
        for idx_ds in range(len(self.data)):
            ds = self.data[idx_ds]
            self.cache_x.append({})
            self.cache_y.append({})
            for type in types:
                self.cache_x[idx_ds][type] = []
                self.cache_y[idx_ds][type] = []
                for idx_batch in range(len(ds.data[type])):
                    batch = ds.data[type][idx_batch]

                    tmp = batch[:, self.x_fields]
                    self.cache_x[idx_ds][type].append(
                        tmp if self.x_converter is None else
                        self.x_converter(tmp))
                    tmp = batch[:, self.y_fields]
                    self.cache_y[idx_ds][type].append(
                        tmp if self.y_converter is None else
                        self.y_converter(tmp))


        """
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.batch_order)

    def __getitem__(self, index):
        'Generate one batch of data'
        index = self.batch_order[index]
        entries = self.data[index[0]].data[index[1]][index[2]][index[3]:index[4]]
        x = entries[:, self.x_fields]
        y = entries[:, self.y_fields]

        if self.count_diff_samples is not None:
            for i in range(len(x)):
                self.generated_sample_hashes.add(self.count_diff_samples(x[i], y[i]))

        if self.similarity is not None:
            (sim_batches, data_sets, measure) = self.similarity
            ary = np.ndarray(shape=(len(entries),), dtype=float)
            for i in range(len(x)):
                ary[i] = measure(entries[i, :], data_sets)
            sim_batches.append(ary)

        if self.x_converter is not None:
            x = self.x_converter(x)
        if self.y_converter is not None:
            y = self.y_converter(y)
        if self.y_remember is not None:
            self.y_remember.append(y)
        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.count_diff_samples = None  # All different samples have been seen
        if self.shuffle:
            for ds in self.data:
                for type in self.types:
                    for batch in ds.data[type]:
                        np.random.shuffle(batch)
            random.shuffle(self.batch_order)


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras_networks.utils.to_categorical(y, num_classes=self.n_classes)
        """
        raise NotImplementedError("__data_generation not implemented")

import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.tools.graph_transforms import TransformGraph


def store_keras_model_as_protobuf(
        model, directory=".", file="network.pb",
        quantize=False, theano=False, num_outputs=1, prefix_outputs="",
        show_outputs=False, store_graphdef=None):
    """
    Copyright (c) 2017, by the Authors: Amir H. Abdi
    This software is freely available under the MIT Public License.

    Converts a
    :param model:
    :param directory:
    :param file:
    :param quantize:
    :param theano:
    :param num_outputs:
    :param prefix_outputs:
    :param show_outputs:
    :param store_graphdef:
    :return:
    """
    # Prepare variables
    if directory == "" or directory is None:
        directory = "."

    previous_learning_phase = K.learning_phase()
    if isinstance(previous_learning_phase, tf.Tensor):
        # HACK!!!
        previous_learning_phase = int(previous_learning_phase.name[-1])
    K.set_learning_phase(0)

    if theano:
        K.set_image_data_format('channels_first')
    else:
        K.set_image_data_format('channels_last')

    # Set node names in computation graph
    pred_node_names = [None] * num_outputs
    pred = [None] * num_outputs
    for i in range(num_outputs):
        pred_node_names[i] = prefix_outputs + str(i)
        pred[i] = tf.identity(model.outputs[i], name=pred_node_names[i])
    if show_outputs:
        print('Output nodes names are: ', pred_node_names)

    # Store readable GraphDef
    sess = K.get_session()
    if store_graphdef is not None:
        tf.train.write_graph(sess.graph.as_graph_def(),
                             directory, store_graphdef, as_text=True)

    # Store Protobuf
    if quantize:
        transforms = ["quantize_weights", "quantize_nodes"]
        transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [],
                                               pred_node_names, transforms)
        constant_graph = graph_util.convert_variables_to_constants(
            sess, transformed_graph_def, pred_node_names)
    else:
        constant_graph = graph_util.convert_variables_to_constants(
            sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constant_graph, directory, file, as_text=False)

    K.set_learning_phase(previous_learning_phase)
