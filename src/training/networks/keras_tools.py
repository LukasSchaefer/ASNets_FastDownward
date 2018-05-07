import keras
import math
import numpy as np


class KerasDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, y_fields=None, x_fields=None, types=["O"], batch_size=100,
                 x_converter=None, y_converter=None, y_remember=None):

        self.data = data
        self.y_fields = (data[0].nb_fields - 1) if y_fields is None else y_fields
        self.x_fields = x_fields
        if self.x_fields is None:
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

        self.types = types
        self.batch_size = batch_size
        self.batch_order = []
        for idx_ds in range(len(self.data)):
            ds = self.data[idx_ds]
            for type in types:
                for idx_batch in range(len(ds.data[type])):
                    batch = ds.data[type][idx_batch]
                    count = math.ceil(len(batch)/batch_size)
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
        if self.x_converter is not None:
            x = self.x_converter(x)
        y = entries[:, self.y_fields]
        if self.y_converter is not None:
            y = self.y_converter(y)
        if self.y_remember is not None:
            self.y_remember.append(y)
        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # TODO SHUFFLE
        pass

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

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        """
        raise NotImplementedError("__data_generation not implemented")