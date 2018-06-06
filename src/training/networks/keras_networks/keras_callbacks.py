import keras
from keras.callbacks import Callback, ModelCheckpoint

from ... import misc
from ... import parser
from ... import parser_tools as parset
from ... import AbstractBaseClass


from ...parser_tools import main_register, ArgumentException

import abc
import os

class BaseKerasCallback(Callback):
    arguments = parset.ClassArguments("BaseKerasCallback", None,
                                      ('id', True, None, str))

    def __init__(self, id=None):
        Callback.__init__(self)
        self._id = id

    def setup(self, network, *args, **kwargs):
        """
        Prepares the Callback function before it is used for training. If using
        this callback objects for multiple trainings, the setup is called before
        each of them. It has the same purpose as on_training_begin of keras,
        but allows us to feed arguments.
        :param network: KerasNetwork object for which this callback shall be used
        :return:
        """
        pass


    def finalize(self, network, *args, **kwargs):
        """

        :param network: instance of the KerasNetwork which used this callback
                        during training
        :param args:
        :param kwargs:
        :return:
        """
        pass


    """DEFAULTS FOR SPECIAL METHODS THE DIFFERENT CALLBACKS NEED"""
    def shall_reinitialize(self):
        return False

    @staticmethod
    def parse(tree, item_cache):
        obj = parser.try_lookup_obj(tree, item_cache, BaseKerasCallback, None)
        if obj is not None:
            return obj
        else:
            raise ArgumentException("The definition of the base network can "
                                    "only be used for look up of any previously"
                                    " defined schema via 'Sampler(id=ID)'")


main_register.append_register(BaseKerasCallback, "keras_callback")


class KerasCallbackWrapper(BaseKerasCallback, AbstractBaseClass):
    arguments = parset.ClassArguments(
        'KerasCallbackWrapper', BaseKerasCallback.arguments)

    def __init__(self, id=None):
        self._internal_callback = None
        BaseKerasCallback.__init__(self, id=id)



    def setup(self, network, *args, **kwargs):
        self._setup_internal_checkpoint(*args, **kwargs)

    @abc.abstractmethod
    def _setup_internal_checkpoint(self, *args, **kwargs):
        pass

    """WRAPPING"""

    def _wrap_var_get_model(self):
        return self._internal_callback.model

    def _wrap_var_set_model(self, value):
        if self._internal_callback is not None:
            self._internal_callback.model = value
    model = property(_wrap_var_get_model, _wrap_var_set_model)

    def _wrap_var_get_val_data(self):
        return self._internal_callback.validation_data

    def _wrap_var_set_val_data(self, value):
        if self._internal_callback is not None:
            self._internal_callback.validation_data = value
    validation_data = property(_wrap_var_get_val_data, _wrap_var_set_val_data)


    def set_params(self, params):
        self._internal_callback.set_params(params)

    def set_model(self, model):
        self._internal_callback.set_model(model)

    def on_epoch_begin(self, epoch, logs=None):
        self._internal_callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        self._internal_callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        self._internal_callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        self._internal_callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        self._internal_callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        self._internal_callback.on_train_end(logs)


    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  KerasCallbackWrapper)


main_register.append_register(KerasCallbackWrapper, "keras_callback_wrapper")



class KerasProgressChecking(BaseKerasCallback):
    arguments = parset.ClassArguments(
        'KerasProgressChecking', BaseKerasCallback.arguments,
        ("monitor", False, None, str, "Metric of the network to monitor"),
        ("epochs", False, None, int, "After how many epochs the check is done"),
        ("threshold", False, None, float, "Threshold value to satisfy"),
        ("restarts", True, -1, int, "How often the training may be restarted "
                                    "according to this callback"),
        ("minimize", True, True, parser.convert_bool, "True = minimize metric else maximiaze"),
        order=["monitor", "epochs", "threshold", "restarts", "minimize", "id"])

    def __init__(self, monitor, epochs, threshold, restarts=-1, minimize=True,
                 id=None):
        BaseKerasCallback.__init__(self, id=id)
        self._monitor = monitor
        self._epochs = epochs
        self._threshold = threshold
        self._restarts = restarts
        self._current_round = -1
        self._minimize = minimize
        self._active = True
        self._failed = False

    def _get_failed(self):
        return self._failed
    failed = property(_get_failed)

    def shall_reinitialize(self):
        return self._failed and (self._restarts == -1
                                 or self._restarts > self._current_round)

    def on_train_begin(self, logs={}):
        self._active = True
        self._failed = False
        self._current_round += 1

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
                self._failed = True

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  KerasProgressChecking)


main_register.append_register(KerasProgressChecking, "keras_progress_checking")




class KerasModelCheckpoint(KerasCallbackWrapper):
    arguments = parset.ClassArguments(
        'KerasModelCheckpoint', KerasCallbackWrapper.arguments,
        ("monitor", False, None, str, "Metric of the network to monitor"),
        ("filepath", True, None, str, "Path to the temporary file for the model checkpoint (or None)"),
        ("mode", True, "auto", str, "Mode of the ModelCheckpoint Callback"),
        ("period", True, 1, int, "In which periods the model shall be checked"),
        ("save_best_only", True, True, "Store only the best network"),
        ("verbose", True, 0, int, "Verbosity level"),
        order=["monitor", "filepath", "mode", "period", "save_best_only",
               "verbose", "id"])

    def __init__(self, monitor, filepath=None, mode="auto", period=1,
                 save_best_only=True, verbose=0, id=None):
        KerasCallbackWrapper.__init__(self, id=id)
        self._monitor = monitor
        self._filepath = filepath
        self._mode = mode
        self._period = period
        self._save_best_only = save_best_only
        self._verbose = verbose

    def setup(self, network, *args, **kwargs):
        if self._filepath is None:
            self._filepath = (network.get_store_path() + "."
                              + str(misc.get_rnd_suffix()) + ".tmp")

        KerasCallbackWrapper.setup(self, network, *args, **kwargs)

    def _setup_internal_checkpoint(self, *args, **kwargs):
        self._internal_callback = ModelCheckpoint(
            filepath=self._filepath, monitor=self._monitor, verbose=self._verbose,
            save_best_only=self._save_best_only, save_weights_only=False, mode=self._mode,
            period=self._period)

    def finalize(self, network, *args, **kwargs):
        network._model = keras.models.load_model(self._filepath)
        os.remove(self._filepath)

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  KerasModelCheckpoint)


main_register.append_register(KerasModelCheckpoint, "keras_model_checkpoint")


