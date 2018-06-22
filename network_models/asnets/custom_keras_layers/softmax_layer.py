import sys
from utils import broadcast_to
from keras import backend as K
from keras.engine.topology import Layer
from tensorflow.python.keras._impl.keras.layers import Lambda
import tensorflow as tf

class SoftmaxOutputLayer(Layer):
    """custom keras layer to implement the custom ASNet masked softmax output used for prediction"""

    def __init__(self, number_prop_actions, **kwargs):
        self.number_prop_actions = number_prop_actions
        super(SoftmaxOutputLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SoftmaxOutputLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        """
        inputs = [activations, mask]:
            activations: tensor of (scalar) activations for each propositional action
                        in order of self.problem_meta.propositional_actions
            mask: input-tensor including one binary value (0/ 1) for every propositional action
                indicating if the action is applicable in the current state (same order again)
        return:
            output tensor of masked softmax function representing the network's policy
            (ith value = probability to choose ith propositional action)
        """
        return inputs
        # if type(inputs) is not list or len(inputs) != 2:
        #     raise Exception('SoftmaxOutputLayer must be called on a list of two tensors '
        #                     '(activations, mask). Got: ' + str(inputs))
        # activations = inputs[0]
        # mask = inputs[1]

        # mask_bool = K.cast(mask, 'bool')
        # min_acts = K.min(activations, axis=-1, keepdims=True)
        # min_acts = broadcast_to(activations, min_acts)
        # disabled_min_out = tf.where(mask_bool, activations, min_acts)

        # # subtract out maximum for numeric stability
        # max_acts = K.max(disabled_min_out, axis=-1, keepdims=True)
        # max_acts = broadcast_to(activations, max_acts)
        # pad_acts = activations -  max_acts

        # exps = mask * K.exp(pad_acts)

        # # use uniform predictions when nothing is valid
        # any_valid = K.any(mask, axis=-1, keepdims=True)
        # any_valid = broadcast_to(activations, any_valid)

        # # condition input left
        # ones_like = K.ones_like(exps)
        # safe_exps = tf.where(any_valid, exps, ones_like)

        # # now we can divide out by sums of exponentials
        # sums = K.sum(safe_exps, axis=-1, keepdims=True)
        # # this is just for safety
        # clipped_sums = K.clip(sums, 1e-5, 1e10)
        # self.output = sums / clipped_sums
        # return self.output

    def compute_output_shape(self, input_shape):
        return input_shape
        # batch_size = input_shape[0][0]
        # return (batch_size, self.number_prop_actions)