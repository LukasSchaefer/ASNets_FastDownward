from keras import backend as K
from keras.engine.topology import Layer
from tensorflow.python.keras._impl.keras.layers import Lambda
import tensorflow as tf

from utils import broadcast_to

class SoftmaxOutputLayer(Layer):
    """
    computes softmax output policy for ASNets given a pair of two tensors:
    (scalar_action_values, action_activation_values)
    scalar_action_values: tensor with one scalar output value for each propositional action
        out of the last action layer
    action_activation_values: input tensor with one binary value for each propositional action
        indicating if the action is applicable (1) or not (0)
    """
        

    def __init__(self, **kwargs):
        super(SoftmaxOutputLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SoftmaxOutputLayer, self).build(input_shape)

    def call(self, x):
        scalar_action_values = x[0]
        action_activation_values = x[1]
        min_acts = K.min(scalar_action_values, axis=-1, keepdims=True)
        min_acts = broadcast_to(scalar_action_values, min_acts)
        disable_layer = Lambda(lambda cond, x, y: tf.where(cond, x, y), name='disabled_to_min')
        disable_layer.arguments = {'x': scalar_action_values, 'y': min_acts}
        # condition input left
        disabled_min_out = disable_layer(action_activation_values)

        # subtract out maximum for numeric stability
        max_acts = K.max(disabled_min_out, axis=-1, keepdims=True)
        max_acts = broadcast_to(scalar_action_values, max_acts)
        pad_acts = scalar_action_values - max_acts

        exps = action_activation_values * K.exp(pad_acts)

        # use uniform predictions when nothing is valid
        any_valid = K.any(action_activation_values, axis=-1, keepdims=True)
        any_valid = broadcast_to(scalar_action_values, any_valid)

        safe_exps_layer = Lambda(lambda cond, x, y: tf.where(cond, x, y), name='safe_exps')
        safe_exps_layer.arguments = {'x': exps, 'y': K.ones_like(exps)}
        # condition input left
        safe_exps = safe_exps_layer(any_valid)

        # now we can divide out by sums of exponentials
        sums = K.sum(safe_exps, axis=-1, keepdims=True)
        # this is just for safety
        clipped_sums = K.clip(sums, 1e-5, 1e10)
        output = safe_exps / clipped_sums
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0]
