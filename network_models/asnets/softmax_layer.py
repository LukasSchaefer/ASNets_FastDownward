from utils import broadcast_to
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf

class SoftmaxOutputLayer(Layer):
    """
    custom keras layer to implement custom ASNet masked softmax output used
    for the prediction
    """

    def __init(self, **kwargs):
        super(SoftmaxOutputLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SoftmaxOutputLayer, self).build(input_shape)

    def call(self, inputs):
        """
        computes masked softmax function used in ASNets for the prediction

        inputs = [activations, mask]
        :param activations: tensor of (scalar) activations for each propositional action
            in order of self.problem_meta.propositional_actions
        :param mask: input-tensor including one binary value (0/ 1) for every propositional action
            indicating if the action is applicable in the current state (same order again)
        :return: output tensor of masked softmax function representing the network's policy
            (ith value = probability to choose ith propositional action)
        """
        activations = inputs[0]
        mask = inputs[1]
        min_acts = K.min(activations, axis=-1, keepdims=True)
        min_acts = broadcast_to(activations, min_acts)
        disabled_min_out = tf.where(K.cast(mask, 'bool'), activations, min_acts)

        # subtract out maximum for numeric stability
        max_acts = K.max(disabled_min_out, axis=-1, keepdims=True)
        max_acts = broadcast_to(activations, max_acts)
        pad_acts = activations - max_acts

        exps = mask * K.exp(pad_acts)

        # use uniform predictions when nothing is valid
        any_valid = K.any(mask, axis=-1, keepdims=True)
        any_valid = broadcast_to(activations, any_valid)

        ones_like =  K.ones_like(exps)
        safe_exps = tf.where(K.cast(any_valid, 'bool'), exps, ones_like)

        # now we can divide out by sums of exponentials
        sums = K.sum(safe_exps, axis=-1, keepdims=True)
        output = safe_exps / sums
        return output


    def compute_output_shape(self, input_shape):
        return input_shape[0]
