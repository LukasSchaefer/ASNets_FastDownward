"""
Utility tools for tensorflow needed for ASNets
(broadcast and softmax mostly taken from https://github.com/qxcv/asnets/blob/master/deepfpg/tf_utils.py
and wrapped in keras lambda layers)
"""

import tensorflow as tf
from keras import backend as K
from tensorflow.python.keras._impl.keras.layers import Lambda

def broadcast_to(pattern, array):
    """Broacast ``array`` to match shape of ``pattern``."""
    shape_layer = Lambda(lambda x: K.shape(x))
    pat_shape = shape_layer(pattern)
    arr_shape = shape_layer(array)
    floordiv_layer = Lambda(lambda x: tf.floordiv(x[0], x[1]))
    multiples = floordiv_layer([pat_shape, arr_shape])
    rv = Lambda(lambda x: K.tile(x[0], x[1]))([array, multiples])
    return rv


def masked_softmax(activations, mask):
    """
    computes masked softmax function used in ASNets for the prediction

    :param activations: tensor of (scalar) activations for each propositional action
        in order of self.problem_meta.propositional_actions
    :param mask: input-tensor including one binary value (0/ 1) for every propositional action
        indicating if the action is applicable in the current state (same order again)
    :return: output tensor of masked softmax function representing the network's policy
        (ith value = probability to choose ith propositional action)
    """
    mask_bool = Lambda(lambda x: K.cast(x, 'bool'), name='softmax_bool_cast_mask')(mask)
    min_acts = Lambda(lambda x: K.min(x, axis=-1, keepdims=True), name='softmax_min_acts')(activations)
    min_acts = broadcast_to(activations, min_acts)
    disable_layer = Lambda(lambda x: tf.where(x[0], x[1], x[2]), name='softmax_disabled_to_min')
    # condition input left
    disabled_min_out = disable_layer([mask_bool, activations, min_acts])

    # subtract out maximum for numeric stability
    max_acts = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(disabled_min_out)
    max_acts = broadcast_to(activations, max_acts)
    subtract_max_layer = Lambda(lambda x: x[0] - x[1], name='softmax_subtract_max_values')
    pad_acts = subtract_max_layer([activations, max_acts])

    exp_layer = Lambda(lambda x: x[0] * K.exp(x[1]), name='softmax_exps')
    exps = exp_layer([mask, pad_acts])

    # use uniform predictions when nothing is valid
    any_valid = Lambda(lambda x: K.any(x, axis=-1, keepdims=True), name='softmax_any_valid')(mask)
    any_valid = broadcast_to(activations, any_valid)

    safe_exps_layer = Lambda(lambda x: tf.where(x[0], x[1], x[2]), name='safe_exps')
    # condition input left
    ones_like = Lambda(lambda x: K.ones_like(x), name='softmax_ones_like')(exps)
    safe_exps = safe_exps_layer([any_valid, exps, ones_like])

    # now we can divide out by sums of exponentials
    sums = Lambda(lambda x: K.sum(x, axis=-1, keepdims=True), name='softmax_sums')(safe_exps)
    div_output_layer = Lambda(lambda x: x[0] / x[1], name='softmax_div_output')
    output = div_output_layer([safe_exps, sums])
    return output


def custom_binary_crossentropy(y_true, y_pred):
    """
    custom binary crossentropy loss-function out of Action Schema Networks: Generalised Policies
    with Deep Learning (https://arxiv.org/abs/1709.04271)

    :param y_true: opt-value for each action indicating whether the action starts an optimal
                   plan according to the teacher policy
    :param y_pred: prediction = probabilities of the network to choose each action for
                   all actions 
    """
    y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
    out = ((1.0 - y_true) * K.log(1.0 - y_pred)) + (y_true * K.log(y_pred))
    return K.mean(out, axis=-1)