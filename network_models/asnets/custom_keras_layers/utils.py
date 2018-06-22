"""
Utility tools for tensorflow needed for ASNets
(mostly taken from https://github.com/qxcv/asnets/blob/master/deepfpg/tf_utils.py)
"""

import tensorflow as tf

def broadcast_to(pattern, array):
    """Broacast ``array`` to match shape of ``pattern``."""
    pat_shape = tf.shape(pattern)
    arr_shape = tf.shape(array)
    multiples = tf.floordiv(pat_shape, arr_shape)
    rv = tf.tile(array, multiples)
    return rv