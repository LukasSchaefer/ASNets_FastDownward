"""
Utility tools for tensorflow needed for ASNets
(taken from https://github.com/qxcv/asnets/blob/master/deepfpg/tf_utils.py)
"""

from keras import backend as K

def broadcast_to(pattern, array):
    """Broacast ``array`` to match shape of ``pattern``."""
    pat_shape = pattern.shape
    arr_shape = array.shape
    multiples = []
    for x, y in pat_shape, arr_shape:
        multiples.append(x / y)
    rv = K.tile(array, multiples)
    return rv
    # with tf.name_scope('broadcast_to'):
    #     pat_shape = tf.shape(pattern)
    #     arr_shape = tf.shape(array)
    #     multiples = tf.floordiv(pat_shape, arr_shape)
    #     pos_assert = tf.Assert(
    #         tf.reduce_all(multiples > 0), [multiples, pat_shape, arr_shape],
    #         name='pos_assert')
    #     with tf.control_dependencies([pos_assert]):
    #         rv = tf.tile(array, multiples)
    #         rv_shape = tf.shape(rv)
    #         shape_assert = tf.Assert(
    #             tf.reduce_all(pat_shape == rv_shape), [pat_shape, rv_shape],
    #             name='shape_assert')
    #         with tf.control_dependencies([shape_assert]):
    #             return rv


# def masked_softmax(activations, mask):
#     """
#     computes masked softmax function used in ASNets for the prediction
# 
#     :param activations: tensor of (scalar) activations for each propositional action
#         in order of self.problem_meta.propositional_actions
#     :param mask: input-tensor including one binary value (0/ 1) for every propositional action
#         indicating if the action is applicable in the current state (same order again)
#     :return: output tensor of masked softmax function representing the network's policy
#         (ith value = probability to choose ith propositional action)
#     """
#     with tf.name_scope('masked_softmax'):
#         eq_size_op = tf.assert_equal(
#             tf.shape(activations),
#             tf.shape(mask),
#             message='activation and mask shape differ')
#         with tf.control_dependencies([eq_size_op]):
#             mask = tf.not_equal(mask, 0)
#         # set all activations for disabled things to have minimum value
#         min_acts = tf.reduce_min(
#             activations, reduction_indices=[-1], keep_dims=True)
#         min_acts = broadcast_to(activations, min_acts)
#         disabled_min_out = tf.where(
#             mask, activations, min_acts, name='disabled_to_min')
#         # subtract out maximum for numeric stability
#         max_acts = tf.reduce_max(
#             disabled_min_out, reduction_indices=[-1], keep_dims=True)
#         max_acts = broadcast_to(activations, max_acts)
#         pad_acts = activations - max_acts
#         exps = tf.cast(mask, tf.float32) * tf.exp(pad_acts, name='masked_exps')
#         # use uniform predictions when nothing is valid
#         any_valid = tf.reduce_any(
#             mask, reduction_indices=[-1], keep_dims=True, name='any_valid')
#         any_valid = broadcast_to(activations, any_valid)
#         # signature: tf.where(switch expr, if true, if false)
#         safe_exps = tf.where(
#             any_valid, exps, tf.ones_like(exps), name='safe_exps')
# 
#         # now we can divide out by sums of exponentials
#         sums = tf.reduce_sum(
#             safe_exps, reduction_indices=-1, keep_dims=True, name='sums')
#         # this is just for safety
#         clipped_sums = tf.clip_by_value(sums, 1e-5, 1e10)
#         output = safe_exps / clipped_sums
# 
#     return output