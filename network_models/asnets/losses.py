"""
Loss functions for ASNets
"""
from keras import backend as K


def custom_binary_crossentropy(y_true, y_pred):
    """
    custom binary crossentropy loss-function for Action Schema Networks
    inspired by Action Schema Networks: Generalised Policies with Deep Learning
    (https://arxiv.org/abs/1709.04271)

    :param y_true: opt-value (binary value 0 or 1) for each action indicating whether the action starts an optimal
                   plan according to the teacher policy
    :param y_pred: prediction = probabilities of the network to choose each action for
                   all actions 
    """
    ones = K.ones(K.shape(y_true))
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    out = -(y_true * K.log(y_pred) + (ones - y_true) * K.log(ones - y_pred))
    return K.sum(out, axis=-1)