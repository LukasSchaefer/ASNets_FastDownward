from keras import backend as K
from keras.engine.topology import Layer
from tensorflow.python.keras._impl.keras.layers import Lambda, concatenate

class ActionInputModule(Layer):
    """
    computes input tensor for action module of propositional action by concatenating
    the correct output vectors of last layer proposition modules (input)
    """
        

    def __init__(self, propositional_action, related_proposition_ids, hidden_representation_size, **kwargs):
        self.propositional_action = propositional_action
        self.related_proposition_ids = related_proposition_ids
        self.hidden_representation_size = hidden_representation_size
        super(ActionInputModule, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ActionInputModule, self).build(input_shape)

    def call(self, x, mask=None):
        # collect outputs of related proposition modules in last layer
        get_index_of_tensor_layer = Lambda(lambda x, index: x[:, self.hidden_representation_size * index:\
            self.hidden_representation_size * (index + 1)])
        related_outputs = []
        for index in self.related_proposition_ids:
            get_index_of_tensor_layer.arguments = {'index': index}
            related_outputs.append(get_index_of_tensor_layer(x))
        # concatenate related output tensors to new input tensor
        return concatenate(related_outputs)

    def compute_output_shape(self, input_shape):
        num_related_propositions = len(self.related_proposition_ids)
        return (input_shape[0], num_related_propositions * self.hidden_representation_size)
