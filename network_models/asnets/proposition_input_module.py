from keras import backend as K
from keras.engine.topology import Layer
from tensorflow.python.keras._impl.keras.layers import Lambda, GlobalMaxPooling1D, concatenate

class PropositionInputModule(Layer):
    """
    computes input tensor for proposition module of proposition by concatenating
    the correct output vectors of last layer action modules (input) and potentially pooling
    vectors
    """
        

    def __init__(self, proposition, related_propoposition_action_ids, hidden_representation_size, **kwargs):
        super(PropositionInputModule, self).__init__(**kwargs)
        self.proposition = proposition
        self.related_propoposition_action_ids = related_propoposition_action_ids
        self.hidden_representation_size = hidden_representation_size

    def build(self, input_shape):
        super(PropositionInputModule, self).build(input_shape)

    def call(self, x):
        # collect outputs of related action modules in last layer and pool all outputs together of
        # action modules of the same underlying action schema
        pooled_related_outputs = []
        for action_schema_list in self.related_propoposition_action_ids:
            action_schema_outputs = []
            get_index_of_tensor_layer = Lambda(lambda x, index: x[:, self.hidden_representation_size * index:\
                self.hidden_representation_size * (index + 1)])
            for index in action_schema_list:
                get_index_of_tensor_layer.arguments = {'index': index}
                action_schema_outputs.append(get_index_of_tensor_layer(x))
            # concatenate outputs
            if not action_schema_outputs:
                # There were no related propositional actions of the corresponding action schema
                # -> create hidden-representation-sized vector of 0s
                zeros = K.zeros((self.hidden_representation_size,))
                zeros = Lambda(lambda x: K.expand_dims(x, 0))(zeros)
                pooled_related_outputs.append(zeros)
            else:
                concatenated_output = concatenate(action_schema_outputs, 0)
                # Pool all those output-vectors together to a single output-vector sized vector
                # (Not sure if this is what I am doing here)
                # expand dim to 3D is needed for GlobalMaxPooling
                concatenated_output = Lambda(lambda x: K.expand_dims(x, 0))(concatenated_output)
                pooled_output = GlobalMaxPooling1D()(concatenated_output)
                pooled_related_outputs.append(pooled_output)

        # concatenate all pooled related output tensors to new input tensor for module
        return concatenate(pooled_related_outputs)

    def compute_output_shape(self, input_shape):
        num_related_action_schemas = len(self.related_propoposition_action_ids)
        return (input_shape[0], num_related_action_schemas * self.hidden_representation_size)
