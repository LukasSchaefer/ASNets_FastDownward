from keras import backend as K

from keras import activations
from keras import initializers
from keras import regularizers

from keras.engine.topology import Layer
from keras.layers import concatenate, GlobalMaxPooling1D, Dropout


class PropositionInputLayer(Layer):
    """
    custom keras layer to compute the input for a ASNet proposition module
    """

    def __init__(self,
                 hidden_representation_size,
                 related_propositional_action_ids,
                 **kwargs):
        """
        :param hidden_representation_size: hidden representation size used by every
            module (= size of module outputs)
        :param related_propositional_action_ids: list of nested lists
            each nested list corresponds to one action schema related to the underlying predicate
            of proposition. All these actions whose ids are in the nested list are related to proposition
        """
        self.hidden_representation_size = hidden_representation_size
        self.related_propositional_action_ids = related_propositional_action_ids
        super(PropositionInputLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        super(PropositionInputLayer, self).build(input_shape)


    def call(self, inputs):
        """
        :param inputs: concatenation of all action module outputs of the last action layer
        :return: concatenation of the outputs of the related actions
                 Thereby, actions with the same unterlying action schema (grouped in one nested list
                 in related_action_ids) are pooled together. If there are no such actions for one
                 nested list, then use a zero tensor of the necessary size
        """
        # collect outputs of related action modules in last layer and pool all outputs together of
        # action modules of the same underlying action schema
        pooled_related_outputs = []
        for action_schema_list in self.related_propositional_action_ids:
            # collect outputs of all actions in the nested action_schema_list
            action_schema_outputs = []
            for index in action_schema_list:
                action_schema_outputs.append(inputs[:, self.hidden_representation_size * index: self.hidden_representation_size * (index + 1)])
            # concatenate outputs
            if not action_schema_outputs:
                # There were no related propositional actions of the corresponding action schema
                # -> create hidden-representation-sized vector of 0s
                # inputs.shape[0] matches batch_size
                shape_like_tensor = inputs[:, 0: self.hidden_representation_size]
                zeros = K.zeros_like(shape_like_tensor)
                pooled_related_outputs.append(zeros)
            else:
                if len(action_schema_outputs) > 1:
                    concat_tensors = []
                    for tensor in action_schema_outputs:
                        # expand tensor dim for global max pooling afterwards (shrinks second dimension)
                        concat_tensors.append(K.expand_dims(tensor, 1))
                    # concatenate those expanded tensors along the second axis ("besides" for pooling these together)
                    concatenated_output = concatenate(concat_tensors, 1)
                    # apply global max pooling
                    pooled_output = GlobalMaxPooling1D()(concatenated_output)
                    pooled_related_outputs.append(pooled_output)
                else:
                    # there is only one action of the action schema outputs
                    pooled_related_outputs.append(action_schema_outputs[0])

        # concatenate all pooled related output tensors to new input tensor for module
        if len(pooled_related_outputs) > 1:
            return concatenate(pooled_related_outputs)
        else:
            return pooled_related_outputs[0]


    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        return (input_shape[0], len(self.related_propositional_action_ids) * self.hidden_representation_size)


class PropositionModuleLayer(Layer):
    """
    custom keras layer to implement ASNet proposition module
    """

    def __init__(self,
                 hidden_representation_size,
                 activation,
                 dropout,
                 kernel_initializer,
                 bias_initializer,
                 regularizer_value,
                 **kwargs):
        """
        :param hidden_representation_size: hidden representation size used by every
            module (= size of module outputs)
        :param activation: name of activation function to be used in all modules
            of all layers but the last output layer
        :param dropout rate used in every intermediate node
        :param kernel_initializer: initializer to be used for all weight matrices/ kernels
            of all modules
        :param bias_initializer: initializer to be used for all bias vectors
            of all modules
        :param regularizer_value: value used for all L2 regularizations applied
            to all weights (-matrices and bias vectors!)
        """
        self.hidden_representation_size = hidden_representation_size
        self.activation = activations.get(activation)
        self.dropout = dropout
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.regularizer_value = regularizer_value
        super(PropositionModuleLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.hidden_representation_size),
                                      initializer=self.kernel_initializer,
                                      regularizer=regularizers.l2(self.regularizer_value),
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.hidden_representation_size,),
                                    initializer=self.bias_initializer,
                                    regularizer=regularizers.l2(self.regularizer_value),
                                    trainable=True)
        super(PropositionModuleLayer, self).build(input_shape)


    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        output = K.bias_add(output, self.bias)
        output = self.activation(output)
        if self.dropout:
            return Dropout(self.dropout)(output)
        return output


    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        return (input_shape[0], self.hidden_representation_size)