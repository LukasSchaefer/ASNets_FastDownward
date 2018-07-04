from keras import backend as K

from keras import activations
from keras import initializers
from keras import regularizers

from keras.engine.topology import Layer
from keras.layers import concatenate, Dropout

class FirstActionInputLayer(Layer):
    """
    custom keras layer to compute input values for ASNet action modules in the first layer
    """

    def __init__(self,
                 sas_task,
                 action_index,
                 related_proposition_ids,
                 related_proposition_names,
                 extra_input_size,
                 **kwargs):
        """
        :param sas_task: SAS task from the problem (used to find out values for pruned variables)
        :param action_index: index of the propositional action in problem_meta
        :param related_proposition_ids: ids (= indeces in self.problem_meta.grounded_predicates)
            of propositions related to propositional action
        :param related_proposition_names: names of propositions related to propositional action
        :param extra_input_size: size of additional input features (per action) or 0 if there are none
        """
        self.sas_task = sas_task
        self.action_index = action_index
        self.related_proposition_ids = related_proposition_ids
        self.related_proposition_names = related_proposition_names
        self.extra_input_size = extra_input_size
        super(FirstActionInputLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        super(FirstActionInputLayer, self).build(input_shape)


    def call(self, inputs):
        """
        extracts the necessary input values for action module for propositional action out of the
        input values
        :param inputs = [proposition_truth_values, proposition_goal_values, action_applicable_values(, additional_input_features)]
        """
        if type(inputs) is not list:
            raise ValueError("FirstActionInputLayer needs a list of the network inputs as the input")

        if self.extra_input_size:
            assert len(inputs) == 4
        else:
            assert len(inputs) == 3

        proposition_truth_values = inputs[0]
        proposition_goal_values = inputs[1]
        action_applicable_values = inputs[2]
        if self.extra_input_size:
            additional_input_features = inputs[3]

        # build input list
        input_list = []
        # add truth values of related propositions
        for related_index, prop_index in enumerate(self.related_proposition_ids):
            if prop_index == -1:
                # related proposition was pruned -> dummy tensor zeros/ ones 
                # get name and check if it is initially true or false
                prop_value = False
                prop_name = self.related_proposition_names[related_index]
                task_init = self.sas_task.init
                for var, val in enumerate(task_init.values):
                    fact_name = self.sas_task.variables.value_names[var][val]
                    if fact_name.startswith("Negated"):
                        fact_name = fact_name[7:]
                    if fact_name == prop_name:
                        prop_value = True
                        break
                # get random truth input tensor for shape
                shape_like_tensor = proposition_truth_values[:, 0: 1]
                if prop_value:
                    replace_tensor = K.zeros_like(shape_like_tensor)
                else:
                    replace_tensor = K.ones_like(shape_like_tensor)
                input_list.append(replace_tensor)
            else:
                input_list.append(proposition_truth_values[:, prop_index: prop_index + 1])

        # add goal values of related propositions
        for prop_index in self.related_proposition_ids:
            if prop_index == -1:
                # related proposition was pruned -> dummy tensor zeros 
                # get name and check if it is in the goal or not
                prop_value = False
                prop_name = self.related_proposition_names[related_index]
                task_goal = self.sas_task.goal
                for var, val in task_goal.pairs:
                    fact_name = self.sas_task.variables.value_names[var][val]
                    if fact_name.startswith("Negated"):
                        fact_name = fact_name[7:]
                    if fact_name == prop_name:
                        prop_value = True
                        break
                # get random goal input tensor for shape
                shape_like_tensor = proposition_goal_values[:, 0:1]
                if prop_value:
                    replace_tensor = K.zeros_like(shape_like_tensor)
                else:
                    replace_tensor = K.ones_like(shape_like_tensor)
                input_list.append(replace_tensor)
            else:
                input_list.append(proposition_goal_values[:, prop_index: prop_index + 1])

        # add value indicating if propositional action is currently applicable
        input_list.append(action_applicable_values[:, self.action_index : self.action_index + 1])

        # add additional input features if existing
        if self.extra_input_size:
            start_index = self.extra_input_size * self.action_index
            end_index = (self.extra_input_size + 1) * self.action_index
            input_list.append(additional_input_features[:, start_index : end_index])

        # convert input list to tensor
        return concatenate(input_list)


    def compute_output_shape(self, input_shape):
        num_related_props = len(self.related_proposition_ids)
        if self.extra_input_size:
            # truth + goal value for each related proposition + one applicable value
            output_dim = num_related_props * 2 + 1
        else:
            # additionally extra_input_size input features for the action
            output_dim = num_related_props * 2 + 1 + self.extra_input_size
        return (input_shape[0][0], output_dim)


class IntermediateActionInputLayer(Layer):
    """
    custom keras layer to compute input values for ASNet action modules in intermediate layers
    """

    def __init__(self,
                 hidden_representation_size,
                 action_index,
                 related_proposition_ids,
                 **kwargs):
        """
        :param hidden_representation_size: hidden representation size used by every
            module (= size of module outputs)
        :param action_index: index of the propositional action in problem_meta
        :param related_proposition_ids: ids (= indeces in self.problem_meta.grounded_predicates)
            of propositions related to propositional action
        """
        self.hidden_representation_size = hidden_representation_size
        self.action_index = action_index
        self.related_proposition_ids = related_proposition_ids
        super(IntermediateActionInputLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        super(IntermediateActionInputLayer, self).build(input_shape)


    def call(self, inputs):
        """
        extracts the necessary input values for action module for propositional action out of the
        concatenated output values of the last proposition layer
        :param inputs: inputs = last_layer_proposition_module_outputs
        """
        # collect outputs of related proposition modules in last layer
        related_outputs = []
        for index in self.related_proposition_ids:
            if index == -1:
                # related prop was pruned -> add zeros
                # get random module output tensor for shape
                shape_like_tensor = inputs[:, 0 : self.hidden_representation_size]
                zeros = K.zeros_like(shape_like_tensor)
                related_outputs.append(zeros)
            else:
                related_outputs.append(inputs[:, self.hidden_representation_size * index : self.hidden_representation_size * (index + 1)])

        # concatenate related output tensors to new input tensor
        if len(related_outputs) > 1:
            return concatenate(related_outputs)
        else:
            return related_outputs[0]
        

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        output_dim = self.hidden_representation_size * len(self.related_proposition_ids)
        return (input_shape[0], output_dim)


class ActionModuleLayer(Layer):
    """
    custom keras layer to implement ASNet action module
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
        super(ActionModuleLayer, self).__init__(**kwargs)

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
        super(ActionModuleLayer, self).build(input_shape)

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