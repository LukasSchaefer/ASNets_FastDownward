import os
import sys
from tensorflow.python.keras._impl.keras.models import Model
from tensorflow.python.keras._impl.keras.layers import Input, Dense, Dropout, Lambda, concatenate, GlobalMaxPooling1D
from keras import backend as K

import problem_meta
from utils import masked_softmax


class ASNet_Model_Builder():
    """responsible for building the keras model of Action Schema Networks"""

    def __init__(self, problem_meta):
        """
        :param problem_meta: ASNet specific meta informations about the underlying
            PDDL task
        """
        self.problem_meta = problem_meta


    def __make_action_module(self, action, layer_index):
        """
        constructs and returns one action module for all propositional actions with
        underlying action schema action in layer layer_index

        :param action: action schema this module is built for
        :param layer_index: index number of the layer this module is built for
        :return: keras layer representing the module
        """
        num_related_predicates = len(self.problem_meta.action_to_related_pred_names[action])
        if layer_index == 0:
            mod_name = 'first_layer_actmod_' + action.name
            # num_related_predicates truth values and goal values -> 2*
            # + 1 for value indicating if action is applicable
            # + extra_dimension for every module in first layer
            input_shape = (2 * num_related_predicates + 1 + self.extra_input_size,)
        else:
            mod_name = ('%d_layer_actmod_' % layer_index) + action.name
            # for every related proposition, one hidden representation vector as input
            # of the last layer
            input_shape = (num_related_predicates * self.hidden_representation_size,)
        return Dense(units=self.hidden_representation_size,
                     input_shape=input_shape,
                     activation=self.activation,
                     use_bias=True,
                     kernel_initializer=self.kernel_initializer,
                     bias_initializer=self.bias_initializer,
                     name=mod_name)


    def __make_final_action_layer_module(self, action):
                                         
        """
        constructs and returns one action module for all propositional actions with
        underlying action schema action in the final output layer
        MASKED SOFTMAX NOT INCLUDED -> HAS TO BE DONE AFTER

        :param action: action schema this module is built for
        :return: keras layer representing the module of the final output layer
                 DOES NOT COMPUTES THE MASKED SOFTMAX YET
        """
        num_related_predicates = len(self.problem_meta.action_to_related_pred_names[action])
        mod_name = 'last_layer_actmod_' + action.name
        # for every related proposition, one hidden representation vector as input
        # of the last layer
        input_shape = (num_related_predicates * self.hidden_representation_size,)
        # scalar output with identity activation function
        return Dense(units=1,
                     input_shape=input_shape,
                     activation='linear',
                     use_bias=True,
                     kernel_initializer=self.kernel_initializer,
                     bias_initializer=self.bias_initializer,
                     name=mod_name)


    def __make_predicate_module(self, predicate, layer_index):
        """
        constructs and returns one proposition module for all propositions with
        underlying predicate predicate in layer layer_index

        :param predicate: predicate this module is built for
        :param layer_index: index number of the layer this module is built for
        :return: keras layer representing the module
        """
        num_related_action_schemas = len(self.problem_meta.pred_to_related_action_names[predicate])
        mod_name = ('%d_layer_propmod_' % layer_index) + predicate.name
        # one hidden representation sized input vector
        input_shape = (num_related_action_schemas * self.hidden_representation_size,)
        return Dense(units=self.hidden_representation_size,
                     input_shape=input_shape,
                     activation=self.activation,
                     use_bias=True,
                     kernel_initializer=self.kernel_initializer,
                     bias_initializer=self.bias_initializer,
                     name=mod_name)


    def __make_action_module_input(self, propositional_action, last_layer_proposition_module_outputs):
        """
        computes input for action module of propositional action

        :param propositional_action: propositional action the module corresponds to
        :param last_layer_proposition_module_outputs: output tensor of last proposition layer
        :return: input for action module of propositional_action
        """
        related_proposition_ids = self.problem_meta.prop_action_to_related_gr_pred_ids[propositional_action]
        # collect outputs of related proposition modules in last layer
        get_index_of_tensor_layer = Lambda(lambda x, index: x[:, self.hidden_representation_size * index:\
            self.hidden_representation_size * (index + 1)])
        related_outputs = []
        for index in related_proposition_ids:
            get_index_of_tensor_layer.arguments = {'index': index}
            related_outputs.append(get_index_of_tensor_layer(last_layer_proposition_module_outputs))
        # concatenate related output tensors to new input tensor
        if len(related_outputs) > 1:
            return concatenate(related_outputs)
        else:
            return related_outputs[0]


    def __make_proposition_module_input(self, proposition, last_action_module_outputs):
        """
        computes input for proposition module of proposition

        :param proposition: proposition the module corresponds to
        :param last_action_module_outputs: output tensor of last action layer
        :return: input for proposition module of proposition
        """
        related_propositional_action_ids = self.problem_meta.gr_pred_to_related_prop_action_ids[proposition]

        # collect outputs of related action modules in last layer and pool all outputs together of
        # action modules of the same underlying action schema
        pooled_related_outputs = []
        for action_schema_list in related_propositional_action_ids:
            action_schema_outputs = []
            get_index_of_tensor_layer = Lambda(lambda x, index: x[:, self.hidden_representation_size * index:\
                self.hidden_representation_size * (index + 1)])
            for index in action_schema_list:
                get_index_of_tensor_layer.arguments = {'index': index}
                action_schema_outputs.append(get_index_of_tensor_layer(last_action_module_outputs))
            # concatenate outputs
            if not action_schema_outputs:
                # There were no related propositional actions of the corresponding action schema
                # -> create hidden-representation-sized vector of 0s

                # TODO: maybe make this more logical/ easier understandable (any output of get_index_of_tensor_layer
                # will be correct shape (batch_size, hidden_representation_size))
                get_index_of_tensor_layer.arguments = {'index': 0}
                shape_like_tensor = get_index_of_tensor_layer(last_action_module_outputs)

                zeros = Lambda(lambda x: K.zeros_like(x))(shape_like_tensor)
                pooled_related_outputs.append(zeros)
            else:
                if len(action_schema_outputs) > 1:
                    concatenated_output = concatenate(action_schema_outputs, 0)
                else:
                    concatenated_output = action_schema_outputs[0]
                # Pool all those output-vectors together to a single output-vector sized vector
                # (Not sure if this is what I am doing here)
                # expand dim to 3D is needed for GlobalMaxPooling
                concatenated_output = Lambda(lambda x: K.expand_dims(x, 1))(concatenated_output)
                pooled_output = GlobalMaxPooling1D()(concatenated_output)
                pooled_related_outputs.append(pooled_output)

        # concatenate all pooled related output tensors to new input tensor for module
        if len(pooled_related_outputs) > 1:
            return concatenate(pooled_related_outputs)
        else:
            return pooled_related_outputs[0]


    def __make_modules(self):
        """
        builds all action and proposition modules based on the ungrounded
        abstract actions and predicates

        :returns: action_layers_modules, proposition_layers_modules
            with action_layers_modules being a list of dicts where the ith dict corresponds
            to the ith action layer's modules mapping from action schema names to the module.
            Similarly proposition_layers_modules is a list of dicts where the ith dict corresponds
            to the ith proposition layer's modules mapping from predicate names to the module
        """
        # list of dicts where the ith dict corresponds to the ith action layer's modules
        # mapping from (abstract) action schema names to the module
        action_layers_modules = []
        # list of dicts where the ith dict corresponds to the ith proposition layer's modules
        # mapping from (abstract) predicate names to the module
        proposition_layers_modules = []

        # create modules and put them in corresponding lists for all but last action layer
        for layer_index in range(self.num_layers):
            action_layer_modules = {}
            for action in self.problem_meta.task.actions:
                # create module for action in layer_index' layer
                action_layer_modules[action.name] = self.__make_action_module(action, layer_index)
            # complete dicts for each layer are put in action_layers_modules
            action_layers_modules.append(action_layer_modules)

            proposition_layer_modules = {}
            for predicate in self.problem_meta.task.predicates:
                # create module for predicate in layer_index' layer
                proposition_layer_modules[predicate.name] = self.__make_predicate_module(
                    predicate, layer_index)
            # complete dicts for each layer are put in proposition_layers_modules
            proposition_layers_modules.append(proposition_layer_modules)

        last_action_layer_modules = {}
        # create modules for last action layer
        for action in self.problem_meta.task.actions:
            last_action_layer_modules[action.name] = self.__make_final_action_layer_module(action)
        action_layers_modules.append(last_action_layer_modules)

        return action_layers_modules, proposition_layers_modules


    def __get_first_layer_action_module_output(self,
                                               propositional_action,
                                               action_index,
                                               related_proposition_ids,
                                               action_module):
        """
        computes output for action module in first layer of concrete propositional action

        :param propositional_action: propositional action the module corresponds to
        :param action_index: index of the propositional action in problem_meta
        :param related_proposition_ids: ids (= indeces in self.problem_meta.grounded_predicates)
            of propositions related to propositional action
        :param action_module: action module of underlying action schema in layer_index layer
        :return: output tensor of module
        """
        # build input list
        input_list = []
        slice_layer_name = "slice_layer_%s" % (propositional_action.name.strip('(').strip(')').replace(' ', '_'))
        slice_layer = Lambda(lambda x, start, end: x[:, start: end], name=slice_layer_name)
        # add truth values of related propositions
        for prop_index in related_proposition_ids:
            slice_layer.arguments = {'start': prop_index, 'end': prop_index + 1}
            truth_value = slice_layer(self.proposition_truth_values)
            input_list.append(truth_value)

        # add goal values of related propositions
        for prop_index in related_proposition_ids:
            slice_layer.arguments = {'start': prop_index, 'end': prop_index + 1}
            goal_value = slice_layer(self.proposition_goal_values)
            input_list.append(goal_value)

        # add value indicating if propositional action is currently applicable
        slice_layer.arguments = {'start': action_index, 'end': action_index + 1}
        input_list.append(slice_layer(self.action_applicable_values))

        # add additional input features if existing
        if self.additional_input_features:
            start_index = self.extra_input_size * action_index
            end_index = (self.extra_input_size + 1) * action_index
            slice_layer.arguments = {'start': start_index, 'end': end_index}
            additional_input_features = slice_layer(self.additional_input_features)
            input_list.append(additional_input_features)
        # convert input list to tensor
        concatenate_layer_name = "1st_layer_input_%s_concatenate" % (propositional_action.name.strip('(').strip(')').replace(' ', '_'))
        input_tensor = concatenate(input_list, name=concatenate_layer_name)
        
        output = action_module(input_tensor)
        if self.dropout:
            dropout_name = 'first_layer_actmod_' + propositional_action.get_underlying_action_name() + \
                str(action_index) + '_dropout'
            return Dropout(self.dropout, name=dropout_name)(output)
        else:
            return output


    def __get_intermediate_layer_action_module_output(self,
                                                      propositional_action,
                                                      action_index,
                                                      input_tensor,
                                                      action_module,
                                                      layer_index):
        """
        computes output tensor for action module in intermediate layer

        :param propositional_action: propositional action the module corresponds to
        :param action_index: index of the propositional action in problem_meta
        :param input_tensor: input for action module
        :param action_module: action module of underlying action schema in layer_index layer
        :param layer_index: index indicating in which layer this module output is computed
        :return: output tensor of module
        """
        output = action_module(input_tensor)
        if self.dropout:
            dropout_name = ('%d_layer_actmod_' % layer_index) + propositional_action.get_underlying_action_name() +\
                str(action_index) + '_dropout'
            return Dropout(self.dropout, name=dropout_name)(output)
        else:
            return output


    def __get_proposition_module_output(self,
                                        proposition,
                                        proposition_index,
                                        input_tensor,
                                        proposition_module,
                                        layer_index):
        """
        computes output for proposition module

        :param proposition: proposition the module corresponds to
        :param proposition_index: index of the proposition in problem_meta
        :param input_tensor: input for proposition module
        :param proposition_module: proposition module of underlying predicate in layer_index layer
        :param layer_index: index indicating in which layer this module output is computed
        :param last_action_module_outputs: tensor as concatenation of all output vectors of the action modules
            in the last action layer (in order of ids)
        :return: output tensor of module
        """
        output = proposition_module(input_tensor)
        if self.dropout:
            dropout_name = ('%d_layer_propmod_' % layer_index) + proposition.predicate + str(proposition_index) + '_dropout'
            return Dropout(self.dropout, name=dropout_name)(output)
        else:
            return output


    def __make_network(self,
                       action_layers_modules,
                       proposition_layers_modules):
        """
        build concrete ASNet with all connections with modules

        :param action_layers_modules: list of dicts where the ith dict corresponds to the
            ith action layer's modules mapping from action schema names to the module
        :param proposition_layers_modules: list of dicts where the ith dict corresponds to
            the ith proposition layer's modules mapping from predicate names to the module

        :return: action_layers_outputs, proposition_layers_outputs
            with action_layers_outputs being a list of tensors where the ith tensor
            is a concatenation of output tensors of all action modules in the ith layer
            (in the order of self.problem_meta.propositional_actions)
            Similarly proposition_layers_outputs is a list of tensors where the ith tensor
            is a concatenation of outputs tensors of all proposition modules in the ith layer
            (in the order of self.problem_meta.grounded_predicates)
        """
        action_layers_outputs = []
        proposition_layers_outputs = []

        # create concrete layers and put them in corresponding lists for all but last action
        # layer
        for layer_index in range(self.num_layers):
            # list of outputs of all action modules in layer_index layer
            action_layer_outputs = []
            for action_index, propositional_action in enumerate(self.problem_meta.propositional_actions):
                if layer_index == 0:
                    # list of ids = indeces of related propositions
                    related_proposition_ids = self.problem_meta.prop_action_to_related_gr_pred_ids[
                        propositional_action]

                    # extract corresponding action module for first layer
                    action_module = action_layers_modules[0][propositional_action.get_underlying_action_name()]
                    # compute output of action module and add to the list for first layer
                    action_layer_outputs.append(self.__get_first_layer_action_module_output(
                        propositional_action, action_index, related_proposition_ids, action_module))
                else:
                    # extract corresponding action input module
                    input_tensor = self.__make_action_module_input(propositional_action, proposition_layers_outputs[layer_index - 1])
                    # extract corresponding action module for layer_index layer
                    action_module = action_layers_modules[layer_index][
                        propositional_action.get_underlying_action_name()]
                    # compute output of action module and add to the list for current layer
                    action_layer_outputs.append(self.__get_intermediate_layer_action_module_output(
                        propositional_action, action_index, input_tensor, action_module, layer_index))
            concatenated_action_layer_outputs = concatenate(action_layer_outputs, name="action_layer_%d_outputs_concatenation" % layer_index)
            action_layers_outputs.append(concatenated_action_layer_outputs)


            # list of outputs of all proposition modules in layer_index layer
            proposition_layer_outputs = []
            for proposition_index, proposition in enumerate(self.problem_meta.grounded_predicates):
                # extract corresponding proposition input module
                input_tensor = self.__make_proposition_module_input(proposition, action_layers_outputs[layer_index])

                # return Model(inputs=[self.proposition_truth_values, self.proposition_goal_values,
                #         self.action_applicable_values], outputs=input_tensor, name="test")

                # extract corresponding proposition module for layer_index layer
                proposition_module = proposition_layers_modules[layer_index][proposition.predicate]
                # compute output of proposition module with pooling and add to the list for current layer
                proposition_layer_outputs.append(self.__get_proposition_module_output(
                    proposition, proposition_index, input_tensor, proposition_module, layer_index))
            concatenated_proposition_layer_outputs = concatenate(proposition_layer_outputs, name="prop_layer_%d_outputs_concatenation" % layer_index)
            proposition_layers_outputs.append(concatenated_proposition_layer_outputs)

        # last action layer
        outputs = []
        for action_index, propositional_action in enumerate(self.problem_meta.propositional_actions):
            input_tensor = self.__make_action_module_input(propositional_action, proposition_layers_outputs[-1])
            action_module = action_layers_modules[-1][propositional_action.get_underlying_action_name()]
            # compute output of action module and add to the list for current layer
            outputs.append(action_module(input_tensor))

        outputs = concatenate(outputs, name="final_outputs_concatenation")
        policy_output = masked_softmax(outputs, self.action_applicable_values)
        if self.extra_input_size:
            asnet_model = Model(inputs=[self.proposition_truth_values, self.proposition_goal_values,
                self.action_applicable_values, self.additional_input_features], outputs=policy_output,
                name="asnet_keras_model")
        else:
            asnet_model = Model(inputs=[self.proposition_truth_values, self.proposition_goal_values,
                self.action_applicable_values], outputs=policy_output, name="asnet_keras_model")

        return asnet_model


    def build_asnet_keras_model(self,
                                num_layers,
                                hidden_representation_size=16,
                                activation='relu',
                                dropout=0.0,
                                kernel_initializer='glorot_normal',
                                bias_initializer='zeros',
                                extra_input_size=0):
        """
        builds and returns a keras network model for Action Schema Networks

        :param num_layers: number of layers of the ASNet
            (num_layers proposition layers and num_layers + 1 action layers)
        :param hidden_representation_size: hidden representation size used by every
            module (= size of module outputs)
        :param activation: name of activation function to be used in all modules
            of all layers but the last output layer
        :param dropout rate used in every intermediate node
        :param kernel_initializer: initializer to be used for all weight matrices/ kernels
            of all modules
        :param bias_initializer: initializer to be used for all bias vectors
            of all modules
        :param extra_input_size: size of additional input features per action
            This usually involves additional heuristic input features like values
            indicating landmark values for actions (as used in the paper of Toyer
            et al. about ASNets)
        :return: returns the built keras model for an ASNet
        """
        self.num_layers = num_layers
        self.hidden_representation_size = hidden_representation_size
        self.activation = activation
        self.dropout = dropout
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.extra_input_size = extra_input_size

        number_of_propositions = len(self.problem_meta.grounded_predicates)
        number_of_propositional_actions = len(self.problem_meta.propositional_actions)

        # input vector with binary value (0/ 1) for each proposition in the task
        # indicating its truth value
        self.proposition_truth_values = Input(shape=(number_of_propositions,), dtype='float32', name='input_prop_truth_values')

        # input vector with binary value (0/ 1) for each proposition in the task
        # indicating whether it is included in the goal
        self.proposition_goal_values = Input(shape=(number_of_propositions,), dtype='float32', name='input_prop_goal_values')

        # input vector with binary value (0/ 1) for each propositional action in the task
        # indicating whether it is applicable in the current state
        self.action_applicable_values = Input(shape=(number_of_propositional_actions,), dtype='float32', name='input_act_applic_values')

        # additional input features
        if extra_input_size:
            self.additional_input_features = Input(shape=(number_of_propositional_actions * extra_input_size,),
                name='additional_input_features')
        else:
            self.additional_input_features = None

        # action_layers_modules is a list of dicts where the ith dict corresponds
        # to the ith action layer's modules mapping from action schema names to the module.
        # Similarly proposition_layers_modules is a list of dicts where the ith dict corresponds
        # to the ith proposition layer's modules mapping from predicate names to the module
        action_layers_modules, proposition_layers_modules = self.__make_modules()

        asnet_model = self.__make_network(action_layers_modules, proposition_layers_modules)

        return asnet_model