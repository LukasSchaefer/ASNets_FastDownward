from tensorflow.python.keras._impl.keras.models import Model
from tensorflow.python.keras._impl.keras.layers import Input, Dense
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.framework.ops import convert_to_tensor

import numpy as np
import problem_meta


class ASNet_Model_Builder():
    """responsible for building the keras model of Action Schema Networks"""

    def __init__(self,
                 problem_meta):
        """
        :param problem_meta: ASNet specific meta informations about the underlying
            PDDL task
        """
        self.problem_meta = problem_meta


    def __make_action_module(self,
                             action,
                             layer_index,
                             hidden_representation_size,
                             activation,
                             kernel_initializer,
                             bias_initializer,
                             extra_input_size):
        """
        constructs and returns one action module for all propositional actions with
        underlying action schema action in layer layer_index

        :param action: action schema this module is built for
        :param layer_index: index number of the layer this module is built for
        :param hidden_representation_size: hidden representation size used by every
            module (= size of module outputs)
        :param activation: name of activation function to be used in all modules
            of all layers but the last output layer
        :param kernel_initializer: initializer to be used for all weight matrices/ kernels
            of all modules
        :param bias_initializer: initializer to be used for all bias vectors
            of all modules
        :param extra_input_size: size of additional input features per action
            This usually involves additional heuristic input features like values
            indicating landmark values for actions (as used in the paper of Toyer
            et al. about ASNets)
        :return: keras layer representing the module
        """
        num_related_predicates = len(self.problem_meta.action_to_related_pred_names[action])
        if layer_index == 0:
            mod_name = 'first_layer_actmod_' + action.name
            # num_related_predicates truth values and goal values -> 2*
            # + 1 for value indicating if action is applicable
            # + extra_dimension for every module in first layer
            input_shape = (2 * num_related_predicates + 1 + extra_input_size,)
        else:
            mod_name = ('%d_layer_actmod_' % layer_index) + action.name
            # for every related proposition, one hidden representation vector as input
            # of the last layer
            input_shape = (num_related_predicates * hidden_representation_size,)
        return Dense(units=hidden_representation_size,
                     input_shape=input_shape,
                     activation=activation,
                     use_bias=True,
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer,
                     name=mod_name)


    def __make_final_action_layer_module(self,
                                         action,
                                         hidden_representation_size,
                                         kernel_initializer,
                                         bias_initializer):
        """
        constructs and returns one action module for all propositional actions with
        underlying action schema action in the final output layer

        :param action: action schema this module is built for
        :param hidden_representation_size: hidden representation size used by every
            module (= size of module outputs)
        :param kernel_initializer: initializer to be used for all weight matrices/ kernels
            of all modules
        :param bias_initializer: initializer to be used for all bias vectors
            of all modules
        :return: keras layer representing the module of the final output layer
        """
        num_related_predicates = len(problem_meta.action_to_related_pred_names[action])
        mod_name = 'last_layer_actmod_' + action.name
        # TODO How to do the masked softmax?!
        pass


    def __make_predicate_module(self,
                                predicate,
                                layer_index,
                                hidden_representation_size,
                                activation,
                                kernel_initializer,
                                bias_initializer):
        """
        constructs and returns one proposition module for all propositions with
        underlying predicate predicate in layer layer_index

        :param predicate: predicate this module is built for
        :param layer_index: index number of the layer this module is built for
        :param hidden_representation_size: hidden representation size used by every
            module (= size of module outputs)
        :param activation: name of activation function to be used in all modules
            of all layers but the last output layer
        :param kernel_initializer: initializer to be used for all weight matrices/ kernels
            of all modules
        :param bias_initializer: initializer to be used for all bias vectors
            of all modules
        :return: keras layer representing the module
        """
        num_related_action_schemas = len(self.problem_meta.pred_to_related_action_names[predicate])
        mod_name = ('%d_layer_propmod_' % layer_index) + predicate.name
        # one hidden representation sized input vector
        input_shape = (num_related_action_schemas * hidden_representation_size,)
        return Dense(units=hidden_representation_size,
                     input_shape=input_shape,
                     activation=activation,
                     use_bias=True,
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer,
                     name=mod_name)


    def __make_modules(self,
                       num_layers,
                       hidden_representation_size,
                       activation,
                       kernel_initializer,
                       bias_initializer,
                       extra_input_size):
        """
        builds all action and proposition modules based on the ungrounded
        abstract actions and predicates

        :param num_layers: number of layers of the ASNet
            (num_layers proposition layers and num_layers + 1 action layers)
        :param hidden_representation_size: hidden representation size used by every
            module (= size of module outputs)
        :param activation: name of activation function to be used in all modules
            of all layers but the last output layer
        :param kernel_initializer: initializer to be used for all weight matrices/ kernels
            of all modules
        :param bias_initializer: initializer to be used for all bias vectors
            of all modules
        :param extra_input_size: size of additional input features per action
            This usually involves additional heuristic input features like values
            indicating landmark values for actions (as used in the paper of Toyer
            et al. about ASNets)
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
        for layer_index in range(num_layers):
            action_layer_modules = {}
            for action in problem_meta.task.actions:
                # create module for action in layer_index' layer
                action_layer_modules[action.name] = self.__make_action_module(
                    action, layer_index, hidden_representation_size, activation,
                    kernel_initializer, bias_initializer, extra_input_size)
            # complete dicts for each layer are put in action_layers_modules
            action_layers_modules.append(action_layer_modules)

            proposition_layer_modules = {}
            for predicate in problem_meta.task.predicates:
                # create module for predicate in layer_index' layer
                proposition_layer_modules[predicate.name] = self.__make_predicate_module(
                    predicate, layer_index, hidden_representation_size, activation,
                    kernel_initializer, bias_initializer)
            # complete dicts for each layer are put in proposition_layers_modules
            proposition_layers_modules.append(proposition_layer_modules)

        last_action_layer_modules = {}
        # create modules for last action module
        for action in problem_meta.task.actions:
            last_action_layer_modules[action.name] = self.__make_final_action_layer_module(
                action, hidden_representation_size, kernel_initializer, bias_initializer)
        action_layers_modules.append(last_action_layer_modules)

        return action_layers_modules, proposition_layers_modules


    def __get_first_layer_action_module_output(self,
                                               action_index,
                                               related_proposition_ids,
                                               action_module,
                                               proposition_thruth_values,
                                               proposition_goal_values,
                                               action_applicable_values):
        """
        computes output for action module in first layer of concrete propositional action

        :param action_index: index of the propositional action in problem_meta
        :param related_proposition_ids: ids (= indeces in self.problem_meta.grounded_predicates)
            of propositions related to propositional action
        :param action_module: action module of underlying action schema in layer_index layer
        :param proposition_truth_values: keras input vector with binary value (0/ 1) for
            each proposition in the task indicating its truth value
        :param proposition_goal_values: keras input vector with binary value (0/ 1) for
            each proposition in the task indicating whether it is included in the goal
        :param action_applicable_values: keras input vector with binary value (0/ 1) for
            each propositional action in the task indicating whether it is applicable in
            the current state
        :return: output tensor of module
        """
        # build input list
        input_list = []
        # add truth values of related propositions
        input_list.extend([proposition_thruth_values[index] for index in related_proposition_ids])
        # add goal values of related propositions
        input_list.extend([proposition_goal_values[index] for index in related_proposition_ids])
        # add value indicating if propositional action is currently applicable
        input_list.append(action_applicable_values[action_index])
        # convert input list to tensor
        input_tensor = convert_to_tensor(input_list)
        
        return action_module(input_tensor)


    def __get_intermediate_layer_action_module_output(self,
                                                      propositional_action,
                                                      action_module,
                                                      last_layer_proposition_module_outputs):
        """
        computes output for action module in first layer

        :param propositional_action: propositional action the module corresponds to
        :param action_module: action module of underlying action schema in layer_index layer
        :param last_layer_proposition_module_outputs: list with outputs of propositional modules
            of the last layer (in the order of ids)
        :return: output tensor of module
        """
        # list of ids (= indeces in self.problem_meta.grounded_predicates) of related propositions
        related_proposition_ids = self.problem_meta.prop_action_to_related_gr_pred_ids[
                    propositional_action]
        # collect outputs of related proposition modules in last layer
        related_outputs = []
        for index in related_proposition_ids:
            related_outputs.append(last_layer_proposition_module_outputs[index])
        # concatenate related output tensors to new input tensor
        input_tensor = K.concatenate(related_outputs)
        
        return action_module(input_tensor)



    def __make_network(self,
                       num_layers,
                       proposition_thruth_values,
                       proposition_goal_values,
                       action_applicable_values,
                       action_layers_modules,
                       proposition_layers_modules):
        """
        build concrete ASNet with all connections with modules

        :param num_layers: number of layers of the ASNet
        (num_layers proposition layers and num_layers + 1 action layers)
        :param proposition_truth_values: keras input vector with binary value (0/ 1) for
            each proposition in the task indicating its truth value
        :param proposition_goal_values: keras input vector with binary value (0/ 1) for
            each proposition in the task indicating whether it is included in the goal
        :param action_applicable_values: keras input vector with binary value (0/ 1) for
            each propositional action in the task indicating whether it is applicable in
            the current state
        :param action_layers_modules: list of dicts where the ith dict corresponds to the
            ith action layer's modules mapping from action schema names to the module
        :param proposition_layers_modules: list of dicts where the ith dict corresponds to
            the ith proposition layer's modules mapping from predicate names to the module
        :return: action_layers_outputs, proposition_layers_outputs
            with action_layers_outputs being a list of lists where the jth entry in the ith
            list corresponds to the output of the jth action module in the ith layer
            (jth action module = module of jth propositional action in
            self.problem_meta.propositional_actions)
            Similarly proposition_layers_outputs is a list of lists where the jth entry in
            the ith list corresponds to the output of the jth proposition module in the ith
            layer (jth proposition module = module of jth grounded predicate in
            self.problem_meta.grounded_predicates)
        """
        action_layers_outputs = []
        proposition_layers_outputs = []

        # create concrete layers and put them in corresponding lists for all but last action
        # layer
        for layer_index in range(num_layers):
            # list of outputs of all action modules in layer_index layer
            action_layer_outputs = []
            for action_index, propositional_action in enumerate(self.problem_meta.propositional_actions):
                if layer_index == 0:
                    # list of ids = indeces of related propositions
                    related_proposition_ids = self.problem_meta.prop_action_to_related_gr_pred_ids[
                        propositional_action]

                    # extract corresponding action module in first layer
                    action_module = action_layers_modules[0][propositional_action.get_underlying_action_name()]

                    action_layer_outputs.append(self.__get_first_layer_action_module_output(
                        action_index, related_proposition_ids, action_module, proposition_thruth_values,
                        proposition_goal_values, action_applicable_values))
                else:
                    # extract corresponding action module in layer_index layer
                    action_module = action_layers_modules[layer_index][
                        propositional_action.get_underlying_action_name()]
                    action_layer_outputs.append(self.__get_intermediate_layer_action_module_output(
                        propositional_action, action_module, proposition_layers_outputs[layer_index - 1]))
            action_layers_outputs.append(action_layer_outputs)

            # list of outputs of all proposition modules in layer_index layer
            proposition_layer_outputs = []
            for proposition_index, proposition in enumerate(self.problem_meta.grounded_predicates):
                # TODO Implement __get_propositional_module_output with Pooling
            proposition_layers_outputs.append(proposition_layer_outputs)



        return action_layers_outputs, proposition_layers_outputs


    def build_asnet_keras_model(self,
                                num_layers=2,
                                hidden_representation_size=16,
                                activation='relu',
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
        number_of_propositions = len(self.problem_meta.grounded_predicates)
        number_of_propositional_actions = len(self.problem_meta.propositional_actions)

        # input vector with binary value (0/ 1) for each proposition in the task
        # indicating its truth value
        proposition_thruth_values = Input(shape=(number_of_propositions,), name='input_prop_truth_values')

        # input vector with binary value (0/ 1) for each proposition in the task
        # indicating whether it is included in the goal
        proposition_goal_values = Input(shape=(number_of_propositions,), name='input_prop_goal_values')

        # input vector with binary value (0/ 1) for each propositional action in the task
        # indicating whether it is applicable in the current state
        action_applicable_values = Input(shape=(number_of_propositional_actions,), name='input_act_applic_values')

        # additional input features
        if extra_input_size:
            additional_input_features = Input(shape=(extra_input_size,), name='additional_input_features')

        # action_layers_modules is a list of dicts where the ith dict corresponds
        # to the ith action layer's modules mapping from action schema names to the module.
        # Similarly proposition_layers_modules is a list of dicts where the ith dict corresponds
        # to the ith proposition layer's modules mapping from predicate names to the module
        action_layers_modules, proposition_layers_modules = self.__make_modules(
            num_layers, hidden_representation_size, activation, kernel_initializer,
            bias_initializer, extra_input_size)

        # action_layers_outputs is a list of lists where the jth entry in the ith
        # list corresponds to the output of the jth action module in the ith layer
        # (jth action module = module of jth propositional action in
        # self.problem_meta.propositional_actions)
        # Similarly proposition_layers_outputs is a list of lists where the jth entry in
        # the ith list corresponds to the output of the jth proposition module in the ith
        # layer (jth proposition module = module of jth grounded predicate in
        # self.problem_meta.grounded_predicates)
        action_layers_outputs, proposition_layers_outputs = self.__make_network(
            action_layers_modules, proposition_layers_modules)
