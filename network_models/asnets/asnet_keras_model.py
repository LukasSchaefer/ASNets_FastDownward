import os
import sys
import re
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda, concatenate, GlobalMaxPooling1D
from keras import regularizers
from keras import backend as K

import problem_meta
from custom_keras_layers.softmax_layer import SoftmaxOutputLayer
from custom_keras_layers.action_module_layer import FirstActionInputLayer, IntermediateActionInputLayer, ActionModuleLayer
from custom_keras_layers.proposition_module_layer import PropositionInputLayer, PropositionModuleLayer


class ASNet_Model_Builder():
    """responsible for building the keras model of Action Schema Networks"""

    def __init__(self, problem_meta, print_all):
        """
        :param problem_meta: ASNet specific meta informations about the underlying
            PDDL task
        :param print_all: bool value indicating whether all steps should be printed
        """
        self.problem_meta = problem_meta
        self.print_all = print_all

        # initialize all layer counters for unique naming
        # dicts for unique counter per predicate name
        self.pred_counter = {}
        self.pred_act_schema_counter = {}


    def __make_action_module(self, action, layer_index):
        """
        constructs and returns one action module for all propositional actions with
        underlying action schema action in layer layer_index

        :param action: action schema this module is built for
        :param layer_index: index number of the layer this module is built for
        :return: keras layer representing the module
        """
        if layer_index == 0:
            mod_name = '1st_layer_actmod_' + re.sub(r"\W+", "", action.name)
        else:
            mod_name = ('%d_layer_actmod_' % layer_index) + re.sub(r"\W+", "", action.name)

        return ActionModuleLayer(hidden_representation_size=self.hidden_representation_size,
                                 activation=self.activation,
                                 dropout=self.dropout,
                                 kernel_initializer=self.kernel_initializer,
                                 bias_initializer=self.bias_initializer,
                                 regularizer_value=self.regularizer_value,
                                 name=mod_name)
        

    def __make_final_action_layer_module(self, action):
                                         
        """
        constructs and returns one action module for all propositional actions with
        underlying action schema action in the final output layer

        :param action: action schema this module is built for
        :return: keras layer representing the module of the final output layer
        """
        mod_name = 'last_layer_actmod_' + re.sub(r"\W+", "", action.name)
        # scalar output with identity activation function
        return ActionModuleLayer(hidden_representation_size=1,
                                 activation='linear',
                                 dropout=0,
                                 kernel_initializer=self.kernel_initializer,
                                 bias_initializer=self.bias_initializer,
                                 regularizer_value=self.regularizer_value,
                                 name=mod_name)


    def __make_predicate_module(self, predicate, layer_index):
        """
        constructs and returns one proposition module for all propositions with
        underlying predicate predicate in layer layer_index

        :param predicate: predicate this module is built for
        :param layer_index: index number of the layer this module is built for
        :return: keras layer representing the module
        """
        mod_name = ('%d_layer_propmod_' % layer_index) + re.sub(r"\W+", "", predicate.name)
        return PropositionModuleLayer(hidden_representation_size=self.hidden_representation_size,
                                      activation=self.activation,
                                      dropout=self.dropout,
                                      kernel_initializer=self.kernel_initializer,
                                      bias_initializer=self.bias_initializer,
                                      regularizer_value=self.regularizer_value,
                                      name=mod_name)


    def __make_modules(self):
        """
        builds all action and proposition modules based on the ungrounded
        abstract actions and predicates

        :returns: action_layers_modules, proposition_layers_modules,
                  action_input_modules, proposition_input_modules
            with action_layers_modules being a list of dicts where the ith dict corresponds
            to the ith action layer's modules mapping from action schema names to the module.
            Similarly proposition_layers_modules is a list of dicts where the ith dict corresponds
            to the ith proposition layer's modules mapping from predicate names to the module

            action_input_modules is a dict mapping from propositional action name to input module
            (for all layers!).
            Similarly proposition_input_modules is a dict mapping from proposition name to input
            module (for all layers!)
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
            for action in self.problem_meta.pddl_task.actions:
                # create module for action in layer_index' layer
                action_layer_modules[action.name] = self.__make_action_module(action, layer_index)
            # complete dicts for each layer are put in action_layers_modules
            action_layers_modules.append(action_layer_modules)

            proposition_layer_modules = {}
            for predicate in self.problem_meta.pddl_task.predicates:
                # create module for predicate in layer_index' layer
                proposition_layer_modules[predicate.name] = self.__make_predicate_module(
                    predicate, layer_index)
            # complete dicts for each layer are put in proposition_layers_modules
            proposition_layers_modules.append(proposition_layer_modules)

        last_action_layer_modules = {}
        # create modules for last action layer
        for action in self.problem_meta.pddl_task.actions:
            last_action_layer_modules[action.name] = self.__make_final_action_layer_module(action)
        action_layers_modules.append(last_action_layer_modules)

        # dict mapping from propositional action name to input module (for all layers!)
        action_input_modules = {}
        for propositional_action in self.problem_meta.propositional_actions:
            propositional_action_id = self.problem_meta.prop_action_name_to_id[propositional_action.name]
            related_proposition_ids = self.problem_meta.prop_action_to_related_gr_pred_ids[propositional_action]
            mod_name = 'action_inputmod_' + re.sub(r"\W+", "", propositional_action.name)
            input_module = IntermediateActionInputLayer(hidden_representation_size=self.hidden_representation_size,
                                                        action_index=propositional_action_id,
                                                        related_proposition_ids=related_proposition_ids,
                                                        name=mod_name)
            action_input_modules[propositional_action.name] = input_module
        
        # dict mapping from proposition name to input module (for all layers!)
        proposition_input_modules = {}
        for proposition in self.problem_meta.grounded_predicates:
            related_propositional_action_ids = self.problem_meta.gr_pred_to_related_prop_action_ids[proposition]
            mod_name = 'prop_inputmod_' + re.sub(r"\W+", "", proposition.__str__())
            input_module = PropositionInputLayer(hidden_representation_size=self.hidden_representation_size,
                                                 related_propositional_action_ids=related_propositional_action_ids,
                                                 name=mod_name)
            proposition_input_modules[proposition.__str__()] = input_module


        return action_layers_modules, proposition_layers_modules,\
               action_input_modules, proposition_input_modules


    def __make_network(self,
                       action_layers_modules,
                       proposition_layers_modules,
                       action_input_modules,
                       proposition_input_modules):
        """
        build concrete ASNet with all connections and modules

        :param action_layers_modules: list of dicts where the ith dict corresponds to the
            ith action layer's modules mapping from action schema names to the module
        :param proposition_layers_modules: list of dicts where the ith dict corresponds to
            the ith proposition layer's modules mapping from predicate names to the module
        :param action_input_modules: dict mapping from propositional action name to input module
            for intermediate ation modules
        :param proposition_input_modules: dict mapping from proposition name to input module

        :return: action_layers_outputs, proposition_layers_outputs
            with action_layers_outputs being a list of tensors where the ith tensor
            is a concatenation of output tensors of all action modules in the ith layer
            (in the order of self.problem_meta.propositional_actions)
            Similarly proposition_layers_outputs is a list of tensors where the ith tensor
            is a concatenation of outputs tensors of all proposition modules in the ith layer
            (in the order of self.problem_meta.grounded_predicates)
        """
        last_action_layer_output = None
        last_proposition_layer_output = None

        # create concrete layers
        for layer_index in range(self.num_layers):
            # list of outputs of all action modules in layer_index layer
            action_layer_outputs = []
            if self.print_all:
                print("Building act layer %d" % layer_index)
            for action_index, propositional_action in enumerate(self.problem_meta.propositional_actions):
                if self.print_all:
                    print("Computing output of %s" % propositional_action.name)
                if layer_index == 0:
                    # list of ids (= indeces) of related propositions
                    related_proposition_ids = self.problem_meta.prop_action_to_related_gr_pred_ids[propositional_action]
                    related_proposition_names = self.problem_meta.prop_action_to_related_gr_pred_names[propositional_action]

                    first_action_layer_input = FirstActionInputLayer(sas_task=self.problem_meta.sas_task,
                                                                     action_index=action_index,
                                                                     related_proposition_ids=related_proposition_ids,
                                                                     related_proposition_names=related_proposition_names,
                                                                     extra_input_size=self.extra_input_size)
                    # extract corresponding action module for first layer
                    action_module = action_layers_modules[0][propositional_action.get_underlying_action_name()]
                    # compute output of action module and add to the list for first layer
                    if self.extra_input_size:
                        input_tensor = first_action_layer_input([self.proposition_truth_values,
                                                                 self.proposition_goal_values,
                                                                 self.action_applicable_values,
                                                                 self.additional_input_features])
                    else:
                        input_tensor = first_action_layer_input([self.proposition_truth_values,
                                                                 self.proposition_goal_values,
                                                                 self.action_applicable_values])
                    action_module_output = action_module(input_tensor)
                    action_layer_outputs.append(action_module_output)
                else:
                    # compute corresponding action input tensor
                    input_module = action_input_modules[propositional_action.name]
                    if last_proposition_layer_output is None:
                        raise ValueError("In action layer %d was no last proposition layer output available!" % layer_index)
                    input_tensor = input_module(last_proposition_layer_output)

                    # extract corresponding action module for layer_index layer
                    action_module = action_layers_modules[layer_index][propositional_action.get_underlying_action_name()]
                    # compute output of action module and add to the list for current layer
                    action_module_output = action_module(input_tensor)
                    action_layer_outputs.append(action_module_output)
            concatenated_action_layer_outputs = concatenate(action_layer_outputs, name="action_layer_%d_outputs_concatenation" % layer_index)
            last_action_layer_output = concatenated_action_layer_outputs


            # list of outputs of all proposition modules in layer_index layer
            proposition_layer_outputs = []
            if self.print_all:
                print("Build Prop layer %d" % layer_index)
            for proposition in self.problem_meta.grounded_predicates:
                if self.print_all:
                    print("Computing output of %s" % proposition.__str__())
                # compute corresponding proposition input tensor
                input_module = proposition_input_modules[proposition.__str__()]
                if last_action_layer_output is None:
                    raise ValueError("In action layer %d was no last proposition layer output available!" % layer_index)
                input_tensor = input_module(last_action_layer_output)

                # extract corresponding proposition module for layer_index layer
                proposition_module = proposition_layers_modules[layer_index][proposition.predicate]
                # compute output of proposition module with pooling and add to the list for current layer
                proposition_module_output = proposition_module(input_tensor)
                proposition_layer_outputs.append(proposition_module_output)
            concatenated_proposition_layer_outputs = concatenate(proposition_layer_outputs, name="prop_layer_%d_outputs_concatenation" % layer_index)
            last_proposition_layer_output = concatenated_proposition_layer_outputs

        # last action layer
        outputs = []
        if self.print_all:
            print("Build last layer")
        for action_index, propositional_action in enumerate(self.problem_meta.propositional_actions):
            # compute corresponding action input tensor
            input_module = action_input_modules[propositional_action.name]
            if last_proposition_layer_output is None:
                raise ValueError("In last action was no last proposition layer output available!")
            input_tensor = input_module(last_proposition_layer_output)

            action_module = action_layers_modules[-1][propositional_action.get_underlying_action_name()]
            # compute output of action module and add to the list for output tensors
            outputs.append(action_module(input_tensor))

        if self.print_all:
            print("Computing final output")
        outputs = concatenate(outputs, name="final_outputs_concatenation")
        policy_output = SoftmaxOutputLayer()([outputs, self.action_applicable_values])
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
                                dropout=0.25,
                                kernel_initializer='glorot_normal',
                                bias_initializer='zeros',
                                regularizer_value=0.001,
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
        :param regularizer_value: value used for all L2 regularizations applied
            to all weights (-matrices and bias vectors!)
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
        self.regularizer_value = regularizer_value
        self.extra_input_size = extra_input_size

        assert num_layers >= 1, "There has to be at least 1 layer!"
        assert hidden_representation_size > 0, "The hidden representation size has to be at least 1!"
        assert 0.0 <= dropout and dropout <= 1, "Dropout value has to be between 0 and 1!"
        assert extra_input_size >= 0, "The extra input size has to be positive!"

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
        action_layers_modules, proposition_layers_modules, action_input_modules, proposition_input_modules =\
             self.__make_modules()
        if self.print_all:
            print("Modules built")

        asnet_model = self.__make_network(action_layers_modules,
                                          proposition_layers_modules,
                                          action_input_modules,
                                          proposition_input_modules)

        return asnet_model
