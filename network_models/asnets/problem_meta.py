class ProblemMeta:
    """Contains additional information about relations of actions
    and propositions in a pddl-task used for ASNets"""

    def __init__(self,
                 task,
                 propositional_actions,
                 grounded_predicates):
        self.task = task
        self.propositional_actions = propositional_actions
        self.grounded_predicates = grounded_predicates

        # setup dicts to access propositional_actions and grounded predicates by name
        # {prop_act.name: prop_act} and {gr_pred.__str__(): gr_pred}
        self.propositional_actions_by_name = \
                self.__compute_propositional_actions_by_name_dict(propositional_actions)
        self.grounded_predicates_by_name = \
                self.__compute_grounded_predicates_by_name_dict(grounded_predicates)

        # set dicts mapping from abstract actions/ predicates to grounding names
        self.action_name_to_prop_actions_names = \
                self.__compute_act_to_prop_act_names_dict(task.actions, propositional_actions)
        self.predicate_name_to_propositions_names = \
                self.__compute_pred_to_grounding_names_dict(task.predicates, grounded_predicates) 

        self.__compute_and_set_relations_between_groundings(propositional_actions, grounded_predicates)
        self.__compute_and_set_relations_between_abstracts(task.actions, task.predicates)
        

    def __compute_propositional_actions_by_name_dict(self, propositional_actions):
        """
        Computes dict {prop_act.name: prop_act}
        :param propositional_actions: all propositional actions (to include)
        :return: dict {prop_act.name: prop_act}
        """
        propositional_actions_by_name = {}
        for act in propositional_actions:
            propositional_actions_by_name[act.name] = act
        return propositional_actions_by_name


    def __compute_grounded_predicates_by_name_dict(self, grounded_predicates):
        """
        Computes dict {gr_pred.__str__(): gr_pred}
        :param grounded_predicates: all grounded predicates (to include)
        :return: dict {gr_pred.__str__(): gr_pred}
        """
        grounded_predicates_by_name = {}
        for gr_pred in grounded_predicates:
            grounded_predicates_by_name[gr_pred.__str__()] = gr_pred
        return grounded_predicates_by_name


    def __compute_act_to_prop_act_names_dict(self, actions, propositional_actions):
        """
        Computes dict of form {action_name: [corresponding propositional_actions.name]
        :param actions: abstract actions
        :param propositional_actions: all instantiated propositional actions
        :return: dict mapping from action names to list of corresponding propositional
        action names
        """
        action_name_to_prop_actions = {}
        # init all act-entries with empty list
        for act in actions:
            action_name_to_prop_actions[act.name] = []

        # fill all lists with corresponding propositional actions
        for prop_action in propositional_actions:
            action_name_to_prop_actions[prop_action.get_underlying_action_name()] \
                    .append(prop_action.name)
        return action_name_to_prop_actions


    def __compute_pred_to_grounding_names_dict(self, predicates, grounded_predicates):
        """
        Computes dict of form {predicate_name: [corresponding grounded_predicates.__str__()]
        :param predicates: abstract predicates
        :param grounded_predicates: all instantiated grounded predicates
        :return: dict mapping from predicate names to list of corresponding grounded
        predicate names
        """
        predicate_name_to_propositions = {}
        # init all pred-entries with empty list
        for pred in predicates:
            predicate_name_to_propositions[pred.name] = []

        # fill all lists with corresponding propositions
        for prop in grounded_predicates:
            predicate_name_to_propositions[prop.predicate].append(prop.__str__())
        return predicate_name_to_propositions


    def __compute_and_set_relations_between_groundings(self, propositional_actions, grounded_predicates):
        """
        Computes relations among grounded propositional actions and predicates
        (action and prop are related iff prop appears in pre, add or delete list
        of action) and sets these in corresponding dicts in both directions
        dicts:
            - prop_to_related_prop_action_names:
                {gr_pred: related prop_actions.name}
            - prop_action_to_related_prop_names:
                {prop_action: related gr_preds.__str__()}
        :param propositional_actions: propositional actions to use
        :param grounded_predicates: set of predicates to use
        :return: None
        """
        # Dict of type propositional_action -> list(grounded_predicate)
        self.prop_action_to_related_prop_names = {}
        # Dict of type grounded_predicate -> list(propositional_actions)
        self.prop_to_related_prop_action_names = {}

        # initialize dict entries for grounded predicates all with empty list
        # -> if no related actions than value is already correct and no in dict.keys()
        # check necessary in self.__compute_and_set... function below
        for gr_pred in grounded_predicates:
            self.prop_to_related_prop_action_names[gr_pred] = []

        # IMPORTANT: ordered data structure (like list) so that all related-lists
        # of grounded actions with same underlying action schema have corresponding
        # propositions at the same index (necessary for weight sharing in ASNets)
        for propositional_action in propositional_actions:
            self.__compute_and_set_relations_involving_propositional_action(propositional_action)

        # sort all action names in lists alphanumerically
        for gr_pred in grounded_predicates:
            self.prop_to_related_prop_action_names[gr_pred] = sorted(self.prop_to_related_prop_action_names[gr_pred])


    def __compute_and_set_relations_involving_propositional_action(self, propositional_action):
        """
        Computes related grounded predicates (propositions) for grounded
        propositional action (= propositions that either appear in action's
        preconditions, add- or delete-list) and sets these in corresponding
        dicts in both directions
        :param propositional_action: propositional action to compute related
            propositions of
        :return: None
        """
        related_propositions = []
        for proposition in propositional_action.precondition:
                related_propositions.append(proposition.__str__())
                self.prop_to_related_prop_action_names[proposition].append(propositional_action.name)
        for _, proposition in propositional_action.add_effects:
            if proposition.__str__() not in related_propositions:
                related_propositions.append(proposition.__str__())
                self.prop_to_related_prop_action_names[proposition].append(propositional_action.name)
        for _, proposition in propositional_action.del_effects:
            if proposition.__str__() not in related_propositions:
                related_propositions.append(proposition.__str__())
                self.prop_to_related_prop_action_names[proposition].append(propositional_action.name)

        self.prop_action_to_related_prop_names[propositional_action] = sorted(related_propositions)


    def __compute_and_set_relations_between_abstracts(self, actions, predicates):
        """
        Computes relations among abstract actions and predicates
        (action and pred are related iff pred appears in pre, add or delete list
        of action) and sets these in corresponding dicts in both directions
        dicts:
            - pred_to_related_action_names:
                {pred: related actions.name}
            - action_to_related_pred_names:
                {action: related preds.name with pars}
        :param actions: actions to use
        :param predicates: predicates to use
        """
        # Dict of type action -> list(predicate)
        self.action_to_related_pred_names = {}
        # Dict of type predicate -> list(action)
        self.pred_to_related_action_names = {}

        # initialize dict entries for predicates all with empty list
        # -> if no related actions than value is already correct and no in dict.keys()
        # check necessary in self.__compute_and_set... function below
        for pred in predicates:
            self.pred_to_related_action_names[pred] = []

        for action in actions:
            self.__compute_and_set_relations_involving_action(action)

        for pred in predicates:
            self.pred_to_related_action_names[pred] = sorted(self.pred_to_related_action_names[pred])


    def __compute_and_set_relations_involving_action(self, action):
        """
        Computes related predicates for action (= predicates that either appear
        in action's preconditions, add- or delete-list) and sets these in
        corresponding dicts in both directions
        :param action: action to compute related predicates of
        """
        related_predicates = []
        for part in action.precondition.parts:
            related_predicates.append(part)
            predicate = self.task._predicate_dict()[part.predicate]
            if action.name not in self.pred_to_related_action_names[predicate]:
                self.pred_to_related_action_names[predicate].append(action.name)
        for eff in action.effects:
            pred = eff.literal
            if pred.negated:
                pred = pred.negate()
            if pred not in related_predicates:
                related_predicates.append(pred)
                predicate = self.task._predicate_dict()[pred.predicate]
                if action.name not in self.pred_to_related_action_names[predicate]:
                    self.pred_to_related_action_names[predicate].append(action.name)

        self.action_to_related_pred_names[action] = sorted(related_predicates)
