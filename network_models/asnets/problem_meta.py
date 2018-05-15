class ProblemMeta:
    """Contains additional information about relations of actions
    and propositions in a pddl-task used for ASNets"""

    def __init__(self,
                 task,
                 propositional_actions,
                 grounded_predicates):
        self.task = task
        self.propositional_actions = sorted(propositional_actions, key=lambda prop_action: prop_action.name)
        self.grounded_predicates = sorted(grounded_predicates)

        # setup dicts to access propositional_actions and grounded predicates by name
        # {prop_act.name: prop_act} and {gr_pred.__str__(): gr_pred}
        self.propositional_actions_by_name = \
                self.__compute_propositional_actions_by_name_dict()
        self.grounded_predicates_by_name = \
                self.__compute_grounded_predicates_by_name_dict()

        # setup dict from gr_pred.__str__() and prop_act.name to int ids
        # (simply index in self.grounded_predicates and self.propositional_actions)
        self.prop_action_name_to_id = \
                self.__compute_propositional_action_name_to_id_dict()
        self.grounded_predicate_name_to_id = \
                self.__compute_grounded_predicate_name_to_id_dict()

        # setup dicts to access propositional_actions and grounded predicates by id
        # {prop_act_id: prop_act} and {gr_pred_id: gr_pred}
        self.propositional_actions_by_id = {id: self.propositional_actions_by_name[prop_act_name] \
                for prop_act_name, id in self.prop_action_name_to_id.items()}
        self.grounded_predicates_by_id = {id: self.grounded_predicates_by_name[gr_pred_name] \
                for gr_pred_name, id in self.grounded_predicate_name_to_id.items()}

        # set dicts mapping from abstract actions/ predicates to grounding names
        self.action_name_to_prop_actions = \
                self.__compute_act_to_prop_acts_dict()
        self.predicate_name_to_grounded_predicates = \
                self.__compute_pred_to_groundings_dict() 

        # computes relations between propositional actions and grounded predicates and sets
        # gr_pred_to_related_prop_action_names {gr_pred: [prop_action.name, ...]} and
        # prop_action_to_related_gr_pred_names {prop_action: [gr_pred.name, ...]} accordingly
        self.__compute_and_set_relations_between_groundings()

        # computes relations between abstract actions and predicates and sets
        # pred_to_related_action_names {pred: [action.name, ...]} and
        # action_to_related_pred_names {action: [pred.name, ...]} accordingly
        self.__compute_and_set_relations_between_abstracts()
        

    def __compute_propositional_action_name_to_id_dict(self):
        """
        Computes dict {prop_act.name: id} where id corresponds to index in
        propositional_actions list (should be sorted)
        :return: dict {prop_act: id}
        """
        prop_action_to_id_dict = {}
        for index, prop_act in enumerate(self.propositional_actions):
            prop_action_to_id_dict[prop_act.name] = index
        return prop_action_to_id_dict


    def __compute_grounded_predicate_name_to_id_dict(self):
        """
        Computes dict {gr_pred.__str__(): id} where id corresponds to index in
        grounded_predicates list (should be sorted)
        :return: dict {gr_pred: id}
        """
        grounded_predicate_to_id_dict = {}
        for index, gr_pred in enumerate(self.grounded_predicates):
            grounded_predicate_to_id_dict[gr_pred.__str__()] = index
        return grounded_predicate_to_id_dict


    def __compute_propositional_actions_by_name_dict(self):
        """
        Computes dict {prop_act.name: prop_act}
        :return: dict {prop_act.name: prop_act}
        """
        propositional_actions_by_name = {}
        for act in self.propositional_actions:
            propositional_actions_by_name[act.name] = act
        return propositional_actions_by_name


    def __compute_grounded_predicates_by_name_dict(self):
        """
        Computes dict {gr_pred.__str__(): gr_pred}
        :return: dict {gr_pred.__str__(): gr_pred}
        """
        grounded_predicates_by_name = {}
        for gr_pred in self.grounded_predicates:
            grounded_predicates_by_name[gr_pred.__str__()] = gr_pred
        return grounded_predicates_by_name


    def __compute_act_to_prop_acts_dict(self):
        """
        Computes dict of form {action_name: [corresponding propositional_actions]
        :return: dict mapping from action names to list of corresponding propositional
        actions
        """
        action_name_to_prop_actions = {}
        # init all act-entries with empty list
        for act in self.task.actions:
            action_name_to_prop_actions[act.name] = []

        # fill all lists with corresponding propositional actions
        for prop_action in self.propositional_actions:
            action_name_to_prop_actions[prop_action.get_underlying_action_name()] \
                    .append(prop_action)
        return action_name_to_prop_actions


    def __compute_pred_to_groundings_dict(self):
        """
        Computes dict of form {predicate_name: [corresponding grounded_predicates]
        :return: dict mapping from predicate names to list of corresponding grounded
        predicates
        """
        predicate_name_to_propositions = {}
        # init all pred-entries with empty list
        for pred in self.task.predicates:
            predicate_name_to_propositions[pred.name] = []

        # fill all lists with corresponding propositions
        for prop in self.grounded_predicates:
            predicate_name_to_propositions[prop.predicate].append(prop)
        return predicate_name_to_propositions


    def __compute_and_set_relations_between_groundings(self):
        """
        Computes relations among grounded propositional actions and predicates
        (action and prop are related iff prop appears in pre, add or delete list
        of action) and sets these in corresponding dicts in both directions
        dicts:
            - gr_pred_to_related_prop_action_names:
                {gr_pred: related prop_actions.name}
            - prop_action_to_related_gr_pred_names:
                {prop_action: related gr_preds.__str__()}
        :return: None
        """
        # Dict of type propositional_action -> list(grounded_predicate)
        self.prop_action_to_related_gr_pred_names = {}
        # Dict of type grounded_predicate -> list(propositional_actions)
        self.gr_pred_to_related_prop_action_names = {}

        # initialize dict entries for grounded predicates all with empty list
        # -> if no related actions than value is already correct and no in dict.keys()
        # check necessary in self.__compute_and_set... function below
        for gr_pred in self.grounded_predicates:
            self.gr_pred_to_related_prop_action_names[gr_pred] = []

        # IMPORTANT: ordered data structure (like list) so that all related-lists
        # of grounded actions with same underlying action schema have corresponding
        # propositions at the same index (necessary for weight sharing in ASNets)
        for propositional_action in self.propositional_actions:
            self.__compute_and_set_relations_involving_propositional_action(propositional_action)

        # sort all action names in lists alphanumerically
        for gr_pred in self.grounded_predicates:
            self.gr_pred_to_related_prop_action_names[gr_pred] = self.gr_pred_to_related_prop_action_names[gr_pred]


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
                self.gr_pred_to_related_prop_action_names[proposition].append(propositional_action.name)
        for _, proposition in propositional_action.add_effects:
            if proposition.__str__() not in related_propositions:
                related_propositions.append(proposition.__str__())
                self.gr_pred_to_related_prop_action_names[proposition].append(propositional_action.name)
        for _, proposition in propositional_action.del_effects:
            if proposition.__str__() not in related_propositions:
                related_propositions.append(proposition.__str__())
                self.gr_pred_to_related_prop_action_names[proposition].append(propositional_action.name)

        self.prop_action_to_related_gr_pred_names[propositional_action] = related_propositions


    def get_related_grounded_predicate_ids(self, propositional_action):
        """
        Receive list of related grounded predicate ids for propositonal action
        :param propositional_action: propositional_action to compute relations of
        :return: list of related grounded predicate ids
        """
        related_proposition_names = self.prop_action_to_related_gr_pred_names[propositional_action]
        related_proposition_ids = []
        for prop_name in related_proposition_names:
            related_proposition_ids.append(self.grounded_predicate_name_to_id[prop_name])
        return related_proposition_ids


    def get_related_propositional_action_ids(self, grounded_predicate):
        """
        Receive related propositional actions to grounded predicate and returns sorted
        list of ids of these related propositional actions in sub-lists of format so that
        propositional actions instantiated from the same action schema are in the same sub-list.
        E.g. [[act1.id, act2.id], [act3.id, ...], ...] if act1 and act2 are groundings
        of the same action schemas
        :param grounded_predicate: grounded predicate to compute related propositional
            action ids of
        :return: list of sorted lists of related propositional action ids as described
            above
        """
        related_propositional_action_names = self.gr_pred_to_related_prop_action_names[grounded_predicate]
        # dict from action_schema_name to list of propositional action ids
        related_prop_action_id_dict = {}

        # instantiate according to relations of underlying predicate
        underlying_predicate = self.task._predicate_dict()[grounded_predicate.predicate]
        related_action_schema_names = self.pred_to_related_action_names[underlying_predicate]
        for action_schema_name in related_action_schema_names:
            related_prop_action_id_dict[action_schema_name] = []

        for prop_act_name in related_propositional_action_names:
            prop_act_id = self.prop_action_name_to_id[prop_act_name]
            # extract action schema name out of propositional action name
            action_schema_name = prop_act_name.strip('(').split()[0]
            related_prop_action_id_dict[action_schema_name].append(prop_act_id)

        # build related_prop_action_id_list
        related_prop_action_ids = []
        for action_schema_name in related_action_schema_names:
            related_action_ids = related_prop_action_id_dict[action_schema_name]
            related_prop_action_ids.append(related_action_ids)
   
        return related_prop_action_ids


    def __compute_and_set_relations_between_abstracts(self):
        """
        Computes relations among abstract actions and predicates
        (action and pred are related iff pred appears in pre, add or delete list
        of action) and sets these in corresponding dicts in both directions
        dicts:
            - pred_to_related_action_names:
                {pred: related actions.name}
            - action_to_related_pred_names:
                {action: related preds.name with pars}
        """
        # Dict of type action -> list(predicate)
        self.action_to_related_pred_names = {}
        # Dict of type predicate -> list(action)
        self.pred_to_related_action_names = {}

        # initialize dict entries for predicates all with empty list
        # -> if no related actions than value is already correct and no in dict.keys()
        # check necessary in self.__compute_and_set... function below
        for pred in self.task.predicates:
            self.pred_to_related_action_names[pred] = []

        for action in self.task.actions:
            self.__compute_and_set_relations_involving_action(action)

        for pred in self.task.predicates:
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
