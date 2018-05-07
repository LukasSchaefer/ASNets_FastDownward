class ProblemRelationsMeta:
    """Contains additional information about relations of actions
    and propositions in a pddl-task used for ASNets"""

    def __init__(self,
                 name,
                 propositional_actions,
                 grounded_predicates):
        self.name = name

        # Dict of type propositional_action.name -> list(grounded_predicate)
        self.prop_action_name_to_related_props_name = {}
        # Dict of type grounded_predicate.name -> list(propositional_actions)
        self.prop_name_to_related_prop_action_name = {}
        # initialize dict entries for grounded predicates all with empty list
        # -> if no related actions than value is already correct and no in dict.keys()
        # check necessary in self.__compute_and_set... function below
        for proposition in grounded_predicates:
            self.prop_name_to_related_prop_action_name[proposition.__str__()] = []

        # IMPORTANT: ordered data structure (like list) so that all related-lists
        # of grounded actions with same underlying action schema have corresponding
        # propositions at the same index (necessary for weight sharing in ASNets)

        for propositional_action in propositional_actions:
            self.__compute_and_set_relations_involving_action(propositional_action)


    def __compute_and_set_relations_involving_action(self, action):
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
        for proposition in action.precondition:
                related_propositions.append(proposition)
                self.prop_name_to_related_prop_action_name[proposition.__str__()].append(action)
        for _, proposition in action.add_effects:
            if not proposition in related_propositions:
                related_propositions.append(proposition)
                self.prop_name_to_related_prop_action_name[proposition.__str__()].append(action)
        for _, proposition in action.del_effects:
            if not proposition in related_propositions:
                related_propositions.append(proposition)
                self.prop_name_to_related_prop_action_name[proposition.__str__()].append(action)

        self.prop_action_name_to_related_props_name[action.name] = related_propositions
