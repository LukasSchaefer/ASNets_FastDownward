import sys
import bisect
sys.path.append("../../")
from src.translate.pddl.conditions import Literal

class ProblemMeta:
    """Contains additional information about relations of actions
    and propositions in a pddl-task used for ASNets"""

    def __init__(self,
                 pddl_task,
                 sas_task,
                 propositional_actions,
                 grounded_predicates):
        self.pddl_task = pddl_task
        self.sas_task = sas_task
        self.propositional_actions = sorted(propositional_actions, key=lambda prop_action: prop_action.name)
        self.grounded_predicates = sorted(grounded_predicates)

        # remove propositional actions in PDDL task which do not occur in the SAS task
        self.__prune_propositional_actions()
        # remove grounded predicates in PDDL task which do not occur in the SAS task
        self.__prune_grounded_predicates()

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

        # computes relations between abstract actions and predicates and sets
        # pred_to_related_actions {pred: [action, ...]} and
        # action_to_related_preds {action: [pred, ...]} accordingly
        self.__compute_and_set_relations_between_abstracts()

        # compute for abstract actions dicts:
        # {action.name -> [(par, par_type), (par2, par_type), ...]
        # with list of pairs as values, each pair corresponding to a action parameter with type
        self.action_name_to_par_type_pairs = self.__compute_action_par_type_dict()

        # computes relations between propositional actions and grounded predicates
        # gr_pred_to_related_prop_action_names {gr_pred: [prop_action.name, ...]} and
        # prop_action_to_related_gr_pred_names {prop_action: [gr_pred.name, ...]} accordingly
        # same fore ..._ids with ids replacing names
        self.gr_pred_to_related_prop_action_names, self.gr_pred_to_related_prop_action_ids, \
        self.prop_action_to_related_gr_pred_names, self.prop_action_to_related_gr_pred_ids =\
            self.__compute_relations_between_groundings()
        

    def __prune_propositional_actions(self):
        """
        Goes over PDDL propositional actions and removes those that do not occur in the SAS task
        and ensures that there are none propositional actions in the SAS task that are missing in
        the PDDL task
        """
        # sort sas task operators alphabetically
        sas_op_names = []
        for op in self.sas_task.operators:
            bisect.insort(sas_op_names, op.name)
        if not all(sas_op_names[i] <= sas_op_names[i+1] for i in range(len(sas_op_names)-1)):
            # list is not sorted
            raise ValueError("SAS operator names list is not sorted!")
        
        num_sas_task_ops = len(sas_op_names)
        num_pddl_task_prop_actions = len(self.propositional_actions)
        print("PDDL task actions: %d" % num_pddl_task_prop_actions)
        print("SAS task operators: %d" % num_sas_task_ops)

        if num_sas_task_ops != num_pddl_task_prop_actions:
            assert num_sas_task_ops < num_pddl_task_prop_actions, "There are more PDDL propositional actions than SAS operators"

            pddl_index = 0
            sas_index = 0
            pddl_actions = self.propositional_actions
            pddl_actions_to_remove = []
            if not all(pddl_actions[i].name <= pddl_actions[i+1].name for i in range(len(pddl_actions)-1)):
                # list is not sorted
                raise ValueError("PDDL propositional actions list is not sorted by name!")

            while pddl_index < len(pddl_actions) and sas_index < len(sas_op_names):
                if pddl_actions[pddl_index].name == sas_op_names[sas_index]:
                    # names are equal -> go on
                    pddl_index += 1
                    sas_index += 1
                    continue
                elif pddl_actions[pddl_index].name < sas_op_names[sas_index]:
                    pddl_actions_to_remove.append(pddl_actions[pddl_index])
                    pddl_index += 1
                else:
                    raise ValueError("Action %s occurs in SAS task but not in PDDL task!" % sas_op_names[sas_index].name)

            if pddl_index < len(pddl_actions):
                # sas_index was at the end of sas op names -> rest of the PDDL prop actions should be removed
                pddl_actions_to_remove.extend(pddl_actions[pddl_index:])
            elif sas_index < len(sas_op_names):
                # pddl_index was at the end of PDDL prop actions but not for SAS ops -> SAS ops from there
                # on do NOT occur in the PDDL prop actions -> error
                raise ValueError("The last %d ops from the SAS task are not included in the PDDL task" % (len(sas_op_names) - sas_index))

            print("%d propositional actions are to be removed from the PDDL task" % len(pddl_actions_to_remove))
            assert (num_pddl_task_prop_actions - len(pddl_actions_to_remove)) == num_sas_task_ops, "There were " +\
                "%d pddl propositional actions and %d sas operators but %d actions are supposed to be removed" %\
                (num_pddl_task_prop_actions, num_sas_task_ops, len(pddl_actions_to_remove))

            # remove the outstanding propositions which do not occur in the SAS task
            for act in pddl_actions_to_remove:
                pddl_actions.remove(act)

            self.propositional_actions = pddl_actions


    def __prune_grounded_predicates(self):
        """
        Goes over PDDL grounded predicates and removes those that do not occur in the SAS task as
        a fact. Ignores all facts in the SAS task which are NegatedAtoms.
        Additionally ensures that there are no facts in the SAS task that are missing in
        the PDDL task
        """
        # sort sas task facts alphabetically and remove NegatedAtom "fake" facts
        sas_fact_names = []
        sas_negated_names = []
        for var_index in range(len(self.sas_task.variables.value_names)):
            for fact_name in self.sas_task.variables.value_names[var_index]:
                # these are strange values created by the PDDL -> SAS task translations
                if fact_name == "<none of those>":
                    continue
                # don't add negated atom facts (no real new facts)
                if not fact_name.startswith("NegatedAtom"):
                    bisect.insort(sas_fact_names, fact_name)
                else:
                    sas_negated_names.append(fact_name)
        if not all(sas_fact_names[i] <= sas_fact_names[i+1] for i in range(len(sas_fact_names)-1)):
            # list is not sorted
            raise ValueError("SAS Fact name list is not sorted!")

        for sas_fact_name in sas_negated_names:
            # remove negated at the beginning of the name
            non_negated_name = sas_fact_name[7:]
            if not non_negated_name in sas_fact_names:
                raise ValueError("Fact %s only appears as a negated fact in the SAS task." % sas_fact_name)

        num_sas_task_facts = len(sas_fact_names)
        num_pddl_task_propositions = len(self.grounded_predicates)
        print("PDDL task propositions: %d" % num_pddl_task_propositions)
        print("SAS task facts: %d" % num_sas_task_facts)

        # remove propositions from PDDL task which do not occur in the SAS task
        if num_sas_task_facts != num_pddl_task_propositions:
            print("Prune PDDL propositions not occuring in the SAS task")
            assert num_sas_task_facts < num_pddl_task_propositions, "There are more PDDL propositions than SAS facts"
            pddl_index = 0
            sas_index = 0
            pddl_propositions = self.grounded_predicates
            pddl_props_to_remove = []
            if not all(pddl_propositions[i].__str__() <= pddl_propositions[i+1].__str__() for i in range(len(pddl_propositions)-1)):
                # list is not sorted
                raise ValueError("PDDL propositions list is not sorted by name!")

            while pddl_index < len(pddl_propositions) and sas_index < len(sas_fact_names):
                if pddl_propositions[pddl_index].__str__() == sas_fact_names[sas_index]:
                    # names are equal -> go on
                    pddl_index += 1
                    sas_index += 1
                    continue
                elif pddl_propositions[pddl_index].__str__() < sas_fact_names[sas_index]:
                    pddl_props_to_remove.append(pddl_propositions[pddl_index])
                    pddl_index += 1
                else:
                    raise ValueError("Proposition %s occurs in SAS task but not in PDDL task!" % sas_fact_names[sas_index])
            
            if pddl_index < len(pddl_propositions):
                # sas_index was at the end of sas fact names -> rest of the PDDL propositions should be removed
                pddl_props_to_remove.extend(pddl_propositions[pddl_index:])
            elif sas_index < len(sas_fact_names):
                # pddl_index was at the end of PDDL propositions but not for SAS facts -> SAS facts from there
                # on do NOT occur in the PDDL propositions -> error
                raise ValueError("The last %d facts from the SAS task are not included in the PDDL task" % (len(sas_fact_names) - sas_index))

            print("%d propositions are to be removed from the PDDL task" % len(pddl_props_to_remove))
            assert (num_pddl_task_propositions - len(pddl_props_to_remove)) == num_sas_task_facts, "There were " +\
                "%d pddl propositions and %d sas facts but %d propositions are supposed to be removed" %\
                (num_pddl_task_propositions, num_sas_task_facts, len(pddl_props_to_remove))

            # remove the outstanding propositions which do not occur in the SAS task
            for prop in pddl_props_to_remove:
                pddl_propositions.remove(prop)

            self.grounded_predicates = pddl_propositions


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
        for act in self.pddl_task.actions:
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
        for pred in self.pddl_task.predicates:
            predicate_name_to_propositions[pred.name] = []

        # fill all lists with corresponding propositions
        for prop in self.grounded_predicates:
            predicate_name_to_propositions[prop.predicate].append(prop)
        return predicate_name_to_propositions


    def __compute_relations_between_groundings(self):
        """
        Computes relations among grounded propositional actions and predicates
        (action and prop are related iff prop appears in pre, add or delete list
        of action)
            - gr_pred_to_related_prop_action_names:
                {gr_pred: related prop_actions.name}
            - gr_pred_to_related_prop_action_ids:
                {gr_pred: related prop_actions_id}
            - prop_action_to_related_gr_pred_names:
                {prop_action: related gr_preds.__str__()}
            - prop_action_to_related_gr_pred_ids:
                {prop_action: related gr_preds_id}
            for gr_pred_to...:
            related propositional action names/ ids are in sub-lists of format so that
            propositional actions instantiated from the same action schema are in the
            same sub-list.
            E.g. [[act1.id, act2.id], [act3.id, ...], ...] if act1 and act2 are groundings
            of the same action schemas

        :return: gr_pred_to_related_prop_action_names,  
                 gr_pred_to_related_prop_action_ids,
                 prop_action_to_related_gr_pred_names,
                 prop_action_to_related_gr_pred_ids

        """
        # Dict of type propositional_action -> list(grounded_predicate)
        prop_action_to_related_gr_pred_names = {}
        # Dict of type grounded_predicate -> list(propositional_actions)
        gr_pred_to_related_prop_action_names = {}

        # initialize dict entries for grounded predicates all with empty list
        # -> if no related actions than value is already correct and no in dict.keys()
        # check necessary in self.__compute_and_set... function below
        for gr_pred in self.grounded_predicates:
            gr_pred_to_related_prop_action_names[gr_pred] = []

        # IMPORTANT: ordered data structure (like list) so that all related-lists
        # of grounded actions with same underlying action schema have corresponding
        # propositions at the same index (necessary for weight sharing in ASNets)
        for propositional_action in self.propositional_actions:
            self.__compute_relations_involving_propositional_action(propositional_action,
                    prop_action_to_related_gr_pred_names,
                    gr_pred_to_related_prop_action_names)


        # setup prop_action_to_related_gr_pred_ids
        prop_action_to_related_gr_pred_ids = {}
        for propositional_action in self.propositional_actions:
            related_gr_pred_names = prop_action_to_related_gr_pred_names[propositional_action]
            related_gr_pred_ids = []
            for gr_pred_name in related_gr_pred_names:
                if gr_pred_name in self.grounded_predicate_name_to_id.keys():
                    related_gr_pred_ids.append(self.grounded_predicate_name_to_id[gr_pred_name])
                else:
                    # add dummy id -1 for pruned grounded predicate
                    related_gr_pred_ids.append(-1)
            prop_action_to_related_gr_pred_ids[propositional_action] = related_gr_pred_ids

        # adapt gr_pred_to_related_prop_action_names to match described format grouping
        # prop action names/ ids of actions sharing the underlying action schema
        for predicate in self.pddl_task.predicates:
            related_action_schemas = self.pred_to_related_actions[predicate]
            for grounded_predicate in self.predicate_name_to_grounded_predicates[predicate.name]:
                related_prop_action_names = gr_pred_to_related_prop_action_names[grounded_predicate]
                gr_pred_to_related_prop_action_names[grounded_predicate] = \
                        self.__format_related_propositional_action_names(related_action_schemas,
                                                                         related_prop_action_names)

        # setup gr_pred_to_related_prop_action_ids
        gr_pred_to_related_prop_action_ids = {}
        for grounded_predicate in self.grounded_predicates:
            related_prop_action_names_lists = gr_pred_to_related_prop_action_names[grounded_predicate]
            related_prop_action_ids_lists = []
            for related_prop_action_names in related_prop_action_names_lists:
                related_prop_action_ids = [self.prop_action_name_to_id[prop_action_name] for\
                        prop_action_name in related_prop_action_names]
                related_prop_action_ids_lists.append(related_prop_action_ids)
            gr_pred_to_related_prop_action_ids[grounded_predicate] = related_prop_action_ids_lists

        return gr_pred_to_related_prop_action_names, gr_pred_to_related_prop_action_ids,\
               prop_action_to_related_gr_pred_names, prop_action_to_related_gr_pred_ids


    def __compute_relations_involving_propositional_action(self,
                                                           propositional_action,
                                                           prop_action_to_related_gr_pred_names,
                                                           gr_pred_to_related_prop_action_names):
        """
        Computes related grounded predicates (propositions) for grounded
        propositional action (= propositions that either appear in action's
        preconditions, add- or delete-list) by instantiating the related predicates
        of the underlying abstract action
        :param propositional_action: propositional action to compute related
            propositions of
        :param prop_action_to_related_gr_pred_names: dict from propositional actions to
            related grounded predicate names
        :param gr_pred_to_related_prop_action_names: dict from grounded predicates to
            related propositional action names
        :return: None
        """
        # get abstract predicates related to the underlying action
        underlying_action = self.pddl_task._action_dict()[propositional_action.get_underlying_action_name()]
        related_preds = self.action_to_related_preds[underlying_action]

        # get par_type list for abstract action
        # contains pair (par_name, par_type) for each action parameter
        par_type_list = self.action_name_to_par_type_pairs[underlying_action.name]
        # get argument names of propositional_action
        arg_name_list = propositional_action.name.rstrip(')').split()[1:]
        assert len(par_type_list) == len(arg_name_list), "There are not the same number of parameters in %s as arguments in %s"\
            % (underlying_action.name, propositional_action.name)
        # get objects used to instantiate action to propositional_action
        # put pairs in list of (par_name, arg_object)
        argument_objects = []
        for par_index, (par_name, par_type) in enumerate(par_type_list):
            type_objects = self.pddl_task._objects_dict_typed()[par_type]
            arg_found = False
            # look for object of par_type with name as indicated in propositional action name (arg_name_list)
            for obj in type_objects:
                if obj.name == arg_name_list[par_index]:
                    argument_objects.append((par_name, obj))
                    arg_found = True
                    break
            if not arg_found:
                raise ValueError("Object with name %s not found!" % arg_name_list[par_index])

        # instantiate related predicates to get corresponding groundings related to propositional_action
        related_propositions = []

        for pred in related_preds:
            type_hierarchy = self.pddl_task._type_hierarchy()
            pred_args = []
            for par in pred.args:
                if par[0] == "?":
                    # usual parameter -> should be among argument_objects (arguments of propositional action)
                    par_found = False
                    for (par_name, arg_object) in argument_objects:
                        if par_name == par:
                            pred_args.append(arg_object)
                            par_found = True
                            break
                    if not par_found:
                        raise ValueError("Parameter %s occurs in predicate %s related to %s but was not found as an parameter of " +\
                            "the corresponding propositional action %s." % (par, pred.__str__(), underlying_action.name, propositional_action.name))
                else:
                    # not a usual argument/parameter -> constant
                    # find object with par name
                    obj_found = False
                    for obj in self.pddl_task.objects:
                        if obj.name == par:
                            pred_args.append(obj)
                            obj_found = True
                            break
                    if not obj_found:
                        raise ValueError("Parameter %s occurs in predicate %s related to %s and seems to be a constant but no " +\
                            "object with the given name was found." % (par, pred.__str__(), underlying_action.name))

            predicate = self.pddl_task._predicate_dict()[pred.predicate]
            related_propositions.append(predicate.get_grounding(pred_args, typed=True, type_hierarchy=type_hierarchy))
        
        # propositional action is related to all these propositions
        for prop in related_propositions:
            if prop in self.grounded_predicates:
                gr_pred_to_related_prop_action_names[prop].append(propositional_action.name)

        # vice-versa all these propositions are related to propositional action
        prop_action_to_related_gr_pred_names[propositional_action] = [prop.__str__() for prop in related_propositions]


    def __format_related_propositional_action_names(self, related_action_schemas, related_prop_action_names):
        """
        format list of related propositional action names for grounded predicate so that
        the propositional action names are in sub-lists of format, so that names of
        propositional actions instantiated from the same action schema are in the same sub-list.
        E.g. [[act1.name, act2.name], [act3.name, ...], ...] if act1 and act2 are groundings
        of the same action schemas
        :param related_action_schemas: list of (abstract) actions related to
            underlying predicate of grounded predicate
        :param related_prop_action_names: list of names of propositional actions, the grounded
            predicate is related to
        """
        # dict from action_schema_name to list of propositional action names
        related_prop_action_name_dict = {}

        # instantiate according to relations of underlying predicate
        for action_schema in related_action_schemas:
            related_prop_action_name_dict[action_schema.name] = []

        for prop_act_name in related_prop_action_names:
            # extract action schema name out of propositional action name
            action_schema_name = prop_act_name.strip('(').split()[0]
            related_prop_action_name_dict[action_schema_name].append(prop_act_name)

        # build related_prop_action_name_list
        related_prop_action_names = []
        for action_schema in related_action_schemas:
            related_action_names = related_prop_action_name_dict[action_schema.name]
            related_prop_action_names.append(related_action_names)

        return related_prop_action_names


    def __compute_action_par_type_dict(self):
        """
        Computes dict holding for each action a list of pairs with each pair 
        corresponding to a single parameter of the action and its type
        {action.name -> [(par, par_type), (par2, par_type), ...]
        :return: computed dict
        """
        action_name_to_par_type_pairs = {}
        for action in self.pddl_task.actions:
            par_list = []
            for par in action.parameters:
                par_list.append((par.name, par.type_name))
            action_name_to_par_type_pairs[action.name] = par_list
        return action_name_to_par_type_pairs
        

    def __compute_and_set_relations_between_abstracts(self):
        """
        Computes relations among abstract actions and predicates
        (action and pred are related iff pred appears in pre, add or delete list
        of action) and sets these in corresponding dicts in both directions
        dicts:
            - pred_to_related_actions:
                {pred: related actions}
            - action_to_related_preds:
                {action: related preds with pars}
        """
        # Dict of type action -> list(predicate)
        self.action_to_related_preds = {}
        # Dict of type predicate -> list(action)
        self.pred_to_related_actions = {}

        # initialize dict entries for predicates all with empty list
        # -> if no related actions than value is already correct and no in dict.keys()
        # check necessary in self.__compute_and_set... function below
        for pred in self.pddl_task.predicates:
            self.pred_to_related_actions[pred] = []

        for action in self.pddl_task.actions:
            self.__compute_and_set_relations_involving_action(action)

        for pred in self.pddl_task.predicates:
            self.pred_to_related_actions[pred] = sorted(self.pred_to_related_actions[pred], key=lambda act_schema: act_schema.name)


    def __compute_and_set_relations_involving_action(self, action):
        """
        Computes related predicates for action (= predicates that either appear
        in action's preconditions, add- or delete-list) and sets these in
        corresponding dicts in both directions
        :param action: action to compute related predicates of
        """
        related_predicates = []
        preconds = []
        if isinstance(action.precondition, Literal):
            preconds = [action.precondition]
        else:
            preconds = action.precondition.parts
        for cond in preconds:
            if cond.negated:
                cond = cond.negate()
            related_predicates.append(cond)
            predicate = self.pddl_task._predicate_dict()[cond.predicate]
            if action not in self.pred_to_related_actions[predicate]:
                self.pred_to_related_actions[predicate].append(action)
        for eff in action.effects:
            literals = eff.get_literals()
            for pred in literals:
                if pred.negated:
                    pred = pred.negate()
                if pred not in related_predicates:
                    if pred.predicate in self.pddl_task._predicate_dict().keys():
                        related_predicates.append(pred)
                        predicate = self.pddl_task._predicate_dict()[pred.predicate]
                        if action not in self.pred_to_related_actions[predicate]:
                            self.pred_to_related_actions[predicate].append(action)
                    else:
                        raise ValueError("Predicate %s is related to action %s, but is not included in list of predicates" % (pred.predicate, action.name))

        self.action_to_related_preds[action] = sorted(related_predicates)
