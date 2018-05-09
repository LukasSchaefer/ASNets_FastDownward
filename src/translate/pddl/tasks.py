import logging

from . import axioms
from . import conditions
from . import predicates


class Domain(object):
    def __init__(self, domain_name, requirements,
                 types, constants, predicates, functions,
                 actions, axioms):
        self.domain_name = domain_name
        self.requirements = requirements
        self.types = types
        self.__type_hierarchy = None
        self.__inv_type_hierarchy = None
        self.constants = constants
        self.predicates = predicates
        self.__predicates_dict = None
        self.functions = functions
        self.actions = actions
        self.axioms = axioms
        self.axiom_counter = 0

    def _type_hierarchy(self):
        if self.__type_hierarchy is None:
            lookup = {}
            for type in self.types:
                lookup[type.name] = type.basetype_name
            self.__type_hierarchy = lookup
        return self.__type_hierarchy
    type_hierarchy = property(_type_hierarchy)

    def _inv_type_hierarchy(self):
        if self.__inv_type_hierarchy is None:
            lookup = {}
            for type in self.types:
                if type.basetype_name not in lookup:
                    lookup[type.basetype_name] = set()
                lookup[type.basetype_name].add(type.name)
            self.__inv_type_hierarchy = lookup
        return self.__inv_type_hierarchy
    inv_type_hierarchy = property(_inv_type_hierarchy)

    def _predicate_dict(self):
        if self.__predicates_dict is None:
            pd = {}
            for predicate in self.predicates:
                pd[predicate.name] = predicate
            self.__predicates_dict = pd
        return self.__predicates_dict
    predicate_dict = property(_predicate_dict)

class Task(object):
    def __init__(self, domain_name, task_name, requirements,
                 types, objects, predicates, functions, init, goal,
                 actions, axioms, use_metric):
        self.domain_name = domain_name
        self.task_name = task_name
        self.requirements = requirements
        self.types = types
        self.__type_hierarchy = None
        self.__inv_type_hierarchy = None
        self.objects = objects
        self.__objects_dict_typed = None
        self.__objects_dict_untyped = None
        self.predicates = predicates
        self.__predicates_dict = None
        self.functions = functions
        self.init = init
        self.goal = goal
        self.actions = actions
        self.__actions_dict = None
        self.axioms = axioms
        self.axiom_counter = 0
        self.use_min_cost_metric = use_metric
        self.__predicate_name_to_grounded_predicates = {}
        self.__action_name_to_propositional_actions = {}

    def _type_hierarchy(self):
        if self.__type_hierarchy is None:
            lookup = {}
            for type in self.types:
                lookup[type.name] = type.basetype_name
            self.__type_hierarchy = lookup
        return self.__type_hierarchy
    type_hierarchy = property(_type_hierarchy)

    def _inv_type_hierarchy(self):
        if self.__inv_type_hierarchy is None:
            lookup = {}
            for type in self.types:
                if type.basetype_name not in lookup:
                    lookup[type.basetype_name] = set()
                lookup[type.basetype_name].add(type.name)
            self.__inv_type_hierarchy = lookup
        return self.__inv_type_hierarchy
    inv_type_hierarchy = property(_inv_type_hierarchy)

    def _predicate_dict(self):
        if self.__predicates_dict is None:
            pd = {}
            for predicate in self.predicates:
                pd[predicate.name] = predicate
            self.__predicates_dict = pd
        return self.__predicates_dict
    predicate_dict = property(_predicate_dict)

    def _action_dict(self):
        if self.__actions_dict is None:
            ad = {}
            for action in self.actions:
                ad[action.name] = action
            self.__actions_dict = ad
        return self.__actions_dict

    def _objects_dict_typed(self):
        if self.__objects_dict_typed is None:
            self.__objects_dict_typed = self._get_object_dict(typed=True)
        return self.__objects_dict_typed
    objects_dict_typed = property(_objects_dict_typed)

    def _objects_dict_untyped(self):
        if self.__objects_dict_untyped is None:
            self.__objects_dict_untyped = self._get_object_dict(typed=False)
        return self.__objects_dict_untyped
    objects_dict_untyped = property(_objects_dict_untyped)
    objects_dict = property(_objects_dict_untyped)

    def _get_object_dict(self, typed=True):
        """
        Assigns the objects of the problem in a dictionary to their type
        AND the super types of them!
        The dictionary has the layout: {type_name : object}
        :param typed: if True, then TypedObjects are stored in the dictionary,
                      else the object names are stored
        :return: mapping of object types to objects of this type
        """
        mapping = {}
        for object in self.objects:
            type = object.type_name
            while True:
                if type not in mapping:
                    mapping[type] = []
                if typed:
                    mapping[type].append(object)
                else:
                    mapping[type].append(object.name)
                if type not in self.type_hierarchy:
                    break
                type = self.type_hierarchy[type]
        return mapping

    def add_axiom(self, parameters, condition):
        name = "new-axiom@%d" % self.axiom_counter
        self.axiom_counter += 1
        axiom = axioms.Axiom(name, parameters, len(parameters), condition)
        self.predicates.append(predicates.Predicate(name, parameters))
        self.axioms.append(axiom)
        return axiom

    def dump(self, disp=True, log=logging.root, log_level=logging.INFO):
        msg = ("Problem %s: %s [%s]\n"
               % (self.domain_name, self.task_name, self.requirements))
        msg += "Types:\n"
        for type in self.types:
            msg += "  %s\n" % type
        msg += "Objects:\n"
        for obj in self.objects:
            msg += "  %s\n" % obj
        msg += "Predicates:\n"
        for pred in self.predicates:
            msg += "  %s\n" % pred
        msg += "Functions:\n"
        for func in self.functions:
            msg += "  %s\n" % func
        msg += "Init:\n"
        for fact in self.init:
            msg += "  %s\n" % fact
        msg += "Goal:\n"
        msg += self.goal.dump(disp=False) + "\n"
        msg += "Actions:"
        for action in self.actions:
            msg += "\n" + action.dump(disp=False)
        if self.axioms:
            msg += "\nAxioms:"
            for axiom in self.axioms:
                msg += "\n" + axiom.dump(disp=False)
        if disp:
            log.log(log_level, msg)
        return msg


    def simplify(self, fluent_predicates):
        """
        Remove all predicates in init, predicates, action- condition and effects
        which are not included in fluent_predicates
        :param fluent_predicates: predicates which are reachable/ meaningful in
        simplified task
        """
        # simplify predicates
        simplified_predicates = []
        for pred in self.predicates:
            if pred.name in fluent_predicates:
                simplified_predicates.append(pred)
        self.predicates = simplified_predicates

        # simplify predicate_dict
        self.__predicates_dict = None
        self._predicate_dict()

        # simplify actions
        for action in self.actions:
            action.simplify(fluent_predicates)

        # simplify actions_dict
        self.__actions_dict = None
        self._action_dict()

        # simplify init
        simplified_init = []
        for fact in self.init:
            if isinstance(fact, conditions.Condition):
                simplified_fact = fact.simplify(fluent_predicates)
                if simplified_fact is not None:
                    simplified_init.append(simplified_fact)
            else:
                simplified_init.append(fact)
        self.init = simplified_init

        # simplify goal
        simplified_goal = self.goal.simplify(fluent_predicates)
        if simplified_goal is not None:
            self.goal = simplified_goal
        else:
            self.goal = conditions.Truth()


    # Should probably not be used/ instantiation by instantiate.explore is to be preferred
    # because it makes use of normalization etc. saving a lot of unnecessary/ unachievable
    # propositions etc.
    def get_grounded_predicates(self, sort=False, typed=False):
        """
        Computes (or collects already computed) all groundings of abstract all predicates
        :param sort: if True, then the output will be sorted by predicate name
        :param typed: if True, then grounding considers/ checks argument
                      types
        :return: set of grounded predicates
        """
        groundings = []
        for predicate in self.predicates:
            if predicate.name == "=":
                continue
            if predicate.name in self.__predicate_name_to_grounded_predicates.keys():
                groundings.extend(self.__predicate_name_to_grounded_predicates[predicate.name])
            else:
                grounded_predicates = predicate.get_groundings(
                    self._get_object_dict(), typed, self.type_hierarchy)
                self.__predicate_name_to_grounded_predicates[predicate.name] = grounded_predicates
                groundings.extend(grounded_predicates)
        if sort:
            groundings = sorted(groundings, key=lambda x: str(x))
        return groundings

    # Should probably not be used/ instantiation by instantiate.explore is to be preferred
    # because it makes use of normalization etc. saving a lot of unnecessary/ unachievable
    # propositions etc.
    def str_grounded_predicates(self, grounded_predicates=None, sort=False):
        if grounded_predicates is None:
            grounded_predicates = self.get_grounded_predicates(sort=sorted)
        elif sort:
            grounded_predicates = sorted(grounded_predicates, key=lambda x: str(x))
        return [str(atom) for atom in grounded_predicates]

    # Should probably not be used/ instantiation by instantiate.explore is to be preferred
    # because it makes use of normalization etc. saving a lot of unnecessary/ unachievable
    # propositions etc.
    def get_grounded_predicates_from_predicate(self, predicate):
        """
        Computes (or collects already computed) all groundings of specific predicate
        :param predicate: predicate to compute groundings of
        :return: set of grounded predicates
        """
        if predicate.name in self.__predicate_name_to_grounded_predicates.keys():
            return self.__predicate_name_to_grounded_predicates[predicate.name]
        else:
            grounded_predicates = predicate.get_groundings(self._get_object_dict(),
                typed, self.type_hierarchy)
            self.__predicate_name_to_grounded_predicates[predicate.name] = grounded_predicates
            return grounded_predicates

    # Should probably not be used/ instantiation by instantiate.explore is to be preferred
    # because it makes use of normalization etc. saving a lot of unnecessary/ unachievable
    # propositions etc.
    def get_predicate_to_grounded_predicates_dict(self):
        """
        Computes (or collects already computed) all groundings of all predicates and
        saves these in a dict
        :return: dict mapping from predicate names to grounded predicates
        """
        # compute all groundings
        for predicate in self.predicates:
            if predicate.name == "=":
                continue
            if predicate.name not in self.__predicate_name_to_grounded_predicates.keys():
                self.__predicate_name_to_grounded_predicates[predicate.name] = predicate.get_groundings(
                    self._get_object_dict(), typed, self.type_hierarchy)

        # return filled dict
        return self.__predicate_name_to_grounded_predicates

    # Should probably not be used/ instantiation by instantiate.explore is to be preferred
    # because it makes use of normalization etc. saving a lot of unnecessary/ unachievable
    # propositions etc.
    def get_propositional_actions(self, fluent_facts, sort=False):
        """
        Computes (or collects already computed) all groundings of all abstract actions
        :param fluent_facts: fluent facts (can be computed by translate/instantiate
        :param sort: if True, then the output will be sorted by action name
        :return: set of grounded propositional actions
        """
        propositional_actions = []
        for action in self.actions:
            if action.name in self.__action_name_to_propositional_actions.keys():
                propositional_actions.extend(self.__action_name_to_propositional_actions[action.name])
            else:
                # is fixed metric = False?
                actions = action.get_instantiations(self._get_object_dict(),
                    set(self.init), fluent_facts, False)
                self.__action_name_to_propositional_actions[action.name] = actions
                propositional_actions.extend(actions)
        if sort:
            propositional_actions = sorted(propositional_actions, key=lambda x: str(x))
        return propositional_actions

    # Should probably not be used/ instantiation by instantiate.explore is to be preferred
    # because it makes use of normalization etc. saving a lot of unnecessary/ unachievable
    # propositions etc.
    def get_propositional_actions_from_action(self, action, fluent_facts):
        """
        Computes (or collects already computed) all groundings of specific action
        :param fluent_facts: fluent facts (can be computed by translate/instantiate
        :param action: action to compute groundings of
        :return: set of grounded propositional actions
        """
        if action.name in __action_name_to_propositional_actions.keys():
            return self.__action_name_to_propositional_actions[action.name]
        else:
            # is fixed metric = False and all propositions for fluent_facts okay?
            actions = action.get_instantiations(self._get_object_dict(),
                set(self.init), fluent_facts, False)
            self.__action_name_to_propositional_actions[action.name] = actions
            return actions

    # Should probably not be used/ instantiation by instantiate.explore is to be preferred
    # because it makes use of normalization etc. saving a lot of unnecessary/ unachievable
    # propositions etc.
    def get_action_to_propositional_actions_dict(self):
        """
        Computes (or collects already computed) all groundings of all actions and
        saves these in a dict
        :return: dict mapping from action names to grounded propositional actions
        """
        # compute propositional actions
        for action in self.actions:
            if action.name not in self.__action_name_to_propositional_actions.keys():
                self.__action_name_to_propositional_actions[predicate.name] = action.get_instantiations(
                    self._get_object_dict(), set(self.init), fluent_facts, False)

        # return filled dict
        return self.__action_name_to_propositional_actions


class Requirements(object):
    def __init__(self, requirements):
        self.requirements = requirements
        for req in requirements:
            assert req in (
              ":strips", ":adl", ":typing", ":negation", ":equality",
              ":negative-preconditions", ":disjunctive-preconditions",
              ":existential-preconditions", ":universal-preconditions",
              ":quantified-preconditions", ":conditional-effects",
              ":derived-predicates", ":action-costs"), req
    def __str__(self):
        return ", ".join(self.requirements)
