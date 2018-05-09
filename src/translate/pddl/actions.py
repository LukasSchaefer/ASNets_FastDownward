from __future__ import print_function

import copy
import logging

from . import conditions


class Action(object):
    def __init__(self, name, parameters, num_external_parameters,
                 precondition, effects, cost):
        assert 0 <= num_external_parameters <= len(parameters)
        self.name = name
        self.parameters = parameters
        # num_external_parameters denotes how many of the parameters
        # are "external", i.e., should be part of the grounded action
        # name. Usually all parameters are external, but "invisible"
        # parameters can be created when compiling away existential
        # quantifiers in conditions.
        self.num_external_parameters = num_external_parameters
        self.precondition = precondition
        self.effects = effects
        self.cost = cost
        self.uniquify_variables() # TODO: uniquify variables in cost?

    def __repr__(self):
        return "<Action %r at %#x>" % (self.name, id(self))

    def dump(self, disp=True, log=logging.root, log_level=logging.INFO):
        msg = "%s(%s)\n" % (self.name, ", ".join(map(str, self.parameters)))
        msg += "Precondition:\n"
        msg += self.precondition.dump(disp=False) + "\n"
        msg += "Effects:\n"
        for eff in self.effects:
            msg += eff.dump(disp=False) + "\n"
        msg += "Cost:\n"
        if(self.cost):
            msg += self.cost.dump(disp=False)
        else:
            msg += "  None"
        if disp:
            log.log(log_level, msg)
        return msg

    def uniquify_variables(self):
        self.type_map = dict([(par.name, par.type_name)
                              for par in self.parameters])
        self.precondition = self.precondition.uniquify_variables(self.type_map)
        for effect in self.effects:
            effect.uniquify_variables(self.type_map)

    def relaxed(self):
        new_effects = []
        for eff in self.effects:
            relaxed_eff = eff.relaxed()
            if relaxed_eff:
                new_effects.append(relaxed_eff)
        return Action(self.name, self.parameters, self.num_external_parameters,
                      self.precondition.relaxed().simplified(),
                      new_effects)

    def untyped(self):
        # We do not actually remove the types from the parameter lists,
        # just additionally incorporate them into the conditions.
        # Maybe not very nice.
        result = copy.copy(self)
        parameter_atoms = [par.to_untyped_strips() for par in self.parameters]
        new_precondition = self.precondition.untyped()
        result.precondition = conditions.Conjunction(parameter_atoms + [new_precondition])
        result.effects = [eff.untyped() for eff in self.effects]
        return result

    def instantiate(self, var_mapping, init_facts, fluent_facts,
        objects_by_type, metric):
        """Return a PropositionalAction which corresponds to the instantiation of
        this action with the arguments in var_mapping. Only fluent parts of the
        conditions (those in fluent_facts) are included. init_facts are evaluated
        whilte instantiating.
        Precondition and effect conditions must be normalized for this to work.
        Returns None if var_mapping does not correspond to a valid instantiation
        (because it has impossible preconditions or an empty effect list.)"""
        arg_list = [var_mapping[par.name]
                    for par in self.parameters[:self.num_external_parameters]]
        name = "(%s %s)" % (self.name, " ".join(arg_list))

        precondition = []
        try:
            self.precondition.instantiate(var_mapping, init_facts,
                                          fluent_facts, precondition)
        except conditions.Impossible:
            return None
        effects = []
        for eff in self.effects:
            eff.instantiate(var_mapping, init_facts, fluent_facts,
                            objects_by_type, effects)
        if effects:
            if metric:
                if self.cost is None:
                    cost = 0
                else:
                    cost = int(self.cost.instantiate(var_mapping, init_facts).expression.value)
            else:
                cost = 1
            return PropositionalAction(name, precondition, effects, cost)
        else:
            return None

    def get_instantiations(self, objects_by_type, init_facts, fluent_facts, metric):
        """
        Generate all groundings for this action possible with the given objects.
        :param objects_by_type: dictionary of the form {object type : [object, ...]}
        :param init_facts: facts from the initial state (?)
        :param fluent_facts: ? (all propositions not included here are ignored in
            preconditions (?))
        :param metric: boolean value that defines cost (False -> unit cost)
        :return: set of propositional actions
        """
        if len(self.parameters) == 0:
            return set([self.instantiate({}, init_facts, fluent_facts, objects_by_type, metric)])
        propositional_actions = set()
        objects_per_parameter = []
        index_per_parameter = []
        for param in self.parameters:
            objects_per_parameter.append(objects_by_type[param.type_name])
            index_per_parameter.append(0)

        while True:
            var_mapping = {}
            carry = False
            for index, par in enumerate(self.parameters):
                var_mapping[par.name] = objects_per_parameter[index][index_per_parameter[index]].name
                if index == 0 or carry:
                    carry = False
                    index_per_parameter[index] += 1
                    if index_per_parameter[index] == len(objects_per_parameter[index]):
                        index_per_parameter[index] = 0
                        carry = True
            propositional_action = self.instantiate(var_mapping, init_facts, fluent_facts, objects_by_type, metric)
            if propositional_action is not None:
                propositional_actions.add(propositional_action)
            if carry:
                break
        return propositional_actions
        


class PropositionalAction:
    def __init__(self, name, precondition, effects, cost):
        self.name = name
        self.precondition = precondition
        self.add_effects = []
        self.del_effects = []
        for condition, effect in effects:
            if not effect.negated:
                self.add_effects.append((condition, effect))
        # Warning: This is O(N^2), could be turned into O(N).
        # But that might actually harm performance, since there are
        # usually few effects.
        # TODO: Measure this in critical domains, then use sets if acceptable.
        for condition, effect in effects:
            if effect.negated and (condition, effect.negate()) not in self.add_effects:
                self.del_effects.append((condition, effect.negate()))
        self.cost = cost

    def __repr__(self):
        return "<PropositionalAction %r at %#x>" % (self.name, id(self))

    def get_underlying_action_name(self):
        return self.name.strip('(').split()[0]

    def dump(self, disp=True, log=logging.root, log_level=logging.INFO):
        msg = self.name
        for fact in self.precondition:
            msg += "\nPRE: %s" % fact
        for cond, fact in self.add_effects:
            msg += "\nADD: %s -> %s" % (", ".join(map(str, cond)), fact)
        for cond, fact in self.del_effects:
            msg += "\nDEL: %s -> %s" % (", ".join(map(str, cond)), fact)
        msg += "\ncost: %s" % self.cost
        if disp:
            log.log(log_level, msg)
        return msg
