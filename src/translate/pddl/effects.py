from __future__ import print_function

import logging

from . import conditions

def cartesian_product(*sequences):
    # TODO: Also exists in tools.py outside the pddl package (defined slightly
    #       differently). Not good. Need proper import paths.
    if not sequences:
        yield ()
    else:
        for tup in cartesian_product(*sequences[1:]):
            for item in sequences[0]:
                yield (item,) + tup


class Effect(object):
    def __init__(self, parameters, condition, literal):
        self.parameters = parameters
        self.condition = condition
        self.literal = literal
    def __eq__(self, other):
        return (self.__class__ is other.__class__ and
                self.parameters == other.parameters and
                self.condition == other.condition and
                self.literal == other.literal)
    def dump(self, disp=True, log=logging.root, log_level=logging.INFO):
        indent = "  "
        msg = ""
        if self.parameters:
            msg += "%sforall %s\n" % (indent, ", ".join(map(str, self.parameters)))
            indent += "  "
        if self.condition != conditions.Truth():
            msg += "%sif\n" % indent
            msg += self.condition.dump(indent + "  ", disp=False) + "\n"
            msg += "%sthen\n" % indent
            indent += "  "
        msg += "%s%s" % (indent, self.literal)
        if disp:
            log.log(log_level, msg)
        return msg

    def copy(self):
        return Effect(self.parameters, self.condition, self.literal)
    def uniquify_variables(self, type_map):
        renamings = {}
        self.parameters = [par.uniquify_name(type_map, renamings)
                           for par in self.parameters]
        self.condition = self.condition.uniquify_variables(type_map, renamings)
        self.literal = self.literal.rename_variables(renamings)
    def instantiate(self, var_mapping, init_facts, fluent_facts,
                    objects_by_type, result):
        if self.parameters:
            var_mapping = var_mapping.copy() # Will modify this.
            object_lists = [objects_by_type.get(par.type_name, [])
                            for par in self.parameters]
            for object_tuple in cartesian_product(*object_lists):
                for (par, obj) in zip(self.parameters, object_tuple):
                    var_mapping[par.name] = obj
                self._instantiate(var_mapping, init_facts, fluent_facts, result)
        else:
            self._instantiate(var_mapping, init_facts, fluent_facts, result)
    def _instantiate(self, var_mapping, init_facts, fluent_facts, result):
        condition = []
        try:
            self.condition.instantiate(var_mapping, init_facts, fluent_facts, condition)
        except conditions.Impossible:
            return
        effects = []
        self.literal.instantiate(var_mapping, init_facts, fluent_facts, effects)
        assert len(effects) <= 1
        if effects:
            result.append((condition, effects[0]))
    def relaxed(self):
        if self.literal.negated:
            return None
        else:
            return Effect(self.parameters, self.condition.relaxed(), self.literal)
    def simplified(self):
        return Effect(self.parameters, self.condition.simplified(), self.literal)
    def get_literals(self):
        literals = [self.literal]
        if isinstance(self.condition, conditions.Literal):
            literals.append(self.condition)
        else:
            literals += self.condition.parts
        return literals


class ConditionalEffect(object):
    def __init__(self, condition, effect):
        if isinstance(effect, ConditionalEffect):
            self.condition = conditions.Conjunction([condition, effect.condition])
            self.effect = effect.effect
        else:
            self.condition = condition
            self.effect = effect
    def dump(self, indent="  ", disp=True, log=logging.root, log_level=logging.INFO):
        msg = "%sif\n" % (indent)
        msg += self.condition.dump(indent + "  ", disp=False) + "\n"
        msg += "%sthen\n" % (indent)
        msg += self.effect.dump(indent + "  ", disp=False)
        if disp:
            log.log(log_level, msg)
        return msg
    def normalize(self):
        norm_effect = self.effect.normalize()
        if isinstance(norm_effect, ConjunctiveEffect):
            new_effects = []
            for effect in norm_effect.effects:
                assert isinstance(effect, SimpleEffect) or isinstance(effect, ConditionalEffect)
                new_effects.append(ConditionalEffect(self.condition, effect))
            return ConjunctiveEffect(new_effects)
        elif isinstance(norm_effect, UniversalEffect):
            child = norm_effect.effect
            cond_effect = ConditionalEffect(self.condition, child)
            return UniversalEffect(norm_effect.parameters, cond_effect)
        else:
            return ConditionalEffect(self.condition, norm_effect)
    def extract_cost(self):
        return None, self
    def get_literals(self):
        literals = self.effect.get_literals()
        if isinstance(self.condition, conditions.Literal):
            literals.append(self.condition)
        else:
            literals += self.condition.parts
        return literals

class UniversalEffect(object):
    def __init__(self, parameters, effect):
        if isinstance(effect, UniversalEffect):
            self.parameters = parameters + effect.parameters
            self.effect = effect.effect
        else:
            self.parameters = parameters
            self.effect = effect
    def dump(self, indent="  ", disp=True, log=logging.root):
        msg = "%sforall %s\n" % (indent, ", ".join(map(str, self.parameters)))
        msg += self.effect.dump(indent + "  ", disp=False)
        if disp:
            log.info(msg)
        return msg
    def normalize(self):
        norm_effect = self.effect.normalize()
        if isinstance(norm_effect, ConjunctiveEffect):
            new_effects = []
            for effect in norm_effect.effects:
                assert isinstance(effect, SimpleEffect) or isinstance(effect, ConditionalEffect)\
                       or isinstance(effect, UniversalEffect)
                new_effects.append(UniversalEffect(self.parameters, effect))
            return ConjunctiveEffect(new_effects)
        else:
            return UniversalEffect(self.parameters, norm_effect)
    def extract_cost(self):
        return None, self
    def get_literals(self):
        return self.effect.get_literals()


class ConjunctiveEffect(object):
    def __init__(self, effects):
        flattened_effects = []
        for effect in effects:
            if isinstance(effect, ConjunctiveEffect):
                flattened_effects += effect.effects
            else:
                flattened_effects.append(effect)
        self.effects = flattened_effects
    def dump(self, indent="  ", disp=True, log=logging.root, log_level=logging.INFO):
        msg = "%sand" % (indent)
        for eff in self.effects:
            msg += "\n" + eff.dump(indent + "  ", disp=False)
        if disp:
            log.log(log_level, msg)
        return msg
    def normalize(self):
        new_effects = []
        for effect in self.effects:
            new_effects.append(effect.normalize())
        return ConjunctiveEffect(new_effects)
    def extract_cost(self):
        new_effects = []
        cost_effect = None
        for effect in self.effects:
            if isinstance(effect, CostEffect):
                cost_effect = effect
            else:
                new_effects.append(effect)
        return cost_effect, ConjunctiveEffect(new_effects)
    def get_literals(self):
        literals = []
        for effect in self.effects:
            literals += effect.get_literals()
        return literals

class SimpleEffect(object):
    def __init__(self, effect):
        self.effect = effect
    def dump(self, indent="  ", disp=True, log=logging.root, log_level=logging.INFO):
        msg = "%s%s" % (indent, self.effect)
        if disp:
            log.log(log_level, msg)
        return msg
    def normalize(self):
        return self
    def extract_cost(self):
        return None, self
    def get_literals(self):
        return self.effect.get_literals()


class CostEffect(object):
    def __init__(self, effect):
        self.effect = effect
    def dump(self, indent="  ", disp=True, log=logging.root, log_level=logging.INFO):
        msg = "%s%s" % (indent, self.effect)
        if disp:
            log.log(log_level, msg)
        return msg
    def normalize(self):
        return self
    def extract_cost(self):
        return self, None # this would only happen if
    #an action has no effect apart from the cost effect
    def get_literals(self):
        return self.effect.get_literals()
