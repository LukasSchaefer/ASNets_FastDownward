import logging

from . import axioms
from . import conditions
from . import predicates


class Task(object):
    def __init__(self, domain_name, task_name, requirements,
                 types, objects, predicates, functions, init, goal,
                 actions, axioms, use_metric):
        self.domain_name = domain_name
        self.task_name = task_name
        self.requirements = requirements
        self.types = types
        self.objects = objects
        self.predicates = predicates
        self.functions = functions
        self.init = init
        self.goal = goal
        self.actions = actions
        self.axioms = axioms
        self.axiom_counter = 0
        self.use_min_cost_metric = use_metric
        self.type_parent = self.build_type_lookup()

    def build_type_lookup(self):
        # parent type has to defined before child
        lookup = {}
        for type in self.types:
            lookup[type.name] = type.basetype_name
        return lookup

    def get_object_type_mapping(self, as_typed_objects=True):
        mapping = {}
        for object in self.objects:
            type = object.type_name
            while True:
                if type not in mapping:
                    mapping[type] = []
                if as_typed_objects:
                    mapping[type].append(object)
                else:
                    mapping[type].append(object.name)
                if type not in self.type_parent:
                    break
                type = self.type_parent[type]
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


    def get_groundings(self):
        groundings = set()
        for pred in self.predicates:
            if pred.name == "=":
                continue
            assignments = pred.get_groundings(self.get_object_type_mapping(as_typed_objects=False))
            for assignment in assignments:
                atom = conditions.Atom(pred.name, assignment)
                groundings.add(atom)
        return groundings

    def str_groundings(self, groundings=None):
        if groundings is None:
            groundings = self.get_groundings()
        str_groundings = set()
        for atom in groundings:
            str_groundings.add(str(atom))
        return str_groundings



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
