from __future__ import print_function

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

    def get_object_type_mapping(self):
        mapping = {}
        for object in self.objects:
            type = object.type_name
            while True:
                if type not in mapping:
                    mapping[type] = []
                mapping[type].append(object)
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

    def dump(self):
        print("Problem %s: %s [%s]" % (
            self.domain_name, self.task_name, self.requirements))
        print("Types:")
        for type in self.types:
            print("  %s" % type)
        print("Objects:")
        for obj in self.objects:
            print("  %s" % obj)
        print("Predicates:")
        for pred in self.predicates:
            print("  %s" % pred)
        print("Functions:")
        for func in self.functions:
            print("  %s" % func)
        print("Init:")
        for fact in self.init:
            print("  %s" % fact)
        print("Goal:")
        self.goal.dump()
        print("Actions:")
        for action in self.actions:
            action.dump()
        if self.axioms:
            print("Axioms:")
            for axiom in self.axioms:
                axiom.dump()

    def get_groundings(self, ignore_equal=True):
        groundings = set()
        for pred in self.predicates:
            if pred.name == "=":
                continue
            assignments = pred.get_groundings(self.get_object_type_mapping())
            for assignment in assignments:
                atom = conditions.Atom(pred, assignment)
                groundings.add(atom)
        return groundings

    def str_groundings(self, groundings=None, ignore_equal=True):
        if groundings is None:
            groundings = self.get_groundings(ignore_equal=ignore_equal)
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
