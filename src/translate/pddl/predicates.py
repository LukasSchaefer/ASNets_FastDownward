from .conditions import Atom

class Predicate(object):
    def __init__(self, name, arguments):
        """
        PDDL Predicate
        :param name: name of predicate as string
        :param arguments: arguments of predicate as TypedObject in an iterable
        """
        self.name = name
        self.arguments = arguments

    def __str__(self):
        return "%s(%s)" % (self.name, ", ".join(map(str, self.arguments)))

    def get_arity(self):
        return len(self.arguments)

    def get_grounding(self, objects, typed=True):
        """
        Get grounding of this predicate using the given objects.
        :param objects: objects for the grounding in their correct order.
        :param typed: True if TypedObjects are provided,
                      False if objects are given as string
        :return: Atom object
        """
        assert len(objects) == self.get_arity(), \
            "Invalid number of objects given for grounding: %d" % len(objects)
        if typed:
            for i in range(self.get_arity()):
                if objects[i].type_name != self.arguments[i].type_name:
                    raise ValueError("Invalid typed object for grounding. "
                                     "Expected %s, got %s"
                                     % (str(objects[i]), str(self.arguments[i]))
                                     )
                objects[i] = objects[i].name
        return Atom(self.name, objects)



    def get_groundings(self, object_dict, typed=True):
        """
        Generate all groundings for this predicate possible with the given
        objects.
        :param object_dict: dictionary of the form {object type : [object, ...]}
        :param typed: if True, the objects TypedObjects,
                      otherwise they are strings
        :return: set of Atoms
        """
        if len(self.arguments) == 0:
            return set([self.get_grounding([], typed=typed)])
        groundings = set()
        objects_per_argument = []
        index_per_argument = []
        for arg in self.arguments:
            objects_per_argument.append(object_dict[arg.type_name])
            index_per_argument.append(0)

        while True:
            grounding = []
            carry = False
            for a in range(len(self.arguments)):
                grounding.append(objects_per_argument[a][index_per_argument[a]])
                if a == 0 or carry:
                    carry = False
                    index_per_argument[a] += 1
                    if index_per_argument[a] == len(objects_per_argument[a]):
                        index_per_argument[a] = 0
                        carry = True
            groundings.add(self.get_grounding(grounding, typed=typed))
            if carry:
                break
        return groundings
