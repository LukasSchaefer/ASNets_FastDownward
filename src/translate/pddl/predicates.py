class Predicate(object):
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

    def __str__(self):
        return "%s(%s)" % (self.name, ", ".join(map(str, self.arguments)))

    def get_arity(self):
        return len(self.arguments)

    def get_groundings(self, objects):
        """
        Generates all groundings for this predicate
        :param objects: dict of the form {object type : [object, ...]}
        :return: [(obj1, obj2, ...), ...] Each tuple is a grounding
        """
        groundings = []
        objects_per_argument = []
        index_per_argument = []
        for arg in self.arguments:
            objects_per_argument.append(objects[arg.type_name])
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
            groundings.append(tuple(grounding))
            if carry:
                break
        return groundings