from ...translate.pddl import Atom


class Format:
    STRING = 0  # just a plain string for the object names
    PATTERN = 1  # string of the pattern type-index
    TUPLE = 2  # (type as str, index as int)
    ATOM = 3  # Atom objects


SPLIT = "-"


def merge_object_type_and_counter(type_name, counter, split=SPLIT):
    return type_name + split + str(counter)


def split_object_type_and_counter(object_name, split=SPLIT):
    split_point = object_name.rfind(split)
    assert split_point != -1, "Unable to split object name in type and counter"
    return object_name[:split_point], int(object_name[split_point + 1:])


def split_objects_type_and_counter(objects, split=SPLIT):
    splitted = []
    for predicate_objects in objects:
        splitted.append([])
        for object in predicate_objects:
            splitted[-1].append(
                split_object_type_and_counter(object, split=split))
    return splitted


def convert_object_to_type_and_counter(object, type_name, previous_objects,
                                       type_counter):
    if object in previous_objects:
        return previous_objects[object]

    if type_name not in type_counter:
        type_counter[type_name] = 1
    new_value = (type_name, type_counter[type_name])
    type_counter[type_name] += 1
    previous_objects[object] = new_value

    return new_value


def convert_objects_to_type_and_counter(objects, types):
    splitted = []
    previous_objects = {}
    type_counter = {}
    for index_predicate in range(len(objects)):
        splitted.append([])
        for index_object in range(len(objects[index_predicate])):
            splitted[-1].append(convert_object_to_type_and_counter
                                (objects[index_predicate][index_object],
                                 types[index_predicate][index_object],
                                 previous_objects, type_counter))
    return splitted


def convert_atom_from_tupled_atom(tupled_atom):
    args = [x[0] + SPLIT + str(x[1]) for x in tupled_atom.args]
    return Atom(tupled_atom.predicate, args)


def canonize_object_lists(objects, input_format=Format.PATTERN,
                          output_format=Format.PATTERN,
                          split=SPLIT,
                          types=None, predicate_names=None):
    """

    :param objects: 2 dimensional list where the first dimension corresponds to
                    the different predicates and the second the the objects for
                    the predicate. Format:
                    [[Object1_1, Object1_2,...], [Object2_1, Object2_2, ...], ...]
    :param input_format: Define what form the Objecti_j entries have in the
                         input. See Format for available formats.

    :param output_format: Define what form the Objecti_j entries have in the
                          output. See Format for available formats.
    :param split: if input format is STRING, this split string is used to
                  separate the type_name from the counter in the object name
    :param types: 2 dimensional list which tells for every argument of every
                  predicate its type name. This should have the same format as
                  objects and is used if the input format cannot be used to
                  deduce the type names (e.g. STRING, ATOMS)
    :param predicate_names: List of names where the i-th name corresponds to the
                            same predicate as the i-th entry in objects. This
                            information might be needed for some output formats
                            (e.g. ATOMS) and will be automatically populated
                            if possible from the input format (e.g. ATOMS)
    :return: list of same nesting with canonicalized object names in the chosen
             format.
    """

    if input_format == Format.STRING:
        objects = convert_objects_to_type_and_counter(objects, types)
    elif input_format == Format.PATTERN:
        objects = split_objects_type_and_counter(objects, split=split)
    elif input_format == Format.ATOM:
        predicate_names = []
        for i in range(len(objects)):
            predicate_names.append(objects[i].predicate)
            objects[i] = list(objects[i].args)
        objects = convert_objects_to_type_and_counter(objects, types)
    elif not input_format == Format.TUPLE:
        raise ValueError("Invalid input format for canonicalization: %d" % input_format)

    map_idx_reduced = {}  # {type : {idx : reduced idx}}
    map_next_reduced = {}  # {type : next new reduced idx}
    for index_predicate in range(len(objects)):
        for index_arg in range(len(objects[index_predicate])):
            (type_name, index) = objects[index_predicate][index_arg]

            # Get the new reduced index
            new_index = None
            if type_name not in map_next_reduced:
                new_index = 1
                map_idx_reduced[type_name] = {index : new_index}
                map_next_reduced[type_name] = new_index + 1
            else:
                if index not in map_idx_reduced[type_name]:
                    new_index = map_next_reduced[type_name]
                    map_idx_reduced[type_name][index] = new_index
                    map_next_reduced[type_name] += 1
                else:
                    new_index = map_idx_reduced[type_name][index]

            # First conversion for output format. Convert single object entry
            new_value = None
            if output_format in [Format.STRING,
                                 Format.PATTERN,
                                 Format.ATOM]:
                new_value = type_name + split + str(new_index)
            elif output_format == Format.TUPLE:
                new_value = (type_name, new_index)
            else:
                raise ValueError("Invalid output format for canonicalization: %d" % output_format)
            objects[index_predicate][index_arg] = new_value

    # Second conversion for output format. Convert data which is more than a single entry
    if output_format == Format.ATOM:
        if predicate_names is None or len(predicate_names) != len(objects):
            raise ValueError("For the canonicalization to output atom objects,"
                             "the name of every atom to output has to be given."
                             "Not more or less names.")
        for i in range(len(objects)):
            objects[i] = Atom(predicate_names[i], objects[i])
    return objects


def fill_templates_with_objects(
        *atoms, **kwargs):
    """

    :param atoms: (SEQUENCE)
    :param object_occurrences: (KEYWORD ARG)
    :param input_format: (KEYWORD ARG)
    :return:
    """
    object_occurrences = kwargs.pop("object_occurrences", None)
    input_format = kwargs.pop("input_format", Format.PATTERN)
    if object_occurrences is None:
        raise ValueError("cannot fill templates with objects if no object "
                         "counts are provided.")
    objects = {}
    for type_name in object_occurrences:
        objects[type_name] = []
        for i in range(object_occurrences[type_name]):
            objects[type_name].append(type_name + SPLIT + str(i + 1))

    filled = set()
    for atom in atoms:
        vars = {}
        for idx_arg in range(len(atom.args)):
            arg = atom.args[idx_arg]
            if arg not in vars:
                vars[arg] = []
            vars[arg].append(idx_arg)

        map_var_idx = {}
        vars_indices = []
        vars_objects = []
        for var in vars:
            map_var_idx[var] = len(vars_indices)
            vars_indices.append(0)
            type_name = None
            if input_format == Format.PATTERN:
                type_name = split_object_type_and_counter(var)[0]
            elif input_format == Format.TUPLE:
                type_name = var[0]
            else:
                raise ValueError("Unsupported input format for fill_templates_with_objects: %d" % input_format)
            vars_objects.append(objects[type_name])


        while True:
            # Check Mutex
            valid = True
            used = set()
            for i in range(len(vars_indices)):
                next_obj = vars_objects[i][vars_indices[i]]
                if next_obj in used:
                    valid = False
                    break
                else:
                    used.add(next_obj)

            if valid:
                # Add new item
                new_arguments = []
                for arg in atom.args:
                    idx_var = map_var_idx[arg]
                    new_arguments.append(vars_objects[idx_var][vars_indices[idx_var]])
                filled.add(Atom(atom.predicate, new_arguments))

            carry_index = 0
            while True:
                carry = False
                vars_indices[carry_index] += 1
                if vars_indices[carry_index] >= len(vars_objects[carry_index]):
                    carry = True
                    vars_indices[carry_index] = 0
                    carry_index += 1

                    # Break because we are back to the first ever encountered value
                    if carry_index >= len(vars_objects):
                        break
                # Finished generation of next entry
                if not carry:
                    break
            # No next entry anymore
            if carry:
                break
    return objects, filled


def for_canonicalized_atoms_subtyping(
        *atoms, **kwargs):
    """

    :param atoms: (SEQUENCE)
    :param kwargsinv_type_hierarchy: (KEYWORD ARG)
    :param callme: (KEYWORD ARG)
    :param input_format: (KEYWORD ARG)
    :param argument_format: (KEYWORD ARG)
    :return:
    """
    inv_type_hierarchy = kwargs.pop("inv_type_hierarchy", None)
    callme = kwargs.pop("callme", None)
    input_format = kwargs.pop("input_format", Format.PATTERN)
    argument_format = kwargs.pop("argument_format", Format.PATTERN)
    if inv_type_hierarchy is None:
        raise ValueError("Inverted type hierarchy (look up of parents) required.")
    if callme is None:
        raise ValueError("callme function required.")

    counters = {}
    objects = {}
    def add_object(var_tuple, idx_atom, idx_arg):
        type, counter = var_tuple
        if not var_tuple in objects:
            objects[var_tuple] = (type, [])
        objects[var_tuple][1].append((idx_atom, idx_arg))
        if not type in counters:
            counters[type] = counter
        else:
            counters[type] = max(counters[type], counter)

    def all_sub_types(type_name):
        subtypes = set()
        todo = [type_name]
        while len(todo) > 0:
            next = todo[-1]
            del todo[-1]
            subtypes.add(next)
            if next in inv_type_hierarchy:
                todo.extend(inv_type_hierarchy[next])
        return subtypes

    for idx_atom in range(len(atoms)):
        for idx_arg in range(len(atoms[idx_atom].args)):
            name = atoms[idx_atom].args[idx_arg]
            if input_format == Format.PATTERN:
                (type_name, idx) = split_object_type_and_counter(name)
                add_object((type_name, idx), idx_atom, idx_arg)
            elif input_format == Format.TUPLE:
                add_object(name, idx_atom, idx_arg)
            else:
                raise ValueError(
                    "Invalid input format for for_canonicalized_atoms_subtyping: %d" % input_format)

    vars = []
    subs = {}
    for var in objects:
        vars.append(var)
        type_name = objects[var][0]
        subs[var] = list(all_sub_types(type_name))

    # Start objects for atoms
    idx_obj_per_var = [0 for _ in vars]
    # Set up nested arrays
    chosen_vars = []
    for atom in atoms:
        chosen_vars.append([])
        for _ in atom.args:
            chosen_vars[-1].append(None)

    while True:
        #Replace objects in atoms

        for idx_var in range(len(vars)):
            var = vars[idx_var]
            var_type = subs[var][idx_obj_per_var[idx_var]]
            entries = objects[var][1]
            for (idx_atom, idx_arg) in entries:
                #if argument_format in [Format.PATTERN, Format.STRING]:
                #    chosen_vars[idx_atom][idx_arg] = var_type + SPLIT + str(idx_var)
                #elif argument_format == Format.TUPLE:
                chosen_vars[idx_atom][idx_arg] = (var_type, idx_var)
                #else:
                #    raise ValueError("Invalid argument format for canonicalized atoms subtyping: %d" % argument_format)

        chosen_vars = canonize_object_lists(chosen_vars, input_format=Format.TUPLE,
                                            output_format=argument_format)
        new_atoms = []
        for idx_atom in range(len(atoms)):
            atom = atoms[idx_atom]
            new_atoms.append(Atom(atom.predicate, chosen_vars[idx_atom]))

        callme(*new_atoms)
        # Get next indices to determine the objects
        idx_var = 0
        while True:
            carry = False
            idx_obj_per_var[idx_var] += 1
            if idx_obj_per_var[idx_var] >= len(subs[vars[idx_var]]):
                carry = True
                idx_obj_per_var[idx_var] = 0
                idx_var += 1


                # Break because we are back to the first ever encountered value
                if idx_var >= len(vars):
                    break
            # Finished generation of next entry
            if not carry:
                break
        # No next entry anymore
        if carry:
            break



def for_canonicalized_groundings(*predicates, **kwargs):
    """
    Tells which groundings of the given two predicates interact. For grounding
    the predicates placeholder objects are used. The interaction condition
    shall be invariant w.r.t. renaming the objects
    :param *predicates: sequence of predicates for which to determine their
                        interaction as group. More than two predicates can be
                        provided, but then the interaction condition cinteract
                        and the InteractionMap interaction_map have to be able
                        to handle this.
    :param type_hierarchy: hierarchy of types as dictionary {type : parent}
    :param callme: callable object which will be called for all groundings with
                   callme(*Atom_object)
    :param argument_format: Format for the predicate sequence feeded to 'callme'
    :return: None
    """
    type_hierarchy = kwargs.pop("type_hierarchy", None)
    callme = kwargs.pop("callme", None)
    argument_format = kwargs.pop("argument_format", Format.PATTERN)
    if type_hierarchy is None:
        raise ValueError("Type hierarchy (look up of parents) required.")
    if callme is None:
        raise ValueError("callme function required.")


    predicates = sorted(predicates, key=lambda x: x.name)

    # Count number of occurrences of types in both predicates argument list
    type_counts = {}
    for predicate in predicates:
        for arg in predicate.arguments:
            if arg.type_name not in type_counts:
                type_counts[arg.type_name] = 0
            type_counts[arg.type_name] += 1

    # Create sufficient objects to assign every argument a different object
    # Objects are also added for the type_name of their super types.
    object_dict = {}
    for type_name in type_counts:
        for c in range(type_counts[type_name]):
            object_name = (type_name, c + 1)
            super_types = type_name
            while super_types is not None:
                # Add only if necessary
                if super_types in type_counts:
                    if super_types not in object_dict:
                        object_dict[super_types] = []
                    object_dict[super_types].append(object_name)
                super_types = type_hierarchy[super_types]
    # Create all different groundings
    # (omitting groundings redundant due to renaming)

    # [[[Object list for Arg1 of Pred1], [Arg2], ...]
    #  [[Object list for Arg1 of Pred2], [Arg2], ...]]
    objects_per_argument = []
    # [[Index for object of Arg1 of Pred1, Arg2, ...]
    #  [Index for object of Arg1 of Pred2, Arg2, ...]]
    index_per_argument = []
    for index_predicate in range(len(predicates)):
        objects_per_argument.append([])
        index_per_argument.append([])
        for arg in predicates[index_predicate].arguments:
            objects_per_argument[index_predicate].append(object_dict[arg.type_name])
            index_per_argument[index_predicate].append(0)

    # Every ground generates a new grounding for the predicates
    previous_str_groundings = set()  # used for redundancy checks
    while True:
        # Get arguments for grounding the predicates and check for redundancy.
        # [[(type_name, counter) for Arg_i of Pred1] [And for Pred2]]
        grounding_arguments = []
        for index_predicate in range(len(predicates)):
            predicate = predicates[index_predicate]
            grounding_arguments.append([])

            for index_argument in range(predicate.get_arity()):
                index_object = index_per_argument[index_predicate][index_argument]
                grounding_arguments[-1].append(objects_per_argument[index_predicate][index_argument][index_object])

        grounding_arguments = canonize_object_lists(grounding_arguments,
                                                    input_format=Format.TUPLE,
                                                    output_format=argument_format)

        # if not redundant check interactions
        str_gnd_args = str(grounding_arguments)
        if str_gnd_args not in previous_str_groundings:
            previous_str_groundings.add(str_gnd_args)

            grounded_predicates = []
            for index_predicate in range(len(predicates)):
                grounded_predicates.append(
                    Atom(predicates[index_predicate].name,
                         grounding_arguments[index_predicate]))
            callme(*grounded_predicates)


        # Get next indices to determine the arguments for grounding
        index_predicate = 0
        index_argument = 0
        # Each iteration increments the value in [index_predicate][index_arguemnt]
        # If an overflow happens in this entry, its value is reseted and the next
        # iteration processes the next entry. If no overflow happened,
        # then the loop breaks.
        while True:
            carry = False
            index_per_argument[index_predicate][index_argument] += 1
            if index_per_argument[index_predicate][index_argument] >= len(objects_per_argument[index_predicate][index_argument]):
                carry = True
                index_per_argument[index_predicate][index_argument] = 0
                index_argument += 1

                if index_argument >= predicates[index_predicate].get_arity():
                    index_argument = 0
                    index_predicate += 1
                    while (index_predicate < len(predicates)
                           and predicates[index_predicate].get_arity() == 0):
                        index_predicate += 1

                    # Break because we are back to the first ever encountered value
                    if index_predicate >= len(predicates):
                        break
            # Finished generation of next entry
            if not carry:
                break
        # No next entry anymore
        if carry:
            break

