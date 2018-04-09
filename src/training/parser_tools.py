class ArgumentException(Exception):
    pass


class Register(object):
    """
    register = {Class : ClassRegister}
    ClassRegister has register of format: {Key: Class}

    """

    class ClassRegister(object):
        def __init__(self, clazz, register=None):
            self._clazz = clazz
            self._register = {} if register is None else register

        def get_register(self):
            return self._register

        def has_key(self, key):
            return key in self._register

        def _add_key(self, key, clazz):
            if self.has_key(key):
                raise KeyError("The key '" + str(key) + "' used to register " 
                               "class '" + str(clazz) + "' at the register of"
                               " '" + str(self._clazz) + "' is already in use.")
            self._register[key] = clazz

        def get_reference(self, name):
            if name in self._register:
                return self._register[name]
            else:
                raise ArgumentException(
                    "No class reference found for " + str(name)
                    + " in " + str(self._clazz))

        def is_class_registered(self, obj_or_clazz):
            clazz = obj_or_clazz if type(obj_or_clazz) == type else type(obj_or_clazz)

            for key in self._register:
                if self.get_reference(key) == clazz:
                    return True
            return False


    def __init__(self, register=None):
        self._register = {} if register is None else register

    def has_class(self, clazz):
        return clazz in self._register

    def _add_register(self, clazz):
        if self.has_class(clazz):
            raise KeyError("Cannot add a new register for class '" + str(clazz)
                           + "', because this class is already registered.")
        self._register[clazz] = self.ClassRegister(clazz)

    def get_register(self, clazz):
        if self.has_class(clazz):
            return self._register[clazz]
        else:
            raise KeyError(
                "No class register is defined for the desired class: "
                + str(clazz))

    def has_key(self, clazz, key):
        if not self.has_class(clazz):
            raise KeyError("The class '" + str(clazz) + "' in which to look for"
                           + " a key is not  registered.")

        return self.get_register(clazz).has_key(key)

    def _add_key(self, ancestor, key, clazz):
        if not self.has_class(ancestor):
            raise KeyError("The class '" + str(ancestor) + "' for which to add "
                           + "a  key is not  registered.")
        self.get_register(ancestor)._add_key(key, clazz)

        return self.get_register(clazz).has_key(key)

    def get_reference(self, clazz, name):
        return self.get_register(clazz).get_reference(name)

    def append_register(self, clazz, *args):
        """
        Register the given class via the given key words.

        :param clazz: clazz to register
        :param args: names under which the clazz shall be found
        :return: None
        """
        print(clazz)
        ancestor = clazz
        while ancestor is not None:
            if not self.has_class(ancestor):
                self._add_register(ancestor)

            for key in args:
                key = key.lower()
                if not self.has_key(ancestor, key):
                    self._add_key(ancestor, key, clazz)
                else:
                    raise KeyError(
                        "Internal Error: In a register are multiple times "
                        + "items for the same key (" + key + ") " + " for "
                        + "category " + str(ancestor) + " defined.")

            try:
                ancestor = ancestor.__base__
            except AttributeError:
                break



main_register = Register()


class ItemCache(object):
    def __init__(self, item_cache=None):
        if item_cache is not None:
            self._cache = item_cache._cache
        else:
            # {class : {id : obj}}
            self._cache = {}
            self._cache["global"] = {}

        # {Class: set(itself + subclasses)}
        self._class_hierarchies = None
        self._initialize_children()

    def _initialize_children(self):
        self._class_hierarchies = {}
        for clazz in self._cache:
            ancestor = clazz
            while ancestor != object:
                if not ancestor in self._class_hierarchies:
                    self._class_hierarchies[ancestor] = set()
                self._class_hierarchies[ancestor].add(clazz)

                try:
                    ancestor = ancestor.__base__
                except AttributeError:
                    break

        self._class_hierarchies["global"] = set(["global"])

    def to_string(self):
        s = "ItemCache\n"
        for clazz in self._cache:
            s += "\t" + str(clazz) + "\n"
            for key in self._cache[clazz]:
                s += ("\t"*2 + str(key) + " - "
                      + str(self._cache[clazz][key]) + "\n")
        return s

    def _add(self, id, item, ancestor, clazz):
        if id in self._cache[ancestor]:
            raise ArgumentException("The id '" + str(id) + "' given to "
                                    + "cache '" + str(item) + "' is already"
                                    + " used for an item of class "
                                    + str(ancestor) + ".")
        else:
            self._cache[ancestor][id] = item
            if clazz is not None:
                self._class_hierarchies[ancestor].add(clazz)

    def add(self, id, item, glob=False):
        if item is None:
            raise ArgumentException("An item of value None cannot be cached.")

        if glob:
            self._add(id, item, "global", None)
            return

        clazz = type(item)
        ancestor = clazz

        while ancestor != object:
            if ancestor not in self._cache:
                self._cache[ancestor] = {}
                self._class_hierarchies[ancestor] = set()

            self._add(id, item, ancestor, clazz)

            try:
                ancestor = ancestor.__base__
            except AttributeError:
                break

    def _get(self, clazz, id):
        if clazz is None:
            clazz = "global"

        if clazz not in self._class_hierarchies:
            return None
        for child in self._class_hierarchies[clazz]:
            if id in self._cache[child]:
                return self._cache[child][id]
        return None

    def has(self, clazz, id):
        return self._get(clazz, id) is not None

    def get(self, clazz, id, register=None, silent=False):
        """
        Fetches a cached object

        :param clazz: Class for which to fetch an object. The objects fetched
                        can be of class clazz or of its subclasses.
        :param id: id to extract
        :param register: [default None] if provided, then it is checked that the
                            extracted object could have been produced by the
                            register.
        :param silent: [default False] If silent, then not finding an object
                        for the id will return None. If not silent, this will
                        raise an exception.
        :return: object cached for class clazz with id or None
        """
        obj = self._get(clazz, id)
        if obj is None:
            if silent:
                return None
            else:
                raise ArgumentException("Unable to look up '"
                                        + ('global' if clazz is None else str(clazz))
                                        + "' with id '" + str(id) + "'.")
        else:
            if register is None or register.is_class_registered(obj):
                return obj
            else:
                raise ArgumentException("The usage of " + str(id)
                                        + " is invalid here, because its type "
                                        + " could not be produced by the given"
                                        + " register.")

    def get_from_empty_tree(self, clazz, tree, register=None, silent=False):
        if clazz is not None and tree.data[1] != 'id':
            raise ArgumentException("Invalid look up operation. Tree lookup "
                                    "needs an empty tree and if looking up a"
                                    "non global object, then the parameter"
                                    "of the tree node must be 'id'.")
        obj = self.get(clazz, tree.data[0], register, silent)
        if obj is not None and not tree.empty():
            raise ArgumentException(str(tree.data[0]) + " is a previously "
                                        "cached variable. This variable cannot "
                                        "be given parameters again.")
        return obj

    def apply_on_all(self, func):
        for key in self._cache:
            if key == "global" or key.__base__ == object:
                for item in self._cache[key]:
                    func(item)


class ClassArguments:
    def __init__(self, class_name, base_class_arguments, *args, variables = {},
                 order=None):
        """
        List of arguments which a class needs to be constructe. Each entry is
        of the form (name, optional, default, register_or_converter) with:
        name = name of the argument
        optional = if not optional, then the user has to define this field else
                    if the field is missing, the default value is used.
        default = default value to use if the field is not specified
        register_or_converter = if a register is provided (dictionary mapping to
                                classes), then the subtree is given to the parse
                                method of the mapped class. Else the value is
                                interpreted as function and the data of the
                                subtrees root is feed into the node.
                                If the subtrees root is a list, then this is
                                done for every child of it and the results are
                                put into a list.
                                The register can be a list of registers (then
                                the object has to be uniquely defined in one of
                                them)

        :param class_name: name of the class associated with this object
        :param args: sequence of (name, optional, default,
                        register_or_converter) tuple
                        register_or_converter can be:
                            None => the received value is not further processed
                            [Register.ClassRegister, ...]
                                => uses an applicable class register out of
                                   the list of give class registers. If multiple
                                   fit an exception is raised
                            Register.ClassRegister
                                => same as previous, because it will be
                                internally converted to the previous case
                            Callable => calls the callable with the received
                                        data.
        :param variables: Optional. Describes the variables which the associated
                            class may use to provide other objects access to
                            some data. The format is
                            [(name, initial value, value type)].
                            The order of variables is first come the variables
                            from the base_class_arguments (in the order defined
                            there), then are added the variables defined here.
                            Some variable description has the same name
                            as a previous one, then it modifies the previous one
                            and does not appear again in the order.
        :param order: defines a new order for the args order.
        """
        self.class_name = class_name

        self.order = []
        self.parameters = {}
        if base_class_arguments is not None:
            for arg_name in base_class_arguments.order:
                self.parameters[arg_name] = base_class_arguments.parameters[arg_name]
                self.order.append(arg_name)

        for arg in args:
            # if not redefining previously known parameter, add it to the list
            if arg[0] not in self.parameters:
                self.order.append(arg[0])

            if isinstance(arg[3], Register.ClassRegister):
                arg = (arg[0], arg[1], arg[2], [arg[3]])

            self.parameters[arg[0]] = arg


        self.variables_order = []
        self.variables = {}
        for (var_key, init, vtype) in variables:
            if var_key not in self.variables:
                self.variables_order.append(var_key)
            self.variables[var_key] = (init, vtype)

        if order is not None:
            self.change_order(*order)

    def change_order(self, *args):
        if len(set(args)) != len(args):
            raise ValueError("New order contains some element multiple times.")

        if set(self.order) != set(args):
            raise ValueError("New order does not solely reorders the parameters"
                             ", but adds more and/or skips some.")

        self.order = args

    def parse(self, parameter, tree, item_cache):
        # unknown parameter
        if parameter not in self.parameters:
            raise ArgumentException("Tried to parse unknown parameter "
                                    + str(parameter) + " for object of class "
                                    + self.class_name + ".")

        # Remember if reg_or_conv is a list, then it contains soleley
        # ClassRegisters. If it is no list, then it is not a ClassRegister
        (name, optional, default, reg_or_conv) = self.parameters[parameter]

        # argument not provided by user
        if tree is None:
            if optional:
                return default
            else:
                raise ArgumentException("Obligatory argument " + str(name)
                                        + " missing for object of type "
                                        + str(self.class_name))

        # check if globally defined (if = without register, else with register)
        if tree.empty():
            obj = None
            if isinstance(reg_or_conv, list):
                for class_register in reg_or_conv:
                    new = item_cache.get_from_empty_tree(None, tree,
                                                         class_register, True)
                    if new is not None:
                        if obj is not None:
                            raise ArgumentException("Multiple cached objects "
                                                    "fit for a global look up "
                                                    "for " + tree.data[0])
                        obj = new

            else:
                obj = item_cache.get_from_empty_tree(None, tree,
                                                     None, True)
            if obj is not None:
                return obj


        # parse objects from strings
        if tree.data[0] == "list":
            obj = []
            for child in tree.children:
                obj_child = self.parse(parameter, child, item_cache)
                obj.append(obj_child)
            return obj

        elif tree.data[0] == "map":
            obj = {}
            for child in tree.children:
                obj_child = self.parse(parameter, child, item_cache)
                map_key = child.data[1]
                if map_key in obj:
                    raise ArgumentException("Multiple objects defined for the"
                                            " same map key : " + map_key)
                obj[map_key] = obj_child
            return obj

        else:
            if reg_or_conv is None:
                return tree.data[0]

            # is register
            elif isinstance(reg_or_conv, list):
                type_name = tree.data[0].lower()
                valid_register = None
                for class_register in reg_or_conv:
                    if class_register.has_key(type_name):
                        if valid_register is None:
                            valid_register = class_register
                        else:
                            raise ValueError("To parse '" + str(type_name)
                                             + "' for '" + str(self.class_name)
                                             + "' multiple possibilities are "
                                               "valid.")

                if valid_register is None:
                    raise ArgumentException("Unknown object type or id '"
                                            + str(tree.data[0])
                                            + "' for parameter '"
                                            + str(parameter) + "' of type '"
                                            + str(self.class_name) + "'.")

                return valid_register.get_reference(type_name).parse(tree, item_cache)

            # is converter function
            elif callable(reg_or_conv):
                return reg_or_conv(tree.data[0])


        return obj


    def validate_and_return_variables(self, vars, order=None):
        """
        If the given variable object has not defined the initial value and/or
        the value type, this will be set here, too.
        :param vars:
        :param order:
        :return:
        """

        for var_key in vars:
            if var_key not in self.variables:
                continue

            init, vtype = self.variables[var_key]
            var = vars[var_key]
            if var.value is None:
                var.value = init
            elif var.value != init:
                raise ArgumentException("A variable given to " + self.class_name
                                        + " has an invalid initial value ("
                                        + str(var.value) + " instead of "
                                        + str(init) + ").")

            if var.vtype is None:
                var.ctype = vtype
            elif var.vtype != vtype:
                raise ArgumentException("A variable given to " + self.class_name
                                        + " has an invalid value type("
                                        + str(var.vtype) + " instead of "
                                        + str(vtype) + ").")

        to_return = []
        order = self.variables_order if order is None else order

        for var_key in order:
            if var_key not in vars:
                to_return.append(None)
            else:
                to_return.append(vars[var_key])
                del vars[var_key]

        return to_return

