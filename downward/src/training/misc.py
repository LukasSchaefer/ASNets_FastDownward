def append_register(dictionary, item, *args):
    """
    Register in the dictionary for every key given in args the item.

    :param dictionary: dict in which the relation shall be registered
    :param item: item to register (e.g. constructor)
    :param args: names under which the item can be found
    :return: None
    """

    for key in args:
        if not key in dictionary:
            dictionary[key] = item
        else:
            raise KeyError("Internal Error: In a register are multiple times "
                           + "items for the same key (" + key + ") defined.")
