from . import TreeNode


class ParseError(Exception):
    pass


def parse_tree(config):
    """
    Parse an input configuration to a parse tree. This uses the same algorithm
    which FastDownwards search component uses (and therefore, the same syntax).

    :param config: string to parse into a parse tree
    :return: parse tree.
    """
    root = TreeNode("root")
    pseudoroot = TreeNode("pseudoroot", root)

    cur_node = pseudoroot
    buffer = ""
    key = ""

    for idx_c in range(len(config)):
        c = config[idx_c]


        if c in ["(", ")", ","] and buffer != "":
            cur_node.add_child(TreeNode((buffer, key)))
            buffer = ""
            key = ""
        elif c == "(" and buffer == "":
            raise ParseError("Misplaced opening bracket (" + config[:idx_c + 1])


        if c == " ":
            pass
        elif c == "(":
            cur_node = cur_node.last_child
        elif c == ")":
            if cur_node == pseudoroot:
                raise ParseError("missing (" + config[:idx_c])
            cur_node = cur_node.parent
        elif c == "[":
            if buffer != "":
                raise ParseError("Misplaced opening bracket ["
                                 + config[:idx_c + 1])
            cur_node.add_child(TreeNode(("list", key)))
            key = ""
            cur_node = cur_node.last_child
        elif c == "]":
            if buffer != "":
                cur_node.add_child(TreeNode((buffer, key)))
                buffer = ""
                key = ""
            if cur_node.data[0] != "list":
                raise ParseError("Mismatched brackets " + config[: idx_c + 1])
            cur_node = cur_node.parent
        elif c == "{":
            if buffer != "":
                raise ParseError("Misplaced opening bracket {"
                                 + config[:idx_c + 1])
            cur_node.add_child(TreeNode(("map", key)))
            key = ""
            cur_node = cur_node.last_child
        elif c == "}":
            if buffer != "":
                cur_node.add_child(TreeNode((buffer, key)))
                buffer = ""
                key = ""
            if cur_node.data[0] != "map":
                raise ParseError("Mismatched brackets " + config[: idx_c + 1])
            cur_node = cur_node.parent
        elif c == ",":
            pass
        elif c == "=":
            if buffer == "":
                raise ParseError("Expected keyword before = "
                                 + config[: idx_c + 1])
            key = buffer
            buffer = ""
        else:
            buffer += c


    if cur_node.data != pseudoroot.data:
        raise ParseError("Missing )" + str(cur_node.data))
    if buffer != "":
        cur_node.add_child(TreeNode((buffer, key)))

    return cur_node
