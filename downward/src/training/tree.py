from __future__ import print_function

class TreeNode(object):
    def __init__(self, data, parent=None, children=None):
        self.data = data
        #if needed, then parent.add_child will set the correct parent value
        self._parent = None

        if parent is None:
            self.__next_sibling = self
            self.__prev_sibling = self

        else:
            parent.add_child(self)

        if children is None:
            self._children = []
        else:
            self._children = children

            for i in range(len(self._children)):
                self._children[i]._parent = self
                self._children[i].__next_sibling = (
                    self._children[(i + 1) % len(self._children)])
                self._children[i].__prev_sibling = self._children[i - 1]

    def size(self):
        return len(self._children)

    def empty(self):
        return self.size() == 0

    def child_at(self, idx):
        return self._children[idx]

    def index(self, node):
        for idx in range(self.size()):
            if self._children[idx] is node:
                return idx
        return None

    def has_child(self, node):
        return self.index(node) is not None

    def __get_first_child(self):
        if self.empty():
            return None
        else:
            return self._children[0]
    first_child = property(__get_first_child)

    def __get_last_child(self):
        if self.empty():
            return None
        else:
            return self._children[-1]
    last_child = property(__get_last_child)

    def __get_children(self):
        return self._children
    children = property(__get_children)

    def __get_next_sibling(self):
        return self.__next_sibling
    next_sibling = property(__get_next_sibling)

    def __get_prev_sibling(self):
        return self.__prev_sibling
    prev_sibling = property(__get_prev_sibling)

    def add_child(self, node, position=-1):
        if node._parent is not None:
            raise ValueError("Node that shall be added to a parent has "
                             + "already a parent.")

        if position < -len(self._children) -1 or position > len(self._children):
            raise IndexError("Index " + str(position) + " is out of bound for"
                             + " inserting a child to " + str(self.size())
                             + " children.")
        if position < 0:
            position = self.size() + position + 1

        node._parent = self
        self._children.insert(position, node)
        prev = position - 1
        next = 0 if (position + 1 == self.size()) else position + 1

        self._children[prev].__next_sibling = node
        node.__prev_sibling = self._children[prev]

        self._children[next].__prev_sibling = node
        node.__next_sibling = self._children[next]

    def create_child(self, data, children=None):
        return TreeNode(data, self, children)

    def rmv_child(self, node_or_idx):
        node = None
        idx = None
        try:
            node = self.child_at(node_or_idx)
            idx = node_or_idx
        except TypeError:
            idx =  self.index(node_or_idx)
            if idx is None:
                raise ValueError("The provided input to remove a chosen child "
                                 + "form its parent node did not successfully "
                                 + "identify a child node of the parent.")
            node = node_or_idx

        node._parent = None

        node.__next_sibling.__prev_sibling = node.__prev_sibling
        node.__prev_sibling.__next_sibling = node.__next_sibling

        node.__prev_sibling = None
        node.__next_sibling = None

        del self._children[idx]

    def __get_parent(self):
        return self._parent

    def __set_parent(self, new_parent):
        if self._parent is not None:
            self._parent.rmv_child(self)

        if new_parent is not None:
            new_parent.add_child(self)

    parent = property(__get_parent, __set_parent)

    def _toString(self, level=0):
        s = ("\t" * level + str(self.data)
             + "\tparent: "
             + ("None" if self.parent is None else str(self.parent.data))
             + "\tprev: "
             + ("None" if self.prev_sibling is None else str(self.prev_sibling.data))
             + "\tnext: "
             + ("None" if self.next_sibling is None else str(self.next_sibling.data)))

        for child in self.children:
            s += "\n" + child._toString(level + 1)

        return s

    def toString(self):
        return self._toString()

    def disp(self):
        print(self.toString())




"""TESTS"""
"""
import sys


root = TreeNode('root')
root.disp()
print(root.empty())

n11 = root.create_child('11')
root.disp()
n11.disp()

n12 = TreeNode('12')
root.add_child(n12)
root.disp()

n13 = TreeNode('13', root)
root.disp()

n141 = TreeNode('141')
n142 = TreeNode('142')
n143 = TreeNode('143')
c14 = [n141, n142, n143]

root.create_child('14', c14)
n14 = root.child_at(3)
root.disp()

print(root.size())
print(root.empty())
print(root.first_child.data)
print(root.last_child.data)
print(root.index(n12))
print(str(root.index(n142)))
print(root.has_child(n12))
print(root.has_child(n141))


print("RMV n12 from root via object")
root.rmv_child(n12)
root.disp()
n12.disp()

print("RMV n13 from root via index (then 1)")
root.rmv_child(1)
root.disp()
n13.disp()
print("RMV OBJ not there *n12")
try:
    root.rmv_child(n12)
    print("FAIL")
    sys.exit(1)
except ValueError:
    root.disp()

print("Add back at correct positions")

root.add_child(n12, 1)
root.add_child(n13, 2)
root.disp()

print("RMV parent of n14")
n14.parent = None

root.disp()
n14.disp()
"""