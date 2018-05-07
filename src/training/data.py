import numpy as np

class SizeBatchData(object):
    """Every entry (is a list and) hasa fixed number of fields. If it has to
    few, None is added if it has to many, the fields to much are forgotten
    The format is:
    data = {type: [BATCHES]
    BATCHES = [ENTRIES]
    ENTRIES = [field1, field2, ...]
    fieldi can contain any data. If it is a list, then for all entries within
    a batch the length of fieldi is the same (This is done, because Tensorflow
    and Theano do not allow to receive batches of data where the data is of
    different dimensions within a batch)
    """
    def __init__(self, nb_fields, field_descriptions=None, meta=None):

        self.nb_fields = nb_fields
        self.field_descriptions = [] if field_descriptions is None else field_descriptions
        self.data = {}
        self.batches = {}
        self.meta = {} if meta is None else meta

        self.is_finalized = False

    def _check_not_finalized(self):
        if self.is_finalized:
            raise TypeError("SizeBatchData does not support modifications after"
                            "it was finalized.")

    def _modify_all(self, func):
        for type in self.data:
            for idx_batch in range(len(self.data[type])):
                for idx_entry in range(len(self.data[type][idx_batch])):
                    self.data[type][idx_batch][idx_entry] = func(
                        self.data[type][idx_batch][idx_entry])

    def _over_all(self, func):
        for type in self.data:
            for idx_batch in range(len(self.data[type])):
                for idx_entry in range(len(self.data[type][idx_batch])):
                    func(self.data[type][idx_batch][idx_entry])

    def convert_field(self, field, converter):
        def func(entry):
            entry[field] = converter(entry[field])
        self._modify_all(func)

    def get_desc(self, field):
        if field >= self.nb_fields:
            raise ValueError("Field description access out of bounds.")

        if field >= len(self.field_descriptions):
            return None
        else:
            return self.field_descriptions[field]

    def add(self, entry, type=None):
        self._check_not_finalized()

        # Normalize entry to field number
        for i in range(len(entry), self.nb_fields):
            entry.append(None)
        if len(entry) > self.nb_fields:
            entry = entry[:self.nb_fields]

        # Get sizes for correct batch
        key = [type]
        for i in range(self.nb_fields):
            if isinstance(entry[i], list):
                key.append(len(entry[i]))
            else:
                key.append(-1)
        t = tuple(key)
        if t not in self.batches:
            if type not in self.data:
                self.data[type] = []
            self.data[type].append([])
            self.batches[t] = self.data[type][-1]
        self.batches[t].append(entry)


    def empty(self):
        for type in self.data:
            for idx_batch in range(len(self.data[type])):
                for idx_entry in range(len(self.data[type][idx_batch])):
                    return False
        return True

    def size(self):
        def count(entry):
            count.c += 1
        count.c = 0
        self._over_all(count)
        return count.c

    def set_meta(self, name, value):
        self.meta[name] = value

    def has_meta(self, name):
        return name in self.meta

    def get_meta(self, name):
        return self.meta[name]


    def finalize(self):
        self._check_not_finalized()
        for type in self.data:
            for idx_batch in range(len(self.data[type])):
                self.data[type][idx_batch] = np.array(self.data[type][idx_batch], dtype=object)

        self.is_finalized = True


class SampleBatchData(SizeBatchData):
    """
    Use field_XYZ to tell the network in which field to find which information
    Annotate the fields for the current, goal, other state with the format in
    which they are given. Action shall be a string of the grounded action name,
    heuristic shall be an integer
    """
    def __init__(self, nb_fields, field_descriptions=None,
                 field_current_state=None, field_goal_state=None,
                 field_other_state=None,
                 field_action=None, field_heuristic=None,
                 file=None):
        SizeBatchData.__init__(self, nb_fields, field_descriptions)
        self.field_current_state = field_current_state
        self.field_goal_state = field_goal_state
        self.field_other_state = field_other_state
        self.field_action = field_action
        self.field_heuristic = field_heuristic

        self.set_meta("file", file)

    def get_file(self):
        if self.has_meta("file"):
            return self.get_meta("file")
        return None



"""
if __name__ == '__main__':
    print("HELL")
    d = SizeBatchData(3)
    d.add([[1,2,3], 4,5])
    d.add([[1, 2, 4], 4, 5])
    d.add([[1, 2, 5], 4, 5])
    d.add([[1, 2, 3, 4], 4, 5])
    d.add([[1, 2, 3, 5], 4, 5])
    d.add([[1, 2, 3], 4, [5, 6]])
    d.add([[1, 2, 3], 4, [5, 7]])
    d.add([1], "short")
    d.add([9,8,7,6,5], "long")
    print(d.data)
    d.finalize()
    print(d.data)
"""

"""
class Data:
    def __init__(self, fields=None, data=None):
        """'''

        :param fields: name of the fields in the data entries. Nobody is checking
                       the field names on inputting or outputting! They are more
                       for the users pleasure.
                       The entries may have more fields than specified here.
                       On numpyfy multiple things can happen.
                         - it can be tried to interfere the new field name (only
                           if for all entries the same name could be interfered)
                         - ignore the unnamed fields
        :param data: the initial data to be contained by this object (if the
                     data is in an different state than the provided format,
                     fields, or numpyfied variables tell, the behaviour of this
                     object will be unexpected.
                     Format:
                     {entry_type: [entry,...]}
                     entry = [field1, field2, ..., fieldN]'''
"""
self.field_order = [] if fields is None else fields
self.field_map = {}
for i in range(len(self.field_order)):
    self.field_map[i] = self.field_order[i]
self.data = {} if data is None else data

def add_entry(self, entry_type, entry):
if entry_type not in self.data:
    self.data[entry_type] = []
self.data[entry_type].append(entry)

def _for_entry(self, func):
for type in self.data:
    for entry in self.data[type]:
        func(entry)

def _get_field_idx(self, field_name, field_idx):
if field_name is not None:
    field_idx = self.field_map[field_name]
if field_idx is None:
    raise ValueError("Either field_name has to be given or field_idx")
return field_idx

def convert_field(self, field_name=None, field_idx=None, converter=None):
"""'''

        :param field_name: name of the field to convert (Provide EITHER this OR
                           field_idx)
        :param field_idx: idx of the field to convert (Provide EITHER this OR
                          field_name)
        :param converter: callable. Gets the field value for every entry and
                          returns a new value
        :return:'''
"""
field_idx = self._get_field_idx(field_name, field_idx)

def func(entry):
    if len(entry) > field_idx:
        entry[field_idx] = converter(entry[field_idx])
self._for_entry(func)

def rename_field(self, new_name, field_name=None, field_idx=None):
"""'''

        :param field_name: name of the field to rename (Provide EITHER this OR
                           field_idx)
        :param field_idx: idx of the field to rename (Provide EITHER this OR
                          field_name)
        :return:'''
"""
field_idx = self._get_field_idx(field_name, field_idx)
field_name = self.field_order[field_idx]
self.field_order[field_idx] = new_name
self.field_map[new_name] = field_idx
del self.field_map[field_name]


def remove_field(self, field_name=None, field_idx=None):
"""'''

        :param field_name: name of the field to remove (Provide EITHER this OR
                           field_idx)
        :param field_idx: idx of the field to remove (Provide EITHER this OR
                          field_name)
        :return:'''
"""
field_idx = self._get_field_idx(field_name, field_idx)
def func(entry):
    if len(entry) > field_idx:
        del entry[field_idx]
self._for_entry(func)


def reorder_field(self, field_name1=None, field_idx1=None,
              field_name2=None, field_idx2=None):

field_idx1 = self._get_field_idx(field_name1, field_idx1)
field_idx2 = self._get_field_idx(field_name2, field_idx2)
def func(entry):
    if len(entry) > field_idx:
        del entry[field_idx]
self._for_entry(func)
"""




