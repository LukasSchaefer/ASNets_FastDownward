import numpy as np

# TODO:
"""TODO pruning/duplicate detection is currently too strong. It only checks for
the hashes. If collisions happen, it does not check if both objects are the same
or just a collision has occurred. """

class SizeBatchData(object):
    """Every entry (is a list and) hasa fixed number of fields. If it has to
    few, None is added if it has to many, the fields to much are forgotten
    The format is:
    data = {type: [BATCHES*]
    BATCHES = [ENTRIES*]
    ENTRIES = [field1, field2, ...]
    fieldi can contain any data. If it is a list, then for all entries within
    a batch the length of fieldi is the same (This is done, because Tensorflow
    and Theano do not allow to receive batches of data where the data is of
    different dimensions within a batch)
    """
    def __init__(self, nb_fields, field_descriptions=None, meta=None, pruning=None):
        """

        :param nb_fields:
        :param field_descriptions:
        :param meta:
        :param pruning: None means no pruning otherwise callable has to be
                        provided which is used to convert the entries in a
                        hashable format which is then used for duplicate
                        checking. The callable should convert two samples to
                        the same value if and only if they are the same for
                        you and any of them could be pruned as duplicate
        """
        self.nb_fields = nb_fields
        self.field_descriptions = [] if field_descriptions is None else field_descriptions
        for i in range(len(self.field_descriptions), self.nb_fields):
            self.field_descriptions.append(None)

        self.data = {}
        self.batches = {}
        self.meta = {} if meta is None else meta
        self.pruning = pruning
        self.pruning_set = set()

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

    def _over_all(self, func, early_stopping=False):
        for type in self.data:
            for idx_batch in range(len(self.data[type])):
                for idx_entry in range(len(self.data[type][idx_batch])):
                    r = func(self.data[type][idx_batch][idx_entry])
                    if early_stopping and r:
                        return

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

        if self.pruning is not None:
            hash_entry = self.pruning(entry)
            if hash_entry in self.pruning_set:
                return
            else:
                self.pruning_set.add(hash_entry)


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

    def __len__(self):
        return self.size()

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

    def remove_duplicates_from(self, data, hasher=None):
        """
        Removes from THIS SizeBatchData all entries which also occur in data.
        :param data: SizeBatchData object
        :param hasher
        :return:
        """
        self._check_not_finalized()
        if hasher is None:
            if self.pruning == data.pruning:
                hasher = self.pruning
        if hasher is None:
            raise ValueError("No hashing function given to compare the data"
                             " elements and both objects to not agree on a"
                             " hashing function.")
        for batch_key in self.batches:
            if batch_key in data.batches:
                other_hashes = set()
                other_batch = data.batches[batch_key]
                for idx in range(len(other_batch)):
                    other_hashes.add(hasher(other_batch[idx]))

                my_batch = self.batches[batch_key]
                for idx in range(len(my_batch) - 1, -1, -1):
                    if hasher(my_batch[idx]) in other_hashes:
                        del my_batch[idx]



    def remove_duplicates_from_iter(self, datas, hasher=None):
        for data in datas:
            self.remove_duplicates_from(data, hasher=hasher)




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
                 file=None, meta=None, pruning=None):
        SizeBatchData.__init__(self, nb_fields, field_descriptions,
                               meta=meta, pruning=pruning)
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

