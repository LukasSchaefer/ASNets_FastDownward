import numpy as np

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
        :param field_descriptions: List(!) of field descriptions
        :param meta:
        :param pruning: None means no pruning otherwise callable has to be
                        provided which is used to convert the entries in a
                        hashable format which is then used for duplicate
                        checking. The callable should convert two samples to
                        the same value if and only if they are the same for
                        you and any of them could be pruned as duplicate
        """
        # Attention: if adding new attributes, refactor self.copy(...)
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


    def _copy_fill_fields(self, other, only_structure, entry_copier):
        other.nb_fields = self.nb_fields
        other.field_descriptions = self.field_descriptions
        other.meta = self.meta
        other.prining = self.pruning
        if not only_structure:
            other.pruning_set = set(self.pruning_set)
            other.is_finalized = self.is_finalized
            other.add_data(self, False, entry_copier=entry_copier)

    def copy(self, only_structure=False, entry_copier=None):
        """
        Return a copy of this data object. This is NOT a deep copy. For example
        changing the field_descriptions or the meta in the original
        data set also changes this value in the copied data.
        (Internal variables are deep copied, as we know how to do this)

        :param only_structure: Return instead a new SizeBatchData object, which
                               has the same structure (e.g. number of fields,
                               field descriptions, meta, and pruning) as this
                               object, but does not contain data entries.
        :param entry_copier: If given, the entries from data are not simply
                             added to this object, but, entry_copier(entry) is
                             added to this object. Use Cases: create independent
                             entries which are not effected by changing them in
                             one of the two SizeBatchData objects
        :return:
        """
        other = SizeBatchData(self.nb_fields)
        self._copy_fill_fields(other, only_structure=only_structure,
                               entry_copier=entry_copier)
        return other



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

    def _over_all_types(self, func, early_stopping=False, provide_type=False):
        for type in self.data:
            if provide_type:
                r = func(self.data[type], type)
            else:
                r = func(self.data[type])
            if early_stopping and r:
                return

    def _over_all_batches(self, func, early_stopping=False, provide_type=False):
        for type in self.data:
            for idx_batch in range(len(self.data[type])):
                if provide_type:
                    r = func(self.data[type][idx_batch], type)
                else:
                    r = func(self.data[type][idx_batch])
                if early_stopping and r:
                    return

    def _over_all(self, func, early_stopping=False, provide_type=False):
        for type in self.data:
            for idx_batch in range(len(self.data[type])):
                for idx_entry in range(len(self.data[type][idx_batch])):
                    if provide_type:
                        r = func(self.data[type][idx_batch][idx_entry], type)
                    else:
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
        """

        :param entry:
        :param type:
        :return:
        """
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


    def add_data(self, data, check_fields=True, entry_copier=None):
        """

        :param data: another SizeBatchData object
        :param check_fields:
        :param entry_copier: If given, the entries from data are not simply
                             added to this object, but, entry_copier(entry) is
                             added to this object. Use Cases: create independent
                             entries which are not effected by changing them in
                             one of the two SizeBatchData objects
        :return:
        """
        if check_fields:
            if self.nb_fields != data.nb_fields:
                raise ValueError("Given SizeBatchData object has an incompatible"
                                 " number of fields.")
            for i in range(self.nb_fields):
                if self.field_descriptions[i] != data.field_descriptions[i]:
                    raise ValueError("Descriptions of fields differ in the "
                                     "SizeBatchData objects to combine.")
        if entry_copier is not None:
            def add(entry, type=None):
                self.add(entry_copier(entry), type)
            data._over_all(add, provide_type=True)
        else:
            data._over_all(self.add, provide_type=True)


    def empty(self):
        for type in self.data:
            for idx_batch in range(len(self.data[type])):
                if len(self.data[type][idx_batch]) > 0:
                    return False
        return True

    def size(self):
        def count(batch):
            count.c += len(batch)
        count.c = 0
        self._over_all_batches(count)
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

    def splitoff(self, *fractions):
        """

        :param fractions: (SEQUENCE). Each element in the sequence can be a
                          fraction to split off (e.g. 0.2 to split
                          of 20% or 0.2 and 0.3 two split off 20% and 30% of the
                          data where all sets are disjunctive). The splitted data
                          is returned in new SizeBatchData objects in the same
                          order as in the *fractions sequence.
                          Alternatively elements can also be tuples of the
                          format (fraction, SizeBatchData object). For those
                          entries the split off data is directly added to
                          the object.
        :return: SizeBatchData objects containing the split off data in the same
                 order as the fractions are defined
        """
        self._check_not_finalized()
        total_size = self.size()
        split_sizes = []
        summed_split_size = 0
        split_objects = []
        sum_fractions = 0.0
        for i in range(len(fractions)):
            try:
                split_sizes.append(int(total_size * fractions[i][0]))
                sum_fractions += fractions[i][0]
                split_objects.append(fractions[i][1])
            except (TypeError, AttributeError):
                split_sizes.append(int(total_size * fractions[i]))
                sum_fractions += fractions[i]
                split_objects.append(self.copy(only_structure=True))
        if sum_fractions > 1.0:
            raise ValueError("Cannot split of more than 100% of a data set."
                             "That's not how it works.")
        for s in split_sizes:
            summed_split_size += s

        chosen = np.arange(total_size)
        np.random.shuffle(chosen)
        chosen = chosen[:summed_split_size]
        chosen = np.sort(chosen)[::-1]
        highest = total_size - 1
        lowest = None
        idx_chosen = 0
        obj_chosen = []
        for key in self.batches:
            batch = self.batches[key]
            lowest = highest - len(batch) + 1

            #Process next chosen
            while idx_chosen < len(chosen) and chosen[idx_chosen] >= lowest:
                transformed_idx = chosen[idx_chosen] - lowest
                obj_chosen.append((batch[transformed_idx], key[0]))
                del batch[transformed_idx]

                idx_chosen += 1
            # Next round
            highest = lowest - 1

        # Append entries to data objects
        np.random.shuffle(obj_chosen)
        next_lowest = 0
        for i in range(len(fractions)):
            next_highest = next_lowest + split_sizes[i]
            for idx_chosen in range(next_lowest, next_highest):
                split_objects[i].add(obj_chosen[idx_chosen][0], obj_chosen[idx_chosen][1])

            next_lowest = next_highest

        return split_objects








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
        # Attention: if adding new attributes, refactor self.copy(...)

        SizeBatchData.__init__(self, nb_fields, field_descriptions,
                               meta=meta, pruning=pruning)
        self.field_current_state = field_current_state
        self.field_goal_state = field_goal_state
        self.field_other_state = field_other_state
        self.field_action = field_action
        self.field_heuristic = field_heuristic

        self.set_meta("file", file)

    def _copy_fill_fields(self, other, only_structure, entry_copier):
        other.field_current_state = self.field_current_state
        other.field_goal_state = self.field_goal_state
        other.field_other_state = self.field_other_state
        other.field_action = self.field_action
        other.field_heuristic = self.field_heuristic

    def copy(self, only_structure=False, entry_copier=None):
        """
        Return a copy of this data object. This is NOT a deep copy. For example
        changing the field_descriptions or the meta in the original
        data set also changes this value in the copied data.
        (Internal variables are deep copied, as we know how to do this)

        :param only_structure: Return instead a new SizeBatchData object, which
                               has the same structure (e.g. number of fields,
                               field descriptions, meta, and pruning) as this
                               object, but does not contain data entries.
        :param entry_copier: If given, the entries from data are not simply
                             added to this object, but, entry_copier(entry) is
                             added to this object. Use Cases: create independent
                             entries which are not effected by changing them in
                             one of the two SizeBatchData objects
        :return:
        """
        other = SampleBatchData(self.nb_fields)
        SizeBatchData._copy_fill_fields(self, other,
                                        only_structure=only_structure,
                                        entry_copier=entry_copier)
        SampleBatchData._copy_fill_fields(self, other,
                                          only_structure=only_structure,
                                          entry_copier=entry_copier)
        return other


    def get_file(self):
        if self.has_meta("file"):
            return self.get_meta("file")
        return None
