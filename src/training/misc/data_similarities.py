"""
Provides some techniques which tell for a single data entry how similar it is
to a data set.
All techniques shall be registered via a name. in the dict SIMILARITIES.
All techniques shall be similarity distances, and thus, it shall hold:
d(x,y) = d(y,x) for two samples x,y
d(x,y) \in [0,1] with 0 if maximal different and 1 if they are the same
"""

#All available Similarities
SIMILARITIES = {}

def get_similarity(name):
    if name not in SIMILARITIES:
        raise ValueError("No similarity measure registered by the given name: "
                         + str(name) + "\nAvailable similarities: "
                         + str(SIMILARITIES.keys()))
    return SIMILARITIES[name]


def get_wrapper_similarity(similarity, fields=None, comparators=None):
    if isinstance(similarity, str):
        similarity = get_similarity(similarity)

    def wrapped(sample1, sample2):
        nonlocal fields, comparators
        return similarity(sample1, sample2, fields, comparators)
    return wrapped


def get_wrapper_similarity_on_set(similarity, fields=None, comparators=None,
                                  merge=max, init_measure_value=0,
                                  early_stopping=lambda x: x == 1):

    def wrapped(sample, datasets):
        nonlocal similarity, fields, comparators, merge, init_measure_value, early_stopping
        return apply_similarity_on_set(similarity, sample, datasets, fields,
                                       comparators, merge, init_measure_value,
                                       early_stopping)
    return wrapped


"""--------------------STUFF FOR HAMMING SIMILARITY--------------------------"""


HAMMING_COMPARATORS = {}


def hamming_measure_cmp_equal(field1, field2):
    return (1, 1) if field1 == field2 else (0, 1)


HAMMING_COMPARATORS["equal"] = hamming_measure_cmp_equal


def hamming_measure_cmp_iterable_equal(field1, field2):
    if len(field1) != len(field2):
        raise ValueError("Both iterables to compare have to be of the "
                         "same size.")
    total = len(field1)
    same = 0
    for idx in range(len(field1)):
        if field1[idx] == field2[idx]:
            same += 1
    return same, total


HAMMING_COMPARATORS["iterable"] = hamming_measure_cmp_iterable_equal


def hamming_measure(sample1, sample2, fields=None, comparators=None):
    """
    Measures the similarities of two samples by a hamming distance inspired
    approach.
    Distance = (#Same Values)/(#Values)
    Where #Same Values and #Values are counted over all desired fields in
    the entry.
    :param sample1: first sample in form of a list of fields
    :param sample2: second sample in form of a list of fields
    :param fields: list of fields to use from the entries (if None, all fields
                   are used)
    :param comparators: if None #Values is number of chosen fields and all
                        chosen fields are compared via ==.
                        Else it is a list of one function per field which
                        performs the comparison and returns the tuple
                        (#same value in field, #values in field). The list can
                        contain instead of a function None to perform again
                        the == comparison or a string naming a comparison method.
                        All predefined comparators for the hamming measure can
                        be found in HAMMING_COMPARATORS mapped by their string
                        name.
    :return: similarity measure
    """
    if len(sample1) != len(sample2):
        raise ValueError("Both samples need to have the same number of fields")
    if fields is None:
        fields = [i for i in range(len(sample2))]
    same, total = 0, 0

    for idx in range(len(fields)):
        cmp = (hamming_measure_cmp_equal
               if (comparators is None or comparators[idx] is None)
               else comparators[idx])
        if isinstance(cmp, str):
            cmp = HAMMING_COMPARATORS[cmp]
        s, t = cmp(sample1[fields[idx]], sample2[fields[idx]])
        same += s
        total += t
    return float(same) / total

SIMILARITIES["hamming"] = hamming_measure



def apply_similarity_on_set(similarity,
                            sample, datasets, fields=None, comparators=None,
                            merge=max, init_measure_value=0,
                            early_stopping=None):
    """
    Estimates the similarity of a given sample to the samples within
    a given data set. Every entry (sample and entries in data set) have to be
    lists of values where the values are compared (and those values can be
    anything). The given sample is compared with every entry in the data set
    and the obtained measures are combined via
    merged_measure = merge(old, new)

    :param similarity: Function for a similarity measure to use or name of a
                       registered similarity
    :param sample: sample to which to calculate the similarity
    :param datasets: data set to which to compare the sample
    :param fields: fields of the sample to consider
    :param comparators: comparators for the similarity measure to use
    :param merge: function to merge the old measure value and the new measure
    :param init_measure_value: initial value for the measure (used as old in the
                               first merge(old, new)
    :param early_stopping: gets the new measure value every round and returns
                           True, if the comparison shall not be performed for
                           the remaining values in the data set.
    :return: measure for sample to data set
    """
    if isinstance(similarity, str):
        similarity = SIMILARITIES[similarity]
    value = init_measure_value
    def estimate(entry):
        nonlocal value
        val = similarity(sample, entry, fields, comparators)
        value = merge(value, val)
        return False if early_stopping is None else early_stopping(value)

    for data in datasets:
        data._over_all(estimate, early_stopping=True)
    return value

def max_similarity_on_set(similarity, sample, datasets, fields, comparators=None):
    return apply_similarity_on_set(similarity, sample, datasets, fields,
                                   comparators, max, 0, lambda x: x==1)