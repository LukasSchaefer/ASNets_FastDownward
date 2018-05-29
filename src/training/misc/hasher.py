"""
List of hashing functions
"""


def numpy_hasher(x):
    return hash(x.tostring())


# STATE STATE STRING STATE INT Data samples with STATE is a numpy array
def sample_entry_numpy_state_hasher(x):
    return hash((
        numpy_hasher(x[0]),
        numpy_hasher(x[1]),
        x[2],
        numpy_hasher(x[3]),
        x[4]))


def list_hasher(x):
    return hash(tuple([hash(i) for i in x]))
