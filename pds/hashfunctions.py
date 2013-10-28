from sklearn import utils

def generate_hashfunctions(nbr_bits, nbr_slices):
    """ Generate a set of hash functions
        The core method is 32-bits murmur3 hash which have good distribution properties
        Multiple hashes can be generate using the previous hash value as a seed.
    """
    def _make_hashfuncs(key):
        if isinstance(key, unicode):
            key = key.encode('utf-8')
        else:
            key = str(key)
        rval = []
        current_hash = None
        for i in range(nbr_slices):
            seed = current_hash or 0
            current_hash = utils.murmurhash3_32(key, seed, True)
            rval.append(current_hash % nbr_bits)
        return rval
    return _make_hashfuncs

def get_raw_hashfunctions():
    return utils.murmurhash3_32
