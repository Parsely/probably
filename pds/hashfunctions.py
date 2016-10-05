import smhasher
from six import text_type
from six.moves import range


def generate_hashfunctions(nbr_bits, nbr_slices):
    """Generate a set of hash functions.

    The core method is a 64-bit murmur3 hash which has a good distribution.
    Multiple hashes can be generate using the previous hash value as a seed.
    """
    def _make_hashfuncs(key):
        if isinstance(key, text_type):
            key = key.encode('utf-8')
        else:
            key = str(key)
        rval = []
        current_hash = None
        for i in range(nbr_slices):
            seed = current_hash or 0
            current_hash = smhasher.murmur3_x64_64(key, seed)
            rval.append(current_hash % nbr_bits)
        return rval
    return _make_hashfuncs
