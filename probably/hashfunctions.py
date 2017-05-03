import struct

import mmh3
from six import text_type
from six.moves import range


def hash64(key, seed):
    """
    Wrapper around mmh3.hash64 to get us single 64-bit value.

    This also does the extra work of ensuring that we always treat the
    returned values as big-endian unsigned long, like smhasher used to
    do.
    """
    hash_val = mmh3.hash64(key, seed)[0]
    return struct.unpack('>Q', struct.pack('q', hash_val))[0]


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
        current_hash = 0
        for i in range(nbr_slices):
            seed = current_hash
            current_hash = hash64(key, seed)
            rval.append(current_hash % nbr_bits)
        return rval
    return _make_hashfuncs
