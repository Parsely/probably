from __future__ import absolute_import, division, print_function

import bitarray
import numpy as np

from .hashfunctions import generate_hashfunctions

class BloomFilter(object):
    """Basic Bloom Filter."""

    def __init__(self, capacity, error_rate):
        self.error_rate = error_rate
        self.capacity = capacity
        self.nbr_slices = int(np.ceil(np.log2(1.0 / error_rate)))
        self.bits_per_slice = int(np.ceil((capacity * abs(np.log(error_rate))) / (self.nbr_slices * (np.log(2) ** 2))))
        self.nbr_bits = self.nbr_slices * self.bits_per_slice
        self.initialize_bitarray()
        self.count = 0
        self.hashes = generate_hashfunctions(self.bits_per_slice, self.nbr_slices)
        self.hashed_values = []

    def initialize_bitarray(self):
        self.bitarray = bitarray.bitarray(self.nbr_bits)
        self.bitarray.setall(False)

    def __contains__(self, key):
        self.hashed_values = self.hashes(key)
        offset = 0
        for value in self.hashed_values:
            if not self.bitarray[offset + value]:
                return False
            offset += self.bits_per_slice
        return True

    def add(self, key):
        if key in self:
            return True
        offset = 0
        if not self.hashed_values:
            self.hashed_values = self.hashes(key)
        for value in self.hashed_values:
            self.bitarray[offset + value] = True
            offset += self.bits_per_slice
        self.count += 1
        return False


if __name__ == "__main__":
    import numpy as np

    bf = BloomFilter(10000, 0.01)

    random_items = [str(r) for r in np.random.randn(20000)]
    for item in random_items[:10000]:
        bf.add(item)

    false_positive = 0
    for item in random_items[10000:20000]:
        if item in bf:
            false_positive += 1

    print("Error rate (false positive): %s" % str(float(false_positive) / 10000))
