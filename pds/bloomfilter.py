import math
import bitarray

from hashfunctions import generate_hashfunctions

class BloomFilter(object):
    """ Basic Bloom Filter """

    def __init__(self, capacity, error_rate):
        self.error_rate = error_rate
        self.capacity = capacity
        self.nbr_slices = int(math.ceil(math.log(1.0 / error_rate, 2)))
        self.bits_per_slice = int(math.ceil((capacity * abs(math.log(error_rate))) / (self.nbr_slices * (math.log(2) ** 2))))
        self.nbr_bits = self.nbr_slices * self.bits_per_slice
        self.bitarray = bitarray.bitarray(self.nbr_bits, endian='little')
        self.bitarray.setall(False)
        self.count = 0
        self.hashes = generate_hashfunctions(self.bits_per_slice, self.nbr_slices)
        self.hashed_values = []

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



