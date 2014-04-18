cimport cython

import math

from libc.stdlib cimport malloc, free
from libc.math cimport log, ceil
from libc.stdio cimport printf
from cpython.mem cimport PyMem_Malloc, PyMem_Free

DEF LN2_SQUARED = 0.480453013918201  # ln(2)^2
DEF LN2 = 0.693147180559945  # ln(2)

cdef extern from "stdlib.h" nogil:
    long long int llabs(long long int j)

cdef extern from "MurmurHash3.h":
    void MurmurHash3_x86_32(void *key, int len, unsigned long seed, void *out)
    void MurmurHash3_x86_128(void *key, int len, unsigned long seed, void *out)
    void MurmurHash3_x64_128 (void *key, int len, unsigned long seed, void *out)

cdef extern from "MurmurHash2A.h" nogil:
    unsigned int MurmurHash2A (void * key, int len, unsigned int seed)


cdef class BloomFilter:

    cdef unsigned int nbr_slices
    cdef unsigned long long _bucket_indexes[1000]
    cdef unsigned long long nbr_bits
    cdef unsigned char *bitarray
    cdef unsigned long long nbr_bytes
    cdef unsigned long long capacity
    cdef unsigned long long bits_per_slice
    cdef double error_rate

    def __cinit__(self, capacity, error_rate):
        self.capacity = capacity
        self.error_rate = error_rate
        self._initialize_parameters()

    def _initialize_parameters(self):
        self.nbr_slices = int(math.floor(math.log(1.0 / self.error_rate, 2.0)))
        self.bits_per_slice = int(math.ceil(
                self.capacity * abs(math.log(self.error_rate)) /
                (self.nbr_slices * (math.log(2) ** 2))))

        self.nbr_bits = self.nbr_slices * self.bits_per_slice

        for i in range(self.nbr_slices):
            self._bucket_indexes[i]=0

        # Initializing the bitarray
        if self.nbr_bits % 8:
            self.nbr_bytes = (self.nbr_bits / 8) + 1
        else:
            self.nbr_bytes = self.nbr_bits / 8
        self.bitarray = <unsigned char*> PyMem_Malloc(self.nbr_bytes * sizeof(unsigned char))
        self._initialize_bitarray()

    def _set_capacity(self, capacity):
        self.capacity = capacity

    def initialize_bitarray(self):
        self._initialize_bitarray()

    cdef void _initialize_bitarray(self):
        for i in range(self.nbr_bits):
            self._set_bit(i,0)

    def __repr__(self):
        return """ Capacity: %s
                   Error rate: %s
                   nbr. bits: %s
                   nbr. bytes: %s
                   nbr. hashes: %s """ % (self.capacity, self.error_rate, self.nbr_bits, self.nbr_bytes, self.nbr_slices)

    cdef int _get_bit(self, unsigned long index):
        bytepos, bitpos = divmod(index, 8)
        return (self.bitarray[bytepos] >> bitpos) & 1

    def get_bit(self, unsigned long index):
        return self._get_bit(index)

    cdef void _set_bit(self, int index, int value):
        cdef int bytepos
        cdef int bitpos
        bytepos, bitpos = divmod(index, 8)
        if value:
            self.bitarray[bytepos] |= 1 << bitpos
        else:
            self.bitarray[bytepos] &= ~(1 << bitpos)

    def set_bit(self, unsigned long index, int value):
        self._set_bit(index, value)

    @cython.boundscheck(False)
    cdef int __check_or_add(self, const char *value, int should_add=1):
        cdef int hits = 0
        cdef int val_len = len(value)
        #cdef unsigned long a
        #cdef unsigned long b
        cdef unsigned int a = MurmurHash2A(value, val_len, 0x9747b28c)
        cdef unsigned int b = MurmurHash2A(value, val_len, a)
        cdef unsigned int x
        cdef unsigned int i
        cdef unsigned int byte
        cdef unsigned int mask
        cdef unsigned char c

        #MurmurHash3_x86_32(<char*> value, val_len, 0x9747b28c, &a)
        #MurmurHash3_x86_32(<char*> value, val_len, a, &b)

        for i in range(self.nbr_slices):
            x = (a + i * b) % self.nbr_bits
            byte = x >> 3
            c = self.bitarray[byte]
            mask = 1 << (x % 8)

            if c & mask:
                hits += 1
            else:
                if should_add == 1:
                    self.bitarray[byte] = c | mask

        if hits == self.nbr_slices:
            return 1

        return 0

    def add(self, const char *value):
        return self.__check_or_add(value, True) == 1

    def contains(self, const char *value):
        return self.__check_or_add(value, False) == 1

    def __contains__(self, value):
        return self.contains(value)

    def __dealloc__(self):
            PyMem_Free(self.bitarray)
