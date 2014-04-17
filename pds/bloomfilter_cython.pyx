cimport cython

from libc.stdlib cimport malloc, free
from libc.math cimport log, ceil
from libc.stdio cimport printf
from cpython.mem cimport PyMem_Malloc, PyMem_Free

DEF LN2_SQUARED = 0.480453013918201  # ln(2)^2
DEF LN2 = 0.693147180559945  # ln(2)

cdef extern from "stdlib.h" nogil:
    long long int llabs(long long int j)

cdef extern from "MurmurHash2A.h" nogil:
    unsigned int MurmurHash2A (void * key, int len, unsigned int seed)

cdef extern from "MurmurHash3.h":
    void MurmurHash3_x86_32(void *key, int len, unsigned long seed, void *out)
    void MurmurHash3_x86_128(void *key, int len, unsigned long seed, void *out)
    void MurmurHash3_x64_128 (void *key, int len, unsigned long seed, void *out)

cdef class BloomFilter:

    cdef unsigned int nbr_slices
    cdef unsigned long long _bucket_indexes[1000]
    cdef unsigned long long nbr_bits
    cdef unsigned char *bitarray
    cdef unsigned long long nbr_bytes
    cdef unsigned long long capacity
    cdef unsigned long long bits_per_slice
    cdef double error

    def __cinit__(self, capacity, error):
        self.capacity = capacity
        self.error = error

        cdef double numerator = log(self.error)
        self.bits_per_slice = <long long>-(numerator / LN2_SQUARED)

        cdef double dcapacity = <double>capacity
        self.nbr_bits = <long long>(dcapacity * self.bits_per_slice)
        self.nbr_slices = <long long>ceil(LN2 * self.bits_per_slice)

        for i in range(self.nbr_slices):
            self._bucket_indexes[i]=0

        # Initializing the bitarray
        if self.nbr_bits % 8:
            self.nbr_bytes = (self.nbr_bits / 8) + 1
        else:
            self.nbr_bytes = self.nbr_bits / 8
        self.bitarray = <unsigned char*> PyMem_Malloc(self.nbr_bytes * sizeof(unsigned char))

    def __str__(self):
        return """ nbr. bits: %s
                   nbr. bytes: %s
                   nbr. hashes: %s """ % (self.nbr_bits, self.nbr_bytes, self.nbr_slices)


    @cython.boundscheck(False)
    cdef int __check_or_add(self, const char *value, int should_add=1):
        cdef int hits = 0
        cdef unsigned int val_len = len(value)
        cdef unsigned int a = MurmurHash2A(value, val_len, 0x9747b28c)
        cdef unsigned int b = MurmurHash2A(value, val_len, a)
        #cdef unsigned long a[2]
        #cdef unsigned long b[2]

        #MurmurHash3_x86_32(&value, val_len, 0x9747b28c, a)
        #MurmurHash3_x86_32(&value, val_len, a[0], b)

        printf("%d\n", a)
        printf("%d\n", b)

        cdef unsigned int x
        cdef unsigned int i
        cdef unsigned int byte
        cdef unsigned int mask
        cdef unsigned char c

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
            return 1  # 1 == element already in (or collision)

        return 0

    def add(self, const char *value):
        return self.__check_or_add(value, True) == 1

    def __dealloc__(self):
            PyMem_Free(self.bitarray)
