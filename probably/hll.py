from __future__ import absolute_import, division, print_function

import numpy as np
from six import PY3
from six.moves import range

from .hashfunctions import hash64


if PY3:
    long = int


class HyperLogLog(object):
    """ Basic Hyperloglog """

    def __init__(self, error_rate):
        b = int(np.ceil(np.log2((1.04 / error_rate) ** 2)))
        self.precision = 64
        self.alpha = self._get_alpha(b)
        self.b = b
        self.m = 1 << b
        self.M = np.zeros(self.m, dtype=np.uint8)
        self.bitcount_arr = [long(1) << i for i in range(self.precision - b + 1)]

    @staticmethod
    def _get_alpha(b):
        if not (4 <= b <= 16):
            raise ValueError("b=%d should be in range [4 : 16]" % b)
        if b == 4:
            return 0.673
        if b == 5:
            return 0.697
        if b == 6:
            return 0.709
        return 0.7213 / (1.0 + 1.079 / (1 << b))

    def _get_rho(self, w, arr):
        """ Return the least signifiant bit
            O(N) in the worst case
        """
        lsb = 0
        while not (w & arr[lsb]):
            lsb += 1
        return lsb + 1

    def add(self, uuid):
        """ Adds a key to the HyperLogLog """
        if uuid:
            # Computing the hash
            try:
                x = hash64(uuid)
            except UnicodeEncodeError:
                x = hash64(uuid.encode('ascii', 'ignore'))
            # Finding the register to update by using the first b bits as an index
            j = x & ((1 << self.b) - 1)
            # Remove those b bits
            w = x >> self.b
            # Find the first 0 in the remaining bit pattern
            self.M[j] = max(self.M[j], self._get_rho(w, self.bitcount_arr))

    def __len__(self, M=None):
        """ Returns the estimate of the cardinality """
        return self.estimate()

    def __or__(self, other_hll):
        """  Perform a union with another HLL object. """
        self.M = reduce(lambda x, y: np.maximum(x, y),
                        [self.M, other_hll.M]).astype(np.int16)
        return self

    def estimate(self):
        """ Returns the estimate of the cardinality """
        E = self.alpha * float(self.m ** 2) / np.power(2.0, - self.M).sum()
        if E <= 2.5 * self.m:             # Small range correction
            V = self.m - np.count_nonzero(self.M)
            return int(self.m * np.log(self.m / float(V))) if V > 0 else int(E)
        # intermidiate range correction -> No correction
        elif E <= float(long(1) << self.precision) / 30.0:
            return int(E)
        else:
            return int(-(long(1) << self.precision) *
                       np.log(1.0 - E / (long(1) << self.precision)))


if __name__ == "__main__":
    hll = HyperLogLog(0.01)
    for i in range(100000):
        hll.add(str(i))
    print(len(hll))
