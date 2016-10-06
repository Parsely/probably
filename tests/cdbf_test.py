from __future__ import absolute_import, print_function

import time
import unittest

import numpy as np
from six.moves import range

from probably import CountdownBloomFilter


class CountdownBloomFilterTests(unittest.TestCase):
    '''
    Tests for CountdownBloomFilter
    '''
    @classmethod
    def setUp(self):
        self.batch_refresh_period = 0.1
        self.expiration = 5.0
        self.bf = CountdownBloomFilter(1000, 0.02, self.expiration)

    def test_empty(self):
        assert len(self.bf) == 0
        assert self.bf.cellarray.nonzero()[0].shape == (0,)

    def test_cellarray(self):
        assert self.bf.cellarray.shape == (8148,)

    def test_add(self):
        existing = self.bf.add('random_uuid')
        assert existing == False
        existing = self.bf.add('random_uuid')
        assert existing == True
        assert (self.bf.cellarray.nonzero()[0] == np.array([1039, 1376, 3202, 5228, 6295, 7530])).all()

    def test_touch(self):
        existing = self.bf.add('random_uuid')
        assert existing == False
        existing = self.bf.add('random_uuid')
        assert existing == True
        nzi = self.bf.cellarray.nonzero()[0]
        # Check membership just before expiration
        nbr_step = int(self.expiration / self.batch_refresh_period)
        for i in range(nbr_step - 1):
            self.bf.batched_expiration_maintenance(self.batch_refresh_period)
        existing = 'random_uuid' in self.bf
        assert existing == True

        # Check membership right after expiration
        self.bf.batched_expiration_maintenance(2 * self.batch_refresh_period)

        # Touch. This should reset the TTL
        existing = self.bf.add('random_uuid')
        assert existing == False

        existing = 'random_uuid' in self.bf
        assert existing == True



    def test_compute_refresh_time(self):
        assert self.bf.compute_refresh_time() == 2.4132205876674775e-06

    def test_single_batch_expiration(self):
        existing = self.bf.add('random_uuid')
        assert existing == False
        existing = self.bf.add('random_uuid')
        assert existing == True
        nzi = self.bf.cellarray.nonzero()[0]
        assert (self.bf.cellarray[nzi] == np.array([255, 255, 255, 255, 255, 255], dtype=np.uint8)).all()
        self.bf.batched_expiration_maintenance(self.batch_refresh_period)
        assert (self.bf.cellarray[nzi] == np.array([250, 250, 250, 250, 250, 250], dtype=np.uint8)).all()
        self.bf.batched_expiration_maintenance(self.expiration - 2 * self.batch_refresh_period)
        assert (self.bf.cellarray[nzi] == np.array([5, 5, 6, 6, 6, 6], dtype=np.uint8)).all()
        self.bf.batched_expiration_maintenance(self.batch_refresh_period)
        assert (self.bf.cellarray[nzi] == np.array([0, 0, 1, 1, 1, 1], dtype=np.uint8)).all()

    def test_expiration_realtime(self):
        existing = self.bf.add('random_uuid')
        assert existing == False
        existing = self.bf.add('random_uuid')
        assert existing == True
        elapsed = 0
        start = time.time()
        while existing:
            t1 = time.time()
            if elapsed:
                self.bf.batched_expiration_maintenance(elapsed)
            existing = 'random_uuid' in self.bf
            t2 = time.time()
            elapsed = t2 - t1
        experimental_expiration = time.time() - start
        print(experimental_expiration)
        assert (experimental_expiration - self.expiration) < 0.25 # Arbitrary error threshold

    def test_expiration(self):
        existing = self.bf.add('random_uuid')
        assert existing == False
        existing = self.bf.add('random_uuid')
        assert existing == True
        nzi = self.bf.cellarray.nonzero()[0]
        # Check membership just before expiration
        nbr_step = int(self.expiration / self.batch_refresh_period)
        for i in range(nbr_step - 1):
            self.bf.batched_expiration_maintenance(self.batch_refresh_period)
        existing = 'random_uuid' in self.bf
        assert existing == True
        # Check membership right after expiration
        self.bf.batched_expiration_maintenance(self.batch_refresh_period)
        existing = 'random_uuid' in self.bf
        assert existing == False

    def test_count_estimate(self):
        for i in range(500):
            self.bf.add(str(i))
        assert self.bf.count == 500
        self.bf.batched_expiration_maintenance(2.5)
        for i in range(500, 1000):
            self.bf.add(str(i))
        assert self.bf.count == 1000
        for i in range(26):
            self.bf.batched_expiration_maintenance(0.1)
        assert self.bf.count == 492
        self.assertAlmostEqual(self.bf.estimate_z, 0.304, places=3)
        self.assertAlmostEqual(float(self.bf.cellarray.nonzero()[0].shape[0]) / self.bf.nbr_bits, 0.304, places=3)



if __name__ == '__main__':
     unittest.main()
