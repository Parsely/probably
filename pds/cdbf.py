from __future__ import absolute_import, division, print_function

import numpy as np

from .hashfunctions import generate_hashfunctions
from .maintenance import maintenance


class CountdownBloomFilter(object):
    """ Implementation of a Modified Countdown Bloom Filter. Uses a batched maintenance process instead of a continuous one.

        Sanjuas-Cuxart, Josep, et al. "A lightweight algorithm for traffic filtering over sliding windows."
        Communications (ICC), 2012 IEEE International Conference on. IEEE, 2012.

        http://www-mobile.ecs.soton.ac.uk/home/conference/ICC2012/symposia/papers/a_lightweight_algorithm_for_traffic_filtering_over_sliding__.pdf
    """

    def __init__(self, capacity, error_rate=0.001, expiration=60, disable_hard_capacity=False):
        self.error_rate = error_rate
        self.capacity = capacity
        self.expiration = expiration
        self.nbr_slices = int(np.ceil(np.log2(1.0 / error_rate)))
        self.bits_per_slice = int(np.ceil((capacity * abs(np.log(error_rate))) / (self.nbr_slices * (np.log(2) ** 2))))
        self.nbr_bits = self.nbr_slices * self.bits_per_slice
        self.count = 0
        self.cellarray = np.zeros(self.nbr_bits, dtype=np.uint8)
        self.counter_init = 255
        self.refresh_head = 0
        self.make_hashes = generate_hashfunctions(self.bits_per_slice, self.nbr_slices)
        # This is the unset ratio ... and we keep it constant at 0.5
        # since the BF will operate most of the time at his optimal
        # set ratio (50 %) and the overall effect of this parameter
        # on the refresh rate is very minimal anyway.
        self.z = 0.5
        self.estimate_z = 0
        self.disable_hard_capacity = disable_hard_capacity

    def _compute_z(self):
        """ Compute the unset ratio (exact) """
        return self.cellarray.nonzero()[0].shape[0] / self.nbr_bits

    def _estimate_count(self):
        """ Update the count number using the estimation of the unset ratio """
        if self.estimate_z == 0:
            self.estimate_z = (1.0 / self.nbr_bits)
        self.estimate_z = min(self.estimate_z, 0.999999)
        self.count = int(-(self.nbr_bits / self.nbr_slices) * np.log(1 - self.estimate_z))

    def expiration_maintenance(self):
        """ Decrement cell value if not zero
            This maintenance process need to executed each self.compute_refresh_time()
        """
        if self.cellarray[self.refresh_head] != 0:
            self.cellarray[self.refresh_head] -= 1
        self.refresh_head = (self.refresh_head + 1) % self.nbr_bits

    def batched_expiration_maintenance_dev(self, elapsed_time):
        """ Batched version of expiration_maintenance() """
        num_iterations = self.num_batched_maintenance(elapsed_time)
        for i in range(num_iterations):
            self.expiration_maintenance()

    def batched_expiration_maintenance(self, elapsed_time):
        """ Batched version of expiration_maintenance()
            Cython version
        """
        num_iterations = self.num_batched_maintenance(elapsed_time)
        self.refresh_head, nonzero = maintenance(self.cellarray, self.nbr_bits, num_iterations, self.refresh_head)
        if num_iterations != 0:
            self.estimate_z = float(nonzero) / float(num_iterations)
            self._estimate_count()
        processed_interval = num_iterations * self.compute_refresh_time()
        return processed_interval

    def compute_refresh_time(self):
        """ Compute the refresh period for the given expiration delay """
        if self.z == 0:
            self.z = 1E-10
        s = float(self.expiration) * (1.0/(self.nbr_bits)) * (1.0/(self.counter_init - 1 + (1.0/(self.z * (self.nbr_slices + 1)))))
        return s

    def num_batched_maintenance(self, elapsed_time):
        return int(np.floor(elapsed_time / self.compute_refresh_time()))

    def __nonzero__(self):
        return True

    def __bool__(self):
        return True

    def __contains__(self, key):
        if not isinstance(key, list):
            hashes = self.make_hashes(key)
        else:
            hashes = key
        offset = 0
        for k in hashes:
            if self.cellarray[offset + k] == 0:
                return False
            offset += self.bits_per_slice
        return True

    def __len__(self):
        """ Return the number of keys stored by this bloom filter. """
        return self.count

    def add(self, key, skip_check=False):
        hashes = self.make_hashes(key)
        if not skip_check and hashes in self:
            offset = 0
            for k in hashes:
                self.cellarray[offset + k] = self.counter_init
                offset += self.bits_per_slice
            return True
        if (self.count > self.capacity or self.estimate_z > 0.5) and not self.disable_hard_capacity:
            raise IndexError("BloomFilter is at capacity")
        offset = 0
        for k in hashes:
            self.cellarray[offset + k] = self.counter_init
            offset += self.bits_per_slice
        self.count += 1
        return False
