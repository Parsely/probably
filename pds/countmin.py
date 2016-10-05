from __future__ import absolute_import, division, print_function

import heapq
import sys
import random

import numpy as np

from .hashfunctions import generate_hashfunctions


class CountMinSketch(object):
    """ Basic Count-Min Sketch """

    def __init__(self, delta, epsilon, k):
        self.nbr_bits = int(np.ceil(np.exp(1) / epsilon))
        self.nbr_slices = int(np.ceil(np.log(1 / delta)))
        self.k = k
        self.count = np.zeros((self.nbr_slices, self.nbr_bits), dtype=np.int32)
        self.heap = []
        self.top_k = {}
        self.make_hashes = generate_hashfunctions(self.nbr_bits, self.nbr_slices)

    def update(self, key, increment):
        for row, column in enumerate(self.make_hashes(key)):
            self.count[int(row), int(column)] += increment
        return self.update_heap(key)

    def update_heap(self, key):
        estimate = self.get(key)
        poped = key
        if key in self.top_k:
            old_pair = self.top_k.get(key)
            old_pair[0] = estimate
            heapq.heapify(self.heap)
            poped = None
        else:
            if len(self.top_k) < self.k:
                heapq.heappush(self.heap, [estimate, key])
                self.top_k[key] = [estimate, key]
                poped = None
            else:
                new_pair = [estimate, key]
                old_pair = heapq.heappushpop(self.heap, new_pair)
                poped = old_pair[1]
                if old_pair[1] in self.top_k:
                    del self.top_k[old_pair[1]]
                    self.top_k[key] = new_pair
        return poped

    def get(self, key):
        value = float('inf')
        for row, column in enumerate(self.make_hashes(key)):
            value = min(self.count[row, column], value)
        return value



if __name__ == "__main__":
    import random
    import time

    stream = []
    for i in range(100):
        stream = stream + [str(i)] * i

    cms = CountMinSketch(10**-3, 0.01, 10)
    random.shuffle(stream)

    t1 = time.time()
    for s in stream:
        p = cms.update(s, 1)
    print(time.time() - t1)

