import time
import numpy as np
import pandas as pd
import itertools

from bloomfilter_cython import BloomFilter
#from pybloomfilter import BloomFilter

def test_error_rate(bf, capacity, nbr_experiments):
    # uuid for the poor (faster them uuid.uuid4())
    for i in range(nbr_experiments):
        random_items = [str(r) + str(time.time()) for r in np.random.randn(2*capacity)]

        bf.initialize_bitarray()
        for item in random_items[:capacity]:
            bf.add(item)

        false_positive = 0
        for item in random_items[capacity:2*capacity]:
            if item in bf:
                false_positive += 1

        false_positive_rate = float(false_positive) / capacity

        false_negative = 0
        for item in random_items[:capacity]:
            if not item in bf:
                false_negative += 1

        false_negative_rate = float(false_negative) / capacity
        yield false_negative_rate, false_positive_rate


def test_error_rate_range(bf, capacity, nbr_experiments):
    """Use xrange to generate the unique.

    This test is not a true randomized test. It should use only for testing
    the BF on really big cardinality without hammering the RAM.
    """
    for i in range(nbr_experiments):
        bf.initialize_bitarray()
        for item in xrange(0,capacity):
            bf.add(str(item))

        false_positive = 0
        for item in xrange(capacity,2*capacity):
            if str(item) in bf:
                false_positive += 1

        false_positive_rate = float(false_positive) / capacity

        false_negative = 0
        for item in xrange(0,capacity):
            if not str(item) in bf:
                false_negative += 1

        false_negative_rate = float(false_negative) / capacity
        yield false_negative_rate, false_positive_rate


if __name__ == "__main__":
    nbr_experiments = 100
    capacities = np.logspace(3,5,4)
    error_rates = [0.001, 0.01, 0.02, 0.05, 0.1]
    all_results = []

    for capacity, error_rate in itertools.product(capacities, error_rates):
        print "Testing capacity=%s and error_rate=%s" % (capacity, error_rate)
        bf = BloomFilter(int(capacity), error_rate)
        error_rate_result = pd.DataFrame(test_error_rate(bf, int(capacity), nbr_experiments), columns=['false_negative','false_positive'])
        all_results.append((capacity,
                            error_rate,
                            error_rate_result.false_positive.mean(),
                            error_rate_result.false_negative.mean()))

    all_results_df = pd.DataFrame(all_results, columns=['capacity', 'error_rate', 'false_positive', 'false_negative'])
