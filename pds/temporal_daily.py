import os
import glob
import datetime as dt
import numpy as np

from bloomfilter import BloomFilter
from hashfunctions import generate_hashfunctions


class DailyTemporalBloomFilter(BloomFilter):
    """Long Range Temporal BloomFilter using a daily resolution.

    For really high value of expiration (like 60 days) with low requirement on precision.
    The actual error of this BF will the be native error of the BF + the error related
    to the coarse aspect of the expiration, since we no longer expires information precisely.
    Also, as opposed to a classic Bloom Filter, this one will aslo have false positive (reporting membership for a non-member)
    AND false negative (reporting non-membership for a member).

    The upper bound of the temporal_error can be theoricaly quite high. However, if the
    items of the set are uniformly distributed over time, the avg error will be something like 1.0 / expiration
    """

    def __init__(self, capacity, error_rate, expiration, name, snapshot_path):
        super(DailyTemporalBloomFilter, self).__init__(capacity, error_rate)
        self.name = name
        self.snapshot_path = snapshot_path
        self.expiration = expiration
        self.date = self.current_period.strftime("%Y-%m-%d")

    def initialize_period(self):
        self.current_period = dt.datetime.now()
        self.current_period = dt.datetime(self.current_period.year, self.current_period.month, self.current_period.day)

    def maintenance(self):
        """Expire the old element of the set.

        Initialize a new bitarray and load the previous snapshot.
        """
        self.initialize_period()
        self.initialize_bitarray()
        self.restore_from_disk()

    def restore_from_disk(self, clean_old_snapshot=False):
        """Restore the state of the BF using previous snapshots."""
        base_filename = "%s/%s_%s_*.npz" % (self.snapshot_path, self.name, self.expiration)
        availables_snapshots = glob.glob(base_filename)
        for filename in availables_snapshots:
            snapshot_period = dt.datetime.strptime(filename.split('_')[-1].strip('.npz'), "%Y-%m-%d")
            last_period = self.current_period - dt.timedelta(days=self.expiration-1)
            if snapshot_period <  last_period and not clean_old_snapshot:
                continue
            else:
                snapshot = np.load(filename)
                # Unioning the BloomFilters by doing a bitwize OR
                self.bitarray = np.bitwise_or(self.bitarray, snapshot['bitarray'])
            if snapshot_period < last_period:
                os.remove(filename)

    def save_snaphot(self):
        """Save the current state of the snapshot on disk.

        Save the internal representation (numpy array) into a npz file using this format:
            filename : name_expiration_2013-01-01.npz
        """
        filename = "%s/%s_%s_%s" % (self.snapshot_path, self.name, self.expiration, self.date)
        np.savez(filename, bitarray=self.bitarray)


if __name__ == "__main__":
    import numpy as np

    bf = DailyTemporalBloomFilter(10000, 0.01, 30, 'test', './')

    random_items = [str(r) for r in np.random.randn(20000)]
    for item in random_items[:10000]:
        bf.add(item)

    false_positive = 0
    for item in random_items[10000:20000]:
        if item in bf:
            false_positive += 1

    print "Error rate (false positive): %s" % str(float(false_positive) / 10000)
