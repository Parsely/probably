import os
import cPickle
import glob
import datetime as dt
import numpy as np
import zlib

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
        self.initialize_period()

    def initialize_period(self):
        self.current_period = dt.datetime.now()
        self.current_period = dt.datetime(self.current_period.year, self.current_period.month, self.current_period.day)
        self.date = self.current_period.strftime("%Y-%m-%d")

    def maintenance(self):
        """Expire the old element of the set.

        Initialize a new bitarray and load the previous snapshot. Execute this guy
        at the beginining of each day.
        """
        self.initialize_period()
        self.initialize_bitarray()
        self.restore_from_disk()

    def restore_from_disk(self, clean_old_snapshot=False):
        """Restore the state of the BF using previous snapshots."""
        base_filename = "%s/%s_%s_*.dat" % (self.snapshot_path, self.name, self.expiration)
        availables_snapshots = glob.glob(base_filename)
        for filename in availables_snapshots:
            snapshot_period = dt.datetime.strptime(filename.split('_')[-1].strip('.dat'), "%Y-%m-%d")
            last_period = self.current_period - dt.timedelta(days=self.expiration-1)
            if snapshot_period <  last_period and not clean_old_snapshot:
                continue
            else:
                snapshot = cPickle.loads(zlib.decompress(open(filename,'r').read()))
                # Unioning the BloomFilters by doing a bitwize OR
                self.bitarray = self.bitarray | snapshot

            if snapshot_period < last_period and clean_old_snapshot:
                os.remove(filename)

    def save_snaphot(self):
        """Save the current state of the snapshot on disk.

        Save the internal representation (bitarray) into a binary file using this format:
            filename : name_expiration_2013-01-01.dat
        """
        filename = "%s/%s_%s_%s.dat" % (self.snapshot_path, self.name, self.expiration, self.date)
        with open(filename, 'w') as f:
            f.write(zlib.compress(cPickle.dumps(self.bitarray, protocol=cPickle.HIGHEST_PROTOCOL)))


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
