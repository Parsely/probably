import os
import cPickle
import glob
import datetime as dt
import math
import time
import zlib

import bitarray
import numpy as np

from bloomfilter import BloomFilter
from hashfunctions import generate_hashfunctions
from pycassa import NotFoundException
from pycassa.pool import ConnectionPool
from pycassa.system_manager import SystemManager, SIMPLE_STRATEGY
from pycassa.columnfamily import ColumnFamily

class DailyTemporalBloomFilter(object):
    """Long Range Temporal BloomFilter using a daily resolution.

    For really high value of expiration (like 60 days) with low requirement on precision.
    The actual error of this BF will the be native error of the BF + the error related
    to the coarse aspect of the expiration, since we no longer expires information precisely.
    Also, as opposed to a classic Bloom Filter, this one will aslo have false positive (reporting membership for a non-member)
    AND false negative (reporting non-membership for a member).

    The upper bound of the temporal_error can be theoricaly quite high. However, if the
    items of the set are uniformly distributed over time, the avg error will be something like 1.0 / expiration
    """

    def __init__(self, capacity, error_rate, expiration, name, snapshot_path, cassandra_session):
        self.error_rate = error_rate
        self.capacity = capacity
        self._initialize_parameters()
        self.initialize_bitarray()
        self.count = 0
        self.hashed_values = []
        self.name = name
        self.snapshot_path = snapshot_path
        self.expiration = expiration
        self.initialize_period()
        self.snapshot_to_load = None
        self.ready = False
        self.warm_period = None
        self.next_snapshot_load = time.time()
        self.cassandra_session = cassandra_session
        self.cassandra_columns_family = "temporal_bf"
        self.keyspace = 'parsely'
        self.uncommited_keys = []
        self.commit_batch = 1000
        self.columnfamily = None
        self.ensure_cassandra_cf()

    def _initialize_parameters(self):
        self.nbr_slices = int(np.ceil(np.log2(1.0 / self.error_rate)))
        self.bits_per_slice = int(np.ceil((self.capacity * abs(np.log(self.error_rate))) / (self.nbr_slices * (np.log(2) ** 2))))
        self.nbr_bits = self.nbr_slices * self.bits_per_slice
        self.hashes = generate_hashfunctions(self.bits_per_slice, self.nbr_slices)

    def ensure_cassandra_cf(self):
        s = SystemManager()
        if self.keyspace not in s.list_keyspaces():
            s.create_keyspace(self.keyspace, SIMPLE_STRATEGY, {'replication_factor': '1'})
        if self.cassandra_columns_family not in s.get_keyspace_column_families(self.keyspace):
            s.create_column_family(self.keyspace, self.cassandra_columns_family)
        self.columnfamily = ColumnFamily(self.cassandra_session, self.cassandra_columns_family)

    def archive_bf_key(self, bf_key):
        self.uncommited_keys.append(bf_key)
        if len(self.uncommited_keys) >= self.commit_batch:
            current_period_hour = dt.datetime.now().strftime('%Y-%m-%d:%H')
            self.columnfamily.insert('%s_%s' % (self.name, current_period_hour), {k:'' for k in self.uncommited_keys})
            self.uncommited_keys = []

    def _hour_range(self, start, end, reverse=False, inclusive=True):
        """Generator that gives us all the hours between a start and end datetime
        (inclusive)."""

        def total_seconds(td):
            return (td.microseconds + (td.seconds + td.days * 24.0 * 3600.0) * 10.0**6) / 10.0**6

        hours = int(math.ceil(total_seconds(end - start) / (60.0 * 60.0)))
        if inclusive:
            hours += 1
        for i in xrange(hours):
            if reverse:
                yield end - dt.timedelta(hours=i)
            else:
                yield start + dt.timedelta(hours=i)

    def resize(self, new_capacity):
        self.capacity = new_capacity
        self._initialize_parameters()
        self.rebuild_from_archive()

    def _drop_archive(self):
        last_period = self.current_period - dt.timedelta(days=self.expiration-1)
        hours = self._hour_range(last_period, dt.datetime.now())
        for hour in hours:
            try:
                row = "%s_%s" % (self.name, hour.strftime('%Y-%m-%d:%H'))
                nbr_keys = self.columnfamily.get_count(row)
                keys = self.columnfamily.remove(row)
            except:
                pass

    def rebuild_from_archive(self):
        """Rebuild the BF using the archived items"""
        self.initialize_bitarray()
        last_period = self.current_period - dt.timedelta(days=self.expiration-1)
        hours = self._hour_range(last_period, dt.datetime.now())
        rows = []
        for i,hour in enumerate(hours):
            row = "%s_%s" % (self.name, hour.strftime('%Y-%m-%d:%H'))
            rows.append(row)
        rows_content = self.columnfamily.multiget(rows, column_count=1E6)

        for row_content in rows_content.values():
            for k in row_content.keys():
                self.add(k, rebuild_mode=True)

    def initialize_bitarray(self):
        """Initialize both bitarray.

        This BF contain two bit arrays instead of single one like a plain BF. bitarray
        is the main bit array where all the historical items are stored. It's the one
        used for the membership query. The second one, current_day_bitarray is the one
        used for creating the daily snapshot.
        """
        self.bitarray = bitarray.bitarray(self.nbr_bits)
        self.current_day_bitarray = bitarray.bitarray(self.nbr_bits)
        self.bitarray.setall(False)
        self.current_day_bitarray.setall(False)

    def __contains__(self, key):
        """Check membership."""
        self.hashed_values = self.hashes(key)
        offset = 0
        for value in self.hashed_values:
            if not self.bitarray[offset + value]:
                return False
            offset += self.bits_per_slice
        return True

    def add(self, key, rebuild_mode=False):
        if not rebuild_mode:
            self.archive_bf_key(key)
        if key in self:
            return True
        offset = 0
        if not self.hashed_values:
            self.hashed_values = self.hashes(key)
        for value in self.hashed_values:
            self.bitarray[offset + value] = True
            self.current_day_bitarray[offset + value] = True
            offset += self.bits_per_slice
        self.count += 1
        return False

    def initialize_period(self, period=None):
        """Initialize the period of BF.

        :period: datetime.datetime for setting the period explicity.
        """
        if not period:
            self.current_period = dt.datetime.now()
        else:
            self.current_period = period
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

    def compute_refresh_period(self):
        self.warm_period =  (60 * 60 * 24) // (self.expiration-2)

    def _should_warm(self):
        return time.time() >= self.next_snapshot_load

    def warm(self, jittering_ratio=0.2):
        """Progressively load the previous snapshot during the day.

        Loading all the snapshots at once can takes a substantial amount of time. This method, if called
        periodically during the day will progressively load those snapshots one by one. Because many workers are
        going to use this method at the same time, we add a jittering to the period between load to avoid
        hammering the disk at the same time.
        """
        if self.snapshot_to_load == None:
            last_period = self.current_period - dt.timedelta(days=self.expiration-1)
            self.compute_refresh_period()
            self.snapshot_to_load = []
            base_filename = "%s/%s_%s_*.dat" % (self.snapshot_path, self.name, self.expiration)
            availables_snapshots = glob.glob(base_filename)
            for filename in availables_snapshots:
                snapshot_period = dt.datetime.strptime(filename.split('_')[-1].strip('.dat'), "%Y-%m-%d")
                if snapshot_period >= last_period:
                    self.snapshot_to_load.append(filename)
                    self.ready = False

        if self.snapshot_to_load and self._should_warm():
            filename = self.snapshot_to_load.pop()
            self._union_bf_from_file(filename)
            jittering = self.warm_period * (np.random.random()-0.5) * jittering_ratio
            self.next_snapshot_load = time.time() + self.warm_period + jittering
            if not self.snapshot_to_load:
                self.ready = True


    def _union_bf_from_file(self, filename, current=False):
        snapshot = cPickle.loads(zlib.decompress(open(filename,'r').read()))
        if current:
            self.current_day_bitarray = self.current_day_bitarray | snapshot
        else:
            self.bitarray = self.bitarray | snapshot

    def restore_from_disk(self, clean_old_snapshot=False):
        """Restore the state of the BF using previous snapshots.

        :clean_old_snapshot: Delete the old snapshot on the disk (period < current - expiration)
        """
        base_filename = "%s/%s_%s_*.dat" % (self.snapshot_path, self.name, self.expiration)
        availables_snapshots = glob.glob(base_filename)
        last_period = self.current_period - dt.timedelta(days=self.expiration-1)
        for filename in availables_snapshots:
            snapshot_period = dt.datetime.strptime(filename.split('_')[-1].strip('.dat'), "%Y-%m-%d")
            if snapshot_period <  last_period and not clean_old_snapshot:
                continue
            else:
                self._union_bf_from_file(filename)
                if snapshot_period == self.current_period:
                    self._union_bf_from_file(filename, current=True)

            if snapshot_period < last_period and clean_old_snapshot:
                os.remove(filename)
        self.ready = True

    def save_snaphot(self):
        """Save the current state of the current day bitarray on disk.

        Save the internal representation (bitarray) into a binary file using this format:
            filename : name_expiration_2013-01-01.dat
        """
        filename = "%s/%s_%s_%s.dat" % (self.snapshot_path, self.name, self.expiration, self.date)
        with open(filename, 'w') as f:
            f.write(zlib.compress(cPickle.dumps(self.current_day_bitarray, protocol=cPickle.HIGHEST_PROTOCOL)))

    def union_current_day(self, bf):
        """Union only the current_day of an other BF."""
        self.bitarray = self.bitarray | bf.current_day_bitarray


if __name__ == "__main__":
    import numpy as np
    from pycassa.pool import ConnectionPool

    pool = ConnectionPool('parsely')

    bf = DailyTemporalBloomFilter(100000, 0.01, 60, 'session_site', './', pool)

    random_items = [str(r) for r in np.random.randn(200000)]
    for item in random_items[:100000]:
        bf.add(item)

    false_positive = 0
    for item in random_items[100000:200000]:
        if item in bf:
            false_positive += 1

    print "Error rate (false positive): %s" % str(float(false_positive) / 100000)

