import os
import cPickle
import glob
import datetime as dt
import math
import time
import zlib

import numpy as np

from collections import defaultdict
from bloomfilter_cython import BloomFilter, DailyTemporalBase
from pycassa import NotFoundException
from pycassa.pool import ConnectionPool
from pycassa.system_manager import SystemManager, SIMPLE_STRATEGY
from pycassa.columnfamily import ColumnFamily


class PDSError(Exception): pass


class DailyTemporalBloomFilter(DailyTemporalBase):
    """Long Range Temporal BloomFilter using a daily resolution.

    For really high value of expiration (like 60 days) with low requirement on precision.
    The actual error of this BF will the be native error of the BF + the error related
    to the coarse aspect of the expiration, since we no longer expires information precisely.
    Also, as opposed to a classic Bloom Filter, this one will aslo have false positive (reporting membership for a non-member)
    AND false negative (reporting non-membership for a member).

    The upper bound of the temporal_error can be theoricaly quite high. However, if the
    items of the set are uniformly distributed over time, the avg error will be something like 1.0 / expiration
    """

    def __new__(cls, capacity, error_rate, expiration, name, cassandra_session, keyspace, snapshot_path='./'):
        return super(DailyTemporalBloomFilter, cls).__new__(cls, capacity=capacity, error_rate=error_rate)

    def __init__(self, capacity, error_rate, expiration, name, cassandra_session, keyspace, snapshot_path='./'):
        filename = ""
        super(DailyTemporalBloomFilter, self).__init__(capacity=capacity, error_rate=error_rate)
        self.bf_name = name
        self.expiration = expiration
        self.initialize_period()
        self.cassandra_session = cassandra_session
        self.cassandra_columns_family = "temporal_bf"
        self.keyspace = keyspace
        self.uncommited_keys = []
        self.uncommited_keys_per_bucket = defaultdict(list)
        self.uncommited_count = 0
        self.commit_batch_size = 1000
        self.commit_period = 5.0
        self.next_cassandra_commit = 0
        self.columnfamily = None
        self.ensure_cassandra_cf()
        self.snapshot_path = snapshot_path
        self.snapshot_to_load = None
        self.warm_period = None
        self.next_snapshot_load = time.time()
        self.ready = False
        self.rebuild_hash = None
        self.hold_warming = False
        self.initialized_at = time.time()

    @property
    def capacity(self):
        return self._get_capacity()

    @property
    def error_rate(self):
        return self._get_error_rate()

    def ensure_cassandra_cf(self):
        s = SystemManager(self.cassandra_session.server_list[0])
        if self.keyspace not in s.list_keyspaces():
            s.create_keyspace(self.keyspace, SIMPLE_STRATEGY, {'replication_factor': '1'})
        if self.cassandra_columns_family not in s.get_keyspace_column_families(self.keyspace):
            s.create_column_family(self.keyspace, self.cassandra_columns_family)
        self.columnfamily = ColumnFamily(self.cassandra_session, self.cassandra_columns_family)

    def archive_bf_key_strict(self, bf_key, period=None):
        """Store the key in Cassandra.

        If an explicit period is provided, the key will be store in this period bucket.
        Strict version, 5-6X slower.
        """
        if not period:
            period = dt.datetime.now()
        self.uncommited_keys_per_bucket[period.strftime('%Y-%m-%d:%H')].append(bf_key)
        self.uncommited_count += 1
        if (self.uncommited_count >= self.commit_batch_size) or (time.time() > self.next_cassandra_commit):
            for period, keys in self.uncommited_keys_per_bucket.iteritems():
                self.columnfamily.insert('%s_%s' % (self.bf_name, period), {k:'' for k in keys})
                self.next_cassandra_commit = time.time() + self.commit_period
            self.uncommited_keys_per_bucket = defaultdict(list)
            self.uncommited_count = 0

    def archive_bf_key(self, bf_key, period=None):
        """Store the key in Cassandra.

        If an explicit period is provided, the key will be store in this period bucket.
        For effiency's sake, we always store the uncommited keys into a single bucket determined
        by the period of last key. To avoid this optimization, use archive_bf_key_strict.
        """
        self.uncommited_keys.append(bf_key)
        if (len(self.uncommited_keys) >= self.commit_batch_size) or (time.time() > self.next_cassandra_commit):
            if not period:
                period = dt.datetime.now()
            self.columnfamily.insert('%s_%s' % (self.bf_name, period.strftime('%Y-%m-%d:%H')), {k:'' for k in self.uncommited_keys})
            self.uncommited_keys = []
            self.next_cassandra_commit = time.time() + self.commit_period

    def get_age(self):
        return time.time() - self.initialized_at

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

    def _day_range(self, start, end, reverse=False, inclusive=True):
        """Generator that gives us all the days between a start and end datetime
        (inclusive)."""
        days = (end - start).days
        if inclusive:
            days += 1
        for i in xrange(days):
            if reverse:
                yield end - dt.timedelta(days=i)
            else:
                yield start + dt.timedelta(days=i)

    def _drop_archive(self):
        last_period = self.current_period - dt.timedelta(days=self.expiration-1)
        hours = self._hour_range(last_period, dt.datetime.now())
        for hour in hours:
            try:
                row = "%s_%s" % (self.bf_name, hour.strftime('%Y-%m-%d:%H'))
                nbr_keys = self.columnfamily.get_count(row)
                keys = self.columnfamily.remove(row)
            except:
                pass

    def rebuild_from_archive(self, rebuild_snapshot=True, period=None):
        """Rebuild the BF using the archived items.

        :rebuild_snapshot: Regenerate the snapshot on disk.
        :period: Do a partial rebuild using a single period (typically the last hours).
        """

        if not period:
            self.initialize_bitarray()
            last_period = self.current_period - dt.timedelta(days=self.expiration-1)
            hours = self._hour_range(last_period, dt.datetime.now())
            days = self._day_range(last_period, dt.datetime.now())
            rows = []
            for i,day in enumerate(days):
                k = None
                rows = ["%s_%s:%s" % (self.bf_name, day.strftime('%Y-%m-%d'), hour_str) for hour_str in ["%02d" % i for i in range(24)]]
                update_current = day == self.current_period

                for row in rows:
                    for k,v in self.columnfamily.xget(row):
                        self.add_rebuild(k)

                # Make sure there is some data before saving the snapshot.
                # We always save the snapshot of the current day to make
                # sure the BFs properties are in sync.
                if rebuild_snapshot and (k or update_current):
                    self.save_snaphot(override_period=day)

                if not update_current:
                    self.initialize_current_day_bitarray()
        else:
            period_str = period.strftime('%Y-%m-%d:%H')
            rows = ["%s_%s" % (self.bf_name, period_str)]
            for row in rows:
                for k,v in self.columnfamily.xget(row):
                    self.add_rebuild(k)



    def restore_from_disk(self, clean_old_snapshot=False):
        """Restore the state of the BF using previous snapshots.

        :clean_old_snapshot: Delete the old snapshot on the disk (period < current - expiration)

        This method will overwrite the BF parameters with the those in the snapshot.
        """
        base_filename = "%s/%s_%s_*.dat" % (self.snapshot_path, self.bf_name, self.expiration)
        availables_snapshots = glob.glob(base_filename)
        last_period = self.current_period - dt.timedelta(days=self.expiration-1)
        # Reinit the BF using the first snapshot
        if availables_snapshots:
            # Always read first snapshots before unioning in order
            # to retinitialize the parameters correctly
            self._read_snapshot(availables_snapshots[0])
            # Union the remaining
            for filename in availables_snapshots[1:]:
                snapshot_period = dt.datetime.strptime(filename.split('_')[-1].strip('.dat'), "%Y-%m-%d")
                if snapshot_period <  last_period and not clean_old_snapshot:
                    continue
                else:
                    current = snapshot_period == self.current_period
                    result = self._union_bf_from_file(filename, current=current)
                    if not result:
                        raise PDSError("Trying to load heterogeneous snapshots")
                if snapshot_period < last_period and clean_old_snapshot:
                    os.remove(filename)
            self.ready = True
            self.rebuild_hash = None
        return self.ready

    def compute_refresh_period(self):
        if not self.snapshot_to_load:
            return
        now = dt.datetime.utcnow()
        remaining_time = dt.datetime(now.year, now.month, now.day, 23, 59, 59) - now
        self.warm_period = remaining_time.seconds // (len(self.snapshot_to_load) + 2)

    def _should_warm(self):
        return (time.time() >= self.next_snapshot_load) and not self.hold_warming

    def set_rebuild_hash(self, rebuild_hash):
        self.rebuild_hash = rebuild_hash
        # reset warming
        self.snapshot_to_load = None
        self.hold_warming = True

    def reset_rebuild_hash(self):
        self.rebuild_hash = None
        self.hold_warming = False

    def warm(self, jittering_ratio=0.2, force_all=False):
        """Progressively load the previous snapshot during the day.

        Loading all the snapshots at once can takes a substantial amount of time. This method, if called
        periodically during the day will progressively load those snapshots one by one. Because many workers are
        going to use this method at the same time, we add a jittering to the period between load to avoid
        hammering the disk at the same time.
        """
        if not self._should_warm():
            return

        if self.snapshot_to_load == None:
            last_period = self.current_period - dt.timedelta(days=self.expiration-1)
            self.snapshot_to_load = []
            base_filename = "%s/%s_%s_*.dat" % (self.snapshot_path, self.bf_name, self.expiration)
            availables_snapshots = glob.glob(base_filename)
            for filename in availables_snapshots:
                snapshot_period = dt.datetime.strptime(filename.split('_')[-1].strip('.dat'), "%Y-%m-%d")
                if snapshot_period >= last_period:
                    self.snapshot_to_load.append(filename)
                    self.ready = False
            self.compute_refresh_period()

        if self.snapshot_to_load and not force_all:
            filename = self.snapshot_to_load.pop()
            self._union_bf_from_file(filename)
            jittering = self.warm_period * (np.random.random()-0.5) * jittering_ratio
            self.next_snapshot_load = time.time() + self.warm_period + jittering
            if not self.snapshot_to_load:
                self.ready = True
        elif self.snapshot_to_load:
            for filename in self.snapshot_to_load:
                self._union_bf_from_file(filename)
            self.snapshot_to_load = []
            self.ready = True

    def add_rebuild(self, key):
        super(DailyTemporalBloomFilter, self).add(key)

    def add(self, key_string, period=None):
        if isinstance(key_string, unicode):
            key = key_string.encode('utf8')
        else:
            key = key_string

        self.archive_bf_key(key, period)
        result = super(DailyTemporalBloomFilter, self).add(key)

        return result

    def resize(self, new_capacity=None, new_error_rate=None):
        self._set_capacity(new_capacity or self.capacity)
        self._set_error_rate(new_error_rate or self.error_rate)
        self._initialize_parameters()
        self._allocate_bitarrays()
        self.initialize_bitarray()
        self.initialize_current_day_bitarray()
        self.rebuild_from_archive(rebuild_snapshot=True)

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

    def save_snaphot(self, override_period=None):
        """Save the current state of the current day bitarray on disk.

        Save the internal representation (bitarray) into a binary file using this format:
            filename : name_expiration_2013-01-01.dat
        """
        period = override_period or self.current_period
        filename = "%s/%s_%s_%s.dat" % (self.snapshot_path, self.bf_name, self.expiration, period.strftime("%Y-%m-%d"))
        self._save_snapshot(filename)
