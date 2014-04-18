import os
import cPickle
import glob
import datetime as dt
import math
import time
import zlib

import bitarray
import numpy as np

from bloomfilter_cython import BloomFilter, DailyTemporalBase
from pycassa import NotFoundException
from pycassa.pool import ConnectionPool
from pycassa.system_manager import SystemManager, SIMPLE_STRATEGY
from pycassa.columnfamily import ColumnFamily


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

    def __new__(cls, capacity, error_rate, expiration, name, cassandra_session, snapshot_path='./'):
        return super(DailyTemporalBloomFilter, cls).__new__(cls, capacity=capacity, error_rate=error_rate)

    def __init__(self, capacity, error_rate, expiration, name, cassandra_session, snapshot_path='./'):
        filename = ""
        super(DailyTemporalBloomFilter, self).__init__(capacity=capacity, error_rate=error_rate)
        self.bf_name = name
        self.expiration = expiration
        self.initialize_period()
        self.cassandra_session = cassandra_session
        self.cassandra_columns_family = "temporal_bf"
        self.keyspace = 'parsely'
        self.uncommited_keys = []
        self.commit_batch = 1000
        self.columnfamily = None
        self.ensure_cassandra_cf()
        self.snapshot_path = snapshot_path

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
            self.columnfamily.insert('%s_%s' % (self.bf_name, current_period_hour), {k:'' for k in self.uncommited_keys})
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

    def rebuild_from_archive(self, rebuild_snapshot=True):
        """Rebuild the BF using the archived items"""
        self.initialize_bitarray()

        #if rebuild_snapshot:
        #    self.delete_snapshots()

        def multi_rows_itr(rows):
            for row in rows.values():
                for k in row.keys():
                    yield k

        last_period = self.current_period - dt.timedelta(days=self.expiration-1)
        hours = self._hour_range(last_period, dt.datetime.now())
        days = self._day_range(last_period, dt.datetime.now())
        rows = []
        for i,day in enumerate(days):
            rows = ["%s_%s:%s" % (self.bf_name, day.strftime('%Y-%m-%d'), hour_str) for hour_str in ["%02d" % i for i in range(24)]]
            rows_content = self.columnfamily.multiget(rows, column_count=1E6)
            update_current = day == self.current_period

            for k in multi_rows_itr(rows_content):
                self.add_rebuild(k, update_current)

            if rebuild_snapshot:
                self.save_snaphot(override_period=day)

            if not update_current:
                self.initialize_current_day_bitarray()

    def restore_from_disk(self, clean_old_snapshot=False):
        """Restore the state of the BF using previous snapshots.

        :clean_old_snapshot: Delete the old snapshot on the disk (period < current - expiration)
        """
        base_filename = "%s/%s_%s_*.dat" % (self.snapshot_path, self.bf_name, self.expiration)
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

    def add_rebuild(self, key, update_current=True):
        super(DailyTemporalBloomFilter, self).add(key, update_current)

    def add(self, key_string):
        if isinstance(key_string, unicode):
            key = key_string.encode('utf8')
        else:
            key = key_string

        self.archive_bf_key(key)
        result = super(DailyTemporalBloomFilter, self).add(key)

        return result

    def resize(self, new_capacity=None, new_error_rate=None):
        self._set_capacity(new_capacity or self.capacity)
        self._set_error_rate(new_error_rate or self.error_rate)
        self._initialize_parameters()
        self.initialize_bitarray()
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
