import os
import cPickle
import glob
import datetime as dt
import math
import time
import zlib

import bitarray
import numpy as np

from bloomfilter_cython import BloomFilter
from pycassa import NotFoundException
from pycassa.pool import ConnectionPool
from pycassa.system_manager import SystemManager, SIMPLE_STRATEGY
from pycassa.columnfamily import ColumnFamily


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

    def __new__(cls, capacity, error_rate, expiration, name, cassandra_session):
        return super(DailyTemporalBloomFilter, cls).__new__(cls, capacity=capacity, error_rate=error_rate)

    def __init__(self, capacity, error_rate, expiration, name, cassandra_session):
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

    def rebuild_from_archive(self):
        """Rebuild the BF using the archived items"""
        #self.initialize_bitarray()
        last_period = self.current_period - dt.timedelta(days=self.expiration-1)
        hours = self._hour_range(last_period, dt.datetime.now())
        rows = []
        for i,hour in enumerate(hours):
            row = "%s_%s" % (self.bf_name, hour.strftime('%Y-%m-%d:%H'))
            rows.append(row)
        rows_content = self.columnfamily.multiget(rows, column_count=1E6)

        for row_content in rows_content.values():
            for k in row_content.keys():
                self.add_rebuild(k)

    def add_rebuild(self, key):
        super(DailyTemporalBloomFilter, self).add(key)

    def add(self, key_string):
        if isinstance(key_string, unicode):
            key = key_string.encode('utf8')
        else:
            key = key_string

        self.archive_bf_key(key)
        result = super(DailyTemporalBloomFilter, self).add(key)

        return result

    def resize(self, new_capacity):
        self._set_capacity(new_capacity)
        self._initialize_parameters()
        self.initialize_bitarray()
        self.rebuild_from_archive()

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

