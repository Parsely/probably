import os
import datetime as dt
import math
import time

import numpy as np

from collections import defaultdict
from bloomfilter_cython import BloomFilter, DailyTemporalBase
from pycassa import NotFoundException
from pycassa.pool import ConnectionPool
from pycassa.system_manager import SystemManager, SIMPLE_STRATEGY
from pycassa.columnfamily import ColumnFamily


class PDSError(Exception): pass


class WideRowBloomFilter(DailyTemporalBase):

    def __new__(cls, capacity, error_rate, expiration, name, cassandra_session, keyspace):
        return super(WideRowBloomFilter, cls).__new__(cls, capacity=capacity, error_rate=error_rate)

    def __init__(self, capacity, error_rate, expiration, name, cassandra_session, keyspace):
        filename = ""
        super(WideRowBloomFilter, self).__init__(capacity=capacity, error_rate=error_rate)
        self.bf_name = name
        self.expiration = expiration
        self.initialize_period()
        self.cassandra_session = cassandra_session
        self.cassandra_columns_family = "widerow_bf"
        self.keyspace = keyspace
        self.uncommited_keys = []
        self.uncommited_keys_per_bucket = defaultdict(list)
        self.uncommited_count = 0
        self.commit_batch_size = 1000
        self.commit_period = 5.0
        self.next_cassandra_commit = 0
        self.columnfamily = None
        self.ensure_cassandra_cf()
        self.initialized_at = time.time()
        self.ready = False

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

    def rebuild_from_archive(self):
        for k,v in self.columnfamily.xget(self.bf_name):
            self.add_rebuild(k)

    def add_rebuild(self, key):
        super(WideRowBloomFilter, self).add(key)

    def add(self, key_string, timestamp=None):
        if not self.ready:
            self.rebuild_from_archive()
            self.ready = True

        if isinstance(key_string, unicode):
            key = key_string.encode('utf8')
        else:
            key = key_string

        if not timestamp:
            timestamp = time.time()

        self.archive_bf_key(key, timestamp)
        result = super(WideRowBloomFilter, self).add(key)

        return result

    def resize(self, new_capacity=None, new_error_rate=None):
        self._set_capacity(new_capacity or self.capacity)
        self._set_error_rate(new_error_rate or self.error_rate)
        self._initialize_parameters()
        self._allocate_bitarrays()
        self.initialize_bitarray()
        self.initialize_current_day_bitarray()
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

    def _get_ttl(self, ts):
        """Make the correction to the ttl of delayed tuple.

        Looks like the C* driver don't have a expireat. So here we make
        the correction to the ttl.
        """
        return self.expiration - (int(time.time()) - ts)

    def archive_bf_key(self, key, ts):
        """Store the key in Cassandra.

        The batching insert here is using batch_insert() instead of a single multi-column insert()
        like in the other schema which is much less efficient. Anyway... this is just a test.
        """
        self.uncommited_keys.append((key, ts))
        if (time.time() > self.next_cassandra_commit or len(self.uncommited_keys) >= self.commit_batch_size):
            ttl = self._get_ttl(self.uncommited_keys[0][1]) ### Here we pick a single ttl for the batch
            if ttl > 0:
                batch = {k:'' for k,ts in self.uncommited_keys}
                self.columnfamily.insert(self.bf_name, batch, ttl=ttl)
            self.uncommited_keys = []
            self.next_cassandra_commit = time.time() + self.commit_period
