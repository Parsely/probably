import os
import datetime as dt
import math
import time
import logging

import numpy as np

from collections import defaultdict
from bloomfilter_cython import BloomFilter
from scalable_bloomfilter import ScalableBloomFilter
from pycassa import NotFoundException
from pycassa.pool import ConnectionPool
from pycassa.system_manager import SystemManager, SIMPLE_STRATEGY
from pycassa.columnfamily import ColumnFamily


log = logging.getLogger(__name__)

class PDSError(Exception): pass


class WideRowBloomFilter(object):
    """Simple Scalable Bloom Filter backed by a C* widerow.

    TODO: Switch to CQL3
    """
    def __init__(self, capacity, error_rate, expiration, name,
                 cassandra_session, keyspace, should_warm=True,
                 buffer_size=5000):
        self.bf_name = name
        self.expiration = expiration
        self.initialize_period()
        self.cassandra_session = cassandra_session
        self.cassandra_columns_family = "widerow_bf"
        self.keyspace = keyspace
        self.uncommited_keys = []
        self.uncommited_count = 0
        self.commit_batch_size = 1000
        self.commit_period = 5.0
        self.next_cassandra_commit = 0
        self.columnfamily = None
        self.buffer_size = buffer_size
        self.ensure_cassandra_cf()
        self.initial_capacity = capacity
        self.error_rate = error_rate
        self.bf = None
        self.initialize_bf()
        self.ready = not should_warm

    @property
    def capacity(self):
        return self.bf.capacity

    def current_row_count(self):
        return self.columnfamily.get_count(self.bf_name)

    def initialize_bf(self):
        """ Initialize the Scalable Bloom Filter with optimal capacity.

        Here we fetch the count of the widerow and add some provisioning. The SBF
        will scale automatically, but choosing the right capacity at this stage will
        make sure that the SBF is not fragmented. This way, we can remove the fragmentation
        of the SBF simply by reinitialize it.
        """
        self.initial_capacity = max(int(self.current_row_count() * 1.5), self.initial_capacity)
        self.bf = ScalableBloomFilter(self.initial_capacity, self.error_rate)
        log.info("Initialized %s ScalableBloomFilter with initial_capacity=%d "
                 "error_rate=%.2f", self.bf_name, self.initial_capacity,
                 self.error_rate)

    def ensure_cassandra_cf(self):
        s = SystemManager(self.cassandra_session.server_list[0])
        if self.keyspace not in s.list_keyspaces():
            s.create_keyspace(self.keyspace, SIMPLE_STRATEGY, {'replication_factor': '2'})
        if self.cassandra_columns_family not in s.get_keyspace_column_families(self.keyspace):
            s.create_column_family(self.keyspace, self.cassandra_columns_family)
        self.columnfamily = ColumnFamily(self.cassandra_session, self.cassandra_columns_family)

    def rebuild_from_archive(self):
        """Rebuild the SBF using data in C*."""
        self.initialize_bf()
        start = time.time()
        log.info("%s.%s rebuild_from_archive starting...",
                 self.cassandra_columns_family, self.bf_name)
        for k,v in self.columnfamily.xget(self.bf_name,
                                          buffer_size=self.buffer_size):
            self.bf.add(k)
        secs = time.time() - start
        log.info("%s.%s rebuild_from_archive completed in %.2fs.",
                 self.cassandra_columns_family, self.bf_name, secs)

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
        result = self.bf.add(key)

        return result

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

    def commit_bf_keys(self):
        if not self.uncommited_keys:
            return
        ts = self.uncommited_keys[-1][1]
        ttl = self._get_ttl(ts) ### Here we pick a single ttl for the batch
        if ttl > 0:
            batch = {k:'' for k,ts in self.uncommited_keys}
            self.columnfamily.insert(self.bf_name, batch, ttl=ttl)
        self.uncommited_keys = []
        self.next_cassandra_commit = time.time() + self.commit_period

    def archive_bf_key(self, key, ts):
        """Store the key in Cassandra."""
        self.uncommited_keys.append((key, ts))
        if (time.time() > self.next_cassandra_commit or len(self.uncommited_keys) >= self.commit_batch_size):
            self.commit_bf_keys()
