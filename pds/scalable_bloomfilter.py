from bloomfilter_cython import BloomFilter

class ScalableBloomFilter(object):
    """ Scalable Bloom Filter.

    Uses approach described in:
    Almeida, Paulo Sergio, et al. "Scalable bloom filters." Information Processing Letters 101.6 (2007): 255-261.
    http://asc.di.fct.unl.pt/~nmp/pubs/ref--04.pdf
    """
    def __init__(self, initial_capacity=100, error_rate=0.001):
        # geometric progression parameter (typical value: 2,4)
        self.scale = 2
        # error probability ratio (typical value: 0.5-0.9)
        # See Fig. 5 of paper.
        self.ratio = 0.5
        self.initial_capacity = initial_capacity
        # The compounded error will converge to this value.
        self.error_rate = error_rate
        self.filters = []

    def __contains__(self, key):
        for f in reversed(self.filters):
            if key in f:
                return True
        return False

    def initialize_bitarray(self):
        self.filters = []

    def add(self, key):
        """
        Add a element to the set and return membership.

        Also create the serie of filters using a geometric
        progression (s=2).
        """
        if key in self:
            return True
        if not self.filters:
            filter = BloomFilter(
                capacity=self.initial_capacity,
                error_rate=self.error_rate * (1.0 - self.ratio))
            self.filters.append(filter)
        else:
            filter = self.filters[-1]
            if filter.count >= filter._get_capacity():
                filter = BloomFilter(
                    capacity=filter._get_capacity() * self.scale,
                    error_rate=filter._get_error_rate() * self.ratio)
                self.filters.append(filter)
        filter.add(key)
        return False

    @property
    def capacity(self):
        """Returns the total capacity for all filters in this SBF."""
        return sum([f._get_capacity() for f in self.filters])

    @property
    def count(self):
        return len(self)

    def compounded_error(self):
        """Return the compounded error.

        In this SBF, a membership query is actually a serie of queries
        over all the filters. For each sub-queries, there is some false
        positive probability. So the effective error (compounded error) is the
        product of the errors.

        In order to maintain this compounded error
        to the defined error rate, each successive filters is created
        with a tighter maximum error probability on a geometric progression.
        """
        cum_error = 1.0
        for bf in self.filters:
            cum_error *= (1.0 - bf._get_error_rate())
        return 1 - cum_error

    def __len__(self):
        """Returns the total number of elements stored in this SBF"""
        return sum([f.count for f in self.filters])

