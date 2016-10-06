'''
Cython module for fast maintenance process
'''

import cython
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, pow
from six.moves import range
cimport numpy as np

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef tuple maintenance_cyt(np.ndarray[np.uint8_t, ndim=1, mode="c"] cells,
                           long int cells_size,
                           long int num_iterations,
                           long int head):
    '''
    Maintenance process for the Countdown Bloom Filter
    '''
    cdef long int refresh_head = head
    cdef long int itr
    cdef long int nonzero = 0

    for itr in range(num_iterations):
        if cells[refresh_head] != 0:
            cells[refresh_head] -= 1
            if cells[refresh_head] != 0:
                nonzero += 1
        refresh_head = (refresh_head + 1) % cells_size
    return refresh_head, nonzero


def maintenance(np.ndarray[np.uint8_t, ndim=1, mode="c"] cells,
                long int cells_size, long int num_iterations, head):
    return maintenance_cyt(cells, cells_size, num_iterations, head)


