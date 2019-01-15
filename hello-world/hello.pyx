from cython.parallel import prange
import numpy as np
cimport numpy as np


ctypedef np.uint8_t uint8


def multiply_serial(double[:] x, double factor):
    cdef Py_ssize_t i
    for i in range(x.shape[0]):
        x[i] = factor * x[i]


def multiply_parallel(double[:] x, double factor):
    cdef Py_ssize_t i
    for i in prange(x.shape[0], nogil=True):
        x[i] = factor * x[i]


def grid_membership(double[:, :, :] corners):
    cdef Py_ssize_t i
    cdef uint8[:] inside = np.zeros(corners.shape[0], dtype=np.uint8)
    for i in range(corners.shape[0]):
        inside[i] = contains(corners[i])
    return inside


cdef uint8 contains(double[:, :] corners):
    return True
