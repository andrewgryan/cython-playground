import numpy as np
cimport numpy as np
cimport cython


ctypedef np.uint8_t uint8


def contains(
        double[:, :, :] vertices,
	double[:] x,
	double[:] y):
    cdef Py_ssize_t i
    cdef uint8[:] inside = np.zeros(vertices.shape[0], dtype=np.uint8)
    for i in range(vertices.shape[0]):
        inside[i] = same_side_test(vertices[i], x[i], y[i])
    return inside


@cython.cdivision(True)
@cython.boundscheck(False)
cpdef bint same_side_test(double[:, :] vertices, double x, double y):
    """Points lie inside convex polygons if they lie on the same side as every
    face.
    :returns: logical indicating point inside polygon
    """
    cdef double x1, y1, x2, y2, current, previous
    cdef int ipoint, npoints
    npoints = vertices.shape[0]
    previous = 0
    for ipoint in range(npoints):
        x1 = vertices[ipoint, 0]
        y1 = vertices[ipoint, 1]
        x2 = vertices[(ipoint + 1) % npoints, 0]
        y2 = vertices[(ipoint + 1) % npoints, 1]
        current = (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)
        if (current == 0):
            # Points on boundary are considered part of the polygon
            return True
        if (previous * current) < 0:
            return False
        previous = current
    return True
