from cython.parallel import prange

cdef int i
cdef int n = 30
cdef int total = 0

for i in prange(n, nogil=True):
    total += i

print(total)
