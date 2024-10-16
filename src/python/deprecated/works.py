import cython
from cython.cimports.cython.view import array as cvarray
import numpy as np

# Memoryview on a NumPy array
narr = np.arange(27, dtype=np.dtype("i")).reshape((3, 3, 3))
narr_view = cython.declare(cython.int[:, :, :], narr)

# Memoryview on a C array
carr = cython.declare(cython.int[3][3][3])
carr_view = cython.declare(cython.int[:, :, :], carr)

# Memoryview on a Cython array
cyarr = cvarray(shape=(3, 3, 3), itemsize=cython.sizeof(cython.int), format="i")
cyarr_view = cython.declare(cython.int[:, :, :], cyarr)

# Show the sum of all the arrays before altering it
print(f"NumPy sum of the NumPy array before assignments: {narr.sum()}")

# We can copy the values from one memoryview into another using a single
# statement, by either indexing with ... or (NumPy-style) with a colon.
carr_view[...] = narr_view
cyarr_view[:] = narr_view
# NumPy-style syntax for assigning a single value to all elements.
narr_view[:, :, :] = 3

# Just to distinguish the arrays
carr_view[0, 0, 0] = 100
cyarr_view[0, 0, 0] = 1000

# Assigning into the memoryview on the NumPy array alters the latter
print(f"NumPy sum of NumPy array after assignments: {narr.sum()}")


# A function using a memoryview does not usually need the GIL
@cython.nogil
@cython.ccall
def sum3d(arr: cython.int[:, :, :]) -> cython.int:
    i: cython.size_t
    j: cython.size_t
    k: cython.size_t
    I: cython.size_t
    J: cython.size_t
    K: cython.size_t
    total: cython.int = 0
    I = arr.shape[0]
    J = arr.shape[1]
    K = arr.shape[2]
    for i in range(I):
        for j in range(J):
            for k in range(K):
                total += arr[i, j, k]
    return total


# A function accepting a memoryview knows how to use a NumPy array,
# a C array, a Cython array...
print(f"Memoryview sum of NumPy array is {sum3d(narr)}")
print(f"Memoryview sum of C array is {sum3d(carr)}")
print(f"Memoryview sum of Cython array is {sum3d(cyarr)}")
# ... and of course, a memoryview.
print(f"Memoryview sum of C memoryview is {sum3d(carr_view)}")
