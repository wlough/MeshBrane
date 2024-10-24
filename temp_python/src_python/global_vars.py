from numpy import int32 as np_int32
from numpy import float64 as np_float64
from numpy import dtype
from numba import from_dtype

INT_TYPE = "int32"
FLOAT_TYPE = "float64"

_NUMPY_INT_ = dtype(INT_TYPE).type
_NUMPY_FLOAT_ = dtype(FLOAT_TYPE).type

_NUMBA_INT_ = from_dtype(_NUMPY_INT_)
_NUMBA_FLOAT_ = from_dtype(_NUMPY_FLOAT_)

# _data_type_relation = [
#     ('int8', 'i1'),
#     ('char', 'i1'),
#     ('uint8', 'u1'),
#     ('uchar', 'b1'),
#     ('uchar', 'u1'),
#     ('int16', 'i2'),
#     ('short', 'i2'),
#     ('uint16', 'u2'),
#     ('ushort', 'u2'),
#     ('int32', 'i4'),
#     ('int', 'i4'),
#     ('uint32', 'u4'),
#     ('uint', 'u4'),
#     ('float32', 'f4'),
#     ('float', 'f4'),
#     ('float64', 'f8'),
#     ('double', 'f8')
# ]
# _data_type_relation_cpp = [
#     ('int8', 'int8_t'),
#     ('char', 'char'),
#     ('uint8', 'uint8_t'),
#     ('uchar', 'unsigned char'),
#     ('int16', 'int16_t'),
#     ('short', 'short'),
#     ('uint16', 'uint16_t'),
#     ('ushort', 'unsigned short'),
#     ('int32', 'int32_t'),
#     ('int', 'int'),
#     ('uint32', 'uint32_t'),
#     ('uint', 'unsigned int'),
#     ('float32', 'float'),
#     ('float', 'float'),
#     ('float64', 'double'),
#     ('double', 'double')
# ]
