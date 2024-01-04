from src.numdiff.some_funs import *

# ##############################################################################
# # save/load finite difference weight functions ###############################
# ##############################################################################
# try:
#     with open("./numdiff/findiff_weights1", "rb") as _f:
#         findiff_weights1 = dill.load(_f)
#     with open("./numdiff/findiff_weights2", "rb") as _f:
#         findiff_weights2 = dill.load(_f)
#     with open("./numdiff/findiff_weights3", "rb") as _f:
#         findiff_weights3 = dill.load(_f)
#     with open("./numdiff/findiff_weights4", "rb") as _f:
#         findiff_weights4 = dill.load(_f)
# except FileNotFoundError:
#     save_findiff_weight_funs()
