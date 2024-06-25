from dolfin import *
from mshr import *
import numpy as np
import time


# ---------------------------------------------------------------------------------------------------------------------
def TV(mesh, vol_face_fn, bdy_length_fn, intlengths, intcells, boundlengths, boundfaces, cut_result):
    var1 = cut_result.vector()[intcells[:, 1]] - cut_result.vector()[intcells[:, 2]]
    TV = sum(intlengths * np.abs(var1)) + sum(boundlengths * np.abs(cut_result.vector()[boundfaces]))
    return TV


def _sparsify(U, K, c):
    ind = np.where(c <= 0.0000001)[0]
    U = np.delete(U, ind, 1)
    K = np.delete(K, ind, 1)
    c = np.delete(c, ind, 0)
    return U, K, c

# ---------------------------------------------------------------------------------------------------------------------
