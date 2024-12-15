import numpy as np
import numba as nb
import math

@nb.jit(parallel=True)
def erp_dist(t0, t1, g):
    n0 = len(t0) + 1
    n1 = len(t1) + 1
    C = np.zeros((n0, n1))

    edgei = 0
    for i in range(1, n0):
        y0 = t0[i-1]
        edgei += math.fabs(y0 - g)
    C[1:, 0] = edgei

    edgej = 0
    for j in range(1, n1):
        y1 = t1[j-1]
        edgej += math.fabs(y1 - g)
    C[0, 1:] = edgej

    for i in range(1, n0):
        for j in range(1, n1):
            y0 = t0[i-1]
            y1 = t1[j-1]
            derp0 = C[i-1, j] + math.fabs(y0 - g)
            derp1 = C[i, j-1] + math.fabs(y1 - g)
            derp01 = C[i-1, j-1] + math.fabs(y0 - y1)
            C[i, j] = min(derp0, min(derp1, derp01))
    
    erp = C[n0-1, n1-1]
    return erp

