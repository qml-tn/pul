import tensorflow as tf
import numpy as np


def bond_dimension(D, d, n, i):
    """Returns the right bond dimension of an MPS with open boundary condidion
    D:  maximum bond dimension
    d:  local hilbert space dimension
    n:  length of the MPS
    i:  Site index for the right bond. First site has index i=0 and the last i=n-1.
    """
    return np.min([d**(i + 1), d**(n - 1 - i), D])
