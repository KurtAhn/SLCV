from __init__ import *
import numpy as np
import scipy.interpolate


AX_DIM = 0
MAG_DIM = cfg_data.get('nm', 60)
PHASE_DIM = cfg_data.get('np', 20)
REAL_DIM = PHASE_DIM
IMAG_DIM = PHASE_DIM
LF0_DIM = 1


MAG = slice(AX_DIM, AX_DIM+MAG_DIM)
AX_DIM += MAG_DIM
REAL = slice(AX_DIM, AX_DIM+REAL_DIM)
AX_DIM += REAL_DIM
IMAG = slice(AX_DIM, AX_DIM+IMAG_DIM)
AX_DIM += IMAG_DIM
LF0 = slice(AX_DIM, AX_DIM+LF0_DIM)
AX_DIM += LF0_DIM


def acoustic(**kwargs):
    rows = kwargs.get('rows', 1)
    return np.concatenate([kwargs.get('mag', np.zeros([rows,MAG_DIM])),
                           kwargs.get('real', np.zeros([rows, REAL_DIM])),
                           kwargs.get('imag', np.zeros([rows, IMAG_DIM])),
                           kwargs.get('lf0', np.zeros([rows, LF0_DIM]))
                           ], axis=1)


def window(a, w):
    w = np.reshape(w, [1, w.size])
    f = np.zeros(a.shape)
    a = np.concatenate([np.zeros((w.size-1, a.shape[1])),
                        a,
                        np.zeros((w.size-1, a.shape[1]))],
                        axis=0)
    for r in range(f.shape[0]):
        f[r,:] = w @ a[r:r+w.size,:]
    return f


def velocity(a):
    return window(a, np.array([-0.5, 0.0, 0.5]))


def acceleration(a):
    return window(a, np.array([1.0, -2.0, 1.0]))


def interpolate_f0(a, kind='linear'):
    a = np.copy(a)
    vi = np.where(a[:,LF0] > 0.0)[0]
    i = scipy.interpolate.interp1d(vi,
                                   a[vi, LF0],
                                   kind=kind,
                                   axis=0,
                                   bounds_error=False,
                                   fill_value='extrapolate')
    a[:, LF0] = i(np.arange(a.shape[0]))
    return a


def voicing(a, threshold):
    b = np.zeros([a.shape[0], 1])
    b[a[:,LF0] > threshold] = 1.0
    return b


def standardize(a, mean, stddev):
    return (a - mean) / stddev


def destandardize(a, mean, stddev):
    return a * stddev + mean
