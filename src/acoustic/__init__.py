import numpy as np


AX_LEN = 0
MAG_LEN = 60
REAL_LEN = 45
IMAG_LEN = 45
LF0_LEN = 1


MAG = slice(AX_LEN, AX_LEN+MAG_LEN)
AX_LEN += MAG_LEN
REAL = slice(AX_LEN, AX_LEN+REAL_LEN)
AX_LEN += REAL_LEN
IMAG = slice(AX_LEN, AX_LEN+IMAG_LEN)
AX_LEN += IMAG_LEN
LF0 = slice(AX_LEN, AX_LEN+LF0_LEN)
AX_LEN += LF0_LEN


def acoustic(**kwargs):
    rows = kwargs.get('rows', 1)
    return np.concatenate([kwargs.get('mag', np.zeros([rows,MAG_LEN])),
                           kwargs.get('real', np.zeros([rows, REAL_LEN])),
                           kwargs.get('imag', np.zeros([rows, IMAG_LEN])),
                           kwargs.get('lf0', np.zeros([rows, LF0_LEN]))
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


def interpolate_f0(a):
    f = a[:,LF0]
    a_ = np.copy(a)
    f_ = a_[:,LF0]
    last = 0.0
    for i in range(f.size):
        if f[i] <= 0.0:
            j = i+1
            for j in range(i+1, f.size):
                if f[j] > 0.0:
                    break
            if j < f.size-1:
                if last > 0.0:
                    step = (f[j] - f[i-1]) / float(j-i)
                    for k in range(i, j):
                        f_[k] = f[i-1] + step * (k-i+1)
                else:
                    for k in range(i, j):
                        f_[k] = f[j]
            else:
                for k in range(i, f.size):
                    f_[k] = last
        else:
            f_[i] = last = f[i]
    return a_


def voicing(a):
    b = np.zeros([a.shape[0], 1])
    b[a[:,LF0] > 1.0] = 1.0
    return b


def standardize(a, mean, stddev):
    return (a - mean) / stddev


def destandardize(a, mean, stddev):
    return a * stddev + mean
