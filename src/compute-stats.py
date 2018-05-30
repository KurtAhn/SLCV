#!/usr/bin/python3

from __init__ import *
from os import path
from argparse import ArgumentParser
import acoustic as ax
import dataset as ds
sys.path.append('/home/kurt/Documents/etc/magphase/src')
import libutils as lu
import numpy as np


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-a', '--acodir', dest='acodir', default=ACODDIR)
    p.add_argument('-x', '--sttdir', dest='sttdir', default=STTDIR)
    a = p.parse_args()

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f if l]

    mean = np.zeros([1, ds.AX_LEN])
    stddev = np.zeros([1, ds.AX_LEN])
    n = 0
    for s in sentences:
        data = lu.read_binfile(path.join(a.acodir, s+'.aco'), dim=ds.AX_LEN)
        for dt in data:
            mean += dt
            stddev += np.multiply(dt, dt)
            n += 1
    mean /= n
    stddev = np.sqrt(stddev / n - np.multiply(mean, mean))
    eprint(mean)
    eprint(stddev)

    lu.write_binfile(mean, path.join(a.sttdir, 'mean'))
    lu.write_binfile(stddev, path.join(a.sttdir, 'stddev'))
