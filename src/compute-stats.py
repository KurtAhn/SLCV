#!/usr/bin/env python
from __init__ import load_config
import sys, os
sys.path.append(os.environ['MAGPHASE'])
import libutils as lu
from os import path
import numpy as np
from argparse import ArgumentParser


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-c', '--config', dest='config', required=True)
    a = p.parse_args()

    load_config(a.config)
    from __init__ import *
    import acoustic as ax
    import dataset as ds

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f if l]

    mean = np.zeros([1, ds.AX_DIM])
    stddev = np.zeros([1, ds.AX_DIM])
    n = 0
    for s in sentences:
        data = lu.read_binfile(path.join(ACO3DIR, s+'.aco'), dim=ds.AX_DIM)
        for dt in data:
            mean += dt
            stddev += np.multiply(dt, dt)
            n += 1
    mean /= n
    stddev = np.sqrt(stddev / n - np.multiply(mean, mean))

    lu.write_binfile(mean, path.join(STTDIR, 'mean'))
    lu.write_binfile(stddev, path.join(STTDIR, 'stddev'))
