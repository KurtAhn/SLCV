#!/usr/bin/env python
from __init__ import load_config
from os import path
import numpy as np
from sklearn.decomposition import PCA
from argparse import ArgumentParser


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-c', '--config', dest='config', required=True)
    p.add_argument('-e', '--embedding', dest='embedding', required=True)
    p.add_argument('-o', '--output', dest='output', required=True)
    p.add_argument('-d', '--dimensions', dest='dimensions', type=int, nargs=2, required=True)
    a = p.parse_args()

    load_config(a.config)
    from __init__ import *

    large = np.memmap(path.join(EMBDIR, a.output+'.emb'),
                      dtype='float', shape=tuple(a.dimensions), mode='w+')

    with open(path.join(EMBDIR, a.output+'.dim'), 'w') as fd:
        fd.write('{} {}'.format(*a.dimensions))

    with open(a.embedding) as fe,\
         open(path.join(EMBDIR, a.output+'.vcb'), 'w') as fv:
        for n, line in enumerate(fe):
            tokens = line.split(' ')
            fv.write(tokens[0]+'\n')
            large[n,:] = np.array([float(t) for t in tokens[-a.dimensions[1]:]])
