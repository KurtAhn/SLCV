#!/usr/bin/env python
from __init__ import load_config
from os import path
import numpy as np
from sklearn.decomposition import PCA
from argparse import ArgumentParser


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-c', '--config', dest='config', required=True)
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-e', '--embedding', dest='embedding', required=True)
    p.add_argument('-o', '--output', dest='output', required=True)
    p.add_argument('-d', '--dimensions', dest='dimensions', required=True)
    a = p.parse_args()

    load_config(a.config)
    from __init__ import *
    from ahn18 import *

    print2('Binarizing')
    large = np.memmap(path.join(EMBDIR, a.output+'.emb'),
                      dtype='float', shape=tuple(a.dimensions), mode='w+')

    with open(a.embedding) as fi,\
         open(path.join(VOCDIR, a.output+'.voc'), 'w') as fv:
        for n, line in enumerate(fi):
            tokens = line.split()
            fv.write(tokens[0]+'\n')
            large[n,:] = np.array([float(t) for t in tokens[1:]])
