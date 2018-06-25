#!/usr/bin/env python
from __init__ import load_config
import sys, os
sys.path.append(os.environ['MAGPHASE'])
import libutils as lu
from os import path
from nltk.tokenize import word_tokenize
import tensorflow as tf
import re
from random import random
from collections import defaultdict
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

    mean = lu.read_binfile(path.join(STTDIR, 'mean'), dim=ds.AX_DIM)
    stddev = lu.read_binfile(path.join(STTDIR, 'stddev'), dim=ds.AX_DIM)

    F = tf.train.Feature
    FF = tf.train.Features
    E = tf.train.Example
    BL = tf.train.BytesList
    FL = tf.train.FloatList
    W = tf.python_io.TFRecordWriter

    for s in sentences:
        l = lu.read_binfile(path.join(LAB3DIR, s+'.lab'), dim=ds.LX_DIM)
        y = lu.read_binfile(path.join(ACO3DIR, s+'.aco'), dim=ds.AX_DIM)
        y = ax.standardize(y, mean, stddev)

        with open(path.join(WORDIR, s+'.txt')) as f:
            words = next(f).lstrip().split()

        n = 0
        w = [words[n]]
        for t in range(1, l.shape[0]):
            if l[t, ds.WIP] <= 0.01:
                w.append('<unk>')
            else:
                try:
                    if l[t-1, ds.WIP] != l[t, ds.WIP]:
                        n += 1
                except IndexError:
                    pass
                try:
                    w.append(words[n])
                except IndexError:
                    continue

        with W(path.join(TRNDIR, s+'.tfr')) as writer:
            for lt, yt, wt in zip(l, y, w):
                feature = {
                    's': F(bytes_list=BL(value=[bytes(s, encoding='ascii')])),
                    'w': F(bytes_list=BL(value=[bytes(wt, encoding='ascii')])),
                    'l': F(float_list=FL(value=lt)),
                    'a': F(float_list=FL(value=yt))
                }
                example = E(features=FF(feature=feature))
                writer.write(example.SerializeToString())
        print1(s)
        flush1()
