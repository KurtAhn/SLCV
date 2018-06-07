#!/usr/bin/env python
from __init__ import load_config
import sys, os
sys.path.append(os.environ['MAGPHASE'])
import libutils as lu
from os import path
import tensorflow as tf
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

    mean = lu.read_binfile(path.join(STTDIR, 'mean'), dim=ds.AX_LEN)
    stddev = lu.read_binfile(path.join(STTDIR, 'stddev'), dim=ds.AX_LEN)

    F = tf.train.Feature
    FF = tf.train.Features
    E = tf.train.Example
    BL = tf.train.BytesList
    FL = tf.train.FloatList

    for s in sentences:
        x = lu.read_binfile(path.join(LAB3DIR, s+'.lab'), dim=ds.LX_LEN)

        with open(path.join(TXTDIR, s+'.txt')) as f:
            words = re.sub(r'[^ A-Za-z\']','', next(f).lower()).split()
        #print2(words)

        n = 0
        z = [words[0]]
        for t in range(1,x.shape[0]):
            try:
                if x[t,ds.WORD] <= 0.01:
                    z.append('#')
                else:
                    try:
                        if x[t-1,ds.WORD] != x[t,ds.WORD]:
                            n += 1
                    except IndexError:
                        pass
                    try:
                        z.append(words[n])
                    except IndexError:
                        #print2(s, t, x.shape, x[:,ds.WORD])
                        raise
            except IndexError:
                continue

        with tf.python_io.TFRecordWriter(path.join(SYNDIR, s+'.tfr')) as w:
            for xt, yt, zt in zip(x, y, z):
                feature = {
                    'w': F(bytes_list=BL(value=[bytes(zt, encoding='ascii')])),
                    'l': F(float_list=FL(value=xt))
                }
                example = E(features=FF(feature=feature))
                w.write(example.SerializeToString())

        print1(s)
        flush1()
