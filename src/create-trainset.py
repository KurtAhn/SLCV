#!/usr/bin/python3

from __init__ import *
from os import path
from argparse import ArgumentParser
import acoustic as ax
import dataset as ds
import tensorflow as tf
sys.path.append('/home/kurt/Documents/etc/magphase/src')
import libutils as lu
import re


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-l', '--labdir', dest='labdir', default=LABNDIR)
    p.add_argument('-a', '--acodir', dest='acodir', default=ACODDIR)
    p.add_argument('-x', '--sttdir', dest='sttdir', default=STTDIR)
    p.add_argument('-t', '--txtdir', dest='txtdir', default=TXTDIR)
    p.add_argument('-o', '--outdir', dest='outdir', default=TRNDIR)
    a = p.parse_args()

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f if l]

    mean = lu.read_binfile(path.join(a.sttdir, 'mean'), dim=ds.AX_LEN)
    stddev = lu.read_binfile(path.join(a.sttdir, 'stddev'), dim=ds.AX_LEN)

    F = tf.train.Feature
    FF = tf.train.Features
    E = tf.train.Example
    BL = tf.train.BytesList
    FL = tf.train.FloatList

    for s in sentences:
        x = lu.read_binfile(path.join(a.labdir, s+'.lab'), dim=ds.LX_LEN)
        y = lu.read_binfile(path.join(a.acodir, s+'.aco'), dim=ds.AX_LEN)
        y = ax.standardize(y, mean, stddev)
        #eprint(y.shape)

        with open(path.join(a.txtdir, s+'.txt')) as f:
            words = re.sub(r'[^ A-Za-z\']','', next(f).lower()).split()
        #print(words)

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
                        #print(s, t, x.shape, x[:,ds.WORD])
                        raise
            except IndexError:
                continue

        with tf.python_io.TFRecordWriter(path.join(a.outdir, s+'.tfr')) as w:
            for xt, yt, zt in zip(x, y, z):
                feature = {
                    's': F(bytes_list=BL(value=[bytes(s, encoding='ascii')])),
                    'w': F(bytes_list=BL(value=[bytes(zt, encoding='ascii')])),
                    'l': F(float_list=FL(value=xt)),
                    'a': F(float_list=FL(value=yt))
                }
                example = E(features=FF(feature=feature))
                w.write(example.SerializeToString())

        print(s)
