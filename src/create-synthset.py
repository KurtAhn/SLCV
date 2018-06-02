#!/usr/bin/python3
from __init__ import *
from os import path
from argparse import ArgumentParser
import acoustic as ax
import dataset as ds
import tensorflow as tf
sys.path.append('/home/kurt/Documents/etc/magphase/src')
import libutils as lu


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-l', '--labdir', dest='labdir', default=LABNDIR)
    p.add_argument('-t', '--txtdir', dest='txtdir', default=TXTDIR)
    p.add_argument('-o', '--outdir', dest='outdir', required=True)
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

        with open(path.join(a.txtdir, s+'.txt')) as f:
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

        with tf.python_io.TFRecordWriter(path.join(a.outdir, s+'.tfr')) as w:
            for xt, yt, zt in zip(x, y, z):
                feature = {
                    'w': F(bytes_list=BL(value=[bytes(zt, encoding='ascii')])),
                    'l': F(float_list=FL(value=xt))
                }
                example = E(features=FF(feature=feature))
                w.write(example.SerializeToString())

        print(s)
