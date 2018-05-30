#!/usr/bin/python3
from __init__ import *
import sys
sys.path.append('/home/kurt/Documents/etc/magphase/src')
import libutils as lu
import acoustic as ax
import dataset as ds
from watts15 import *
import watts15.dnn as dnn
from subprocess import call
from os import path
import matplotlib.pyplot as pyplot
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np

# Vocode synthesized parameters


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-m', '--model', dest='model', required=True)
   # p.add_argument('-e', '--epoch', dest='epoch', type=int, default=None)
    p.add_argument('-r', '--recdir', dest='recdir', default=TRNDIR)
    p.add_argument('-x', '--sttdir', dest='sttdir', default=STTDIR)
    p.add_argument('-o', '--outdir', dest='outdir', default=OUTDIR)
    p.add_argument('-c', '--control', dest='control', type=float, nargs=NC, required=True)
    a = p.parse_args()

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f]

    mean = lu.read_binfile(path.join(a.sttdir, 'mean'), dim=ds.AX_LEN)
    stddev = lu.read_binfile(path.join(a.sttdir, 'stddev'), dim=ds.AX_LEN)

    with tf.Session().as_default() as session:
        n1 = dnn.SLCV1(mdldir=path.join(MDLDIR, a.model))
        n2 = dnn.SLCV2(nl=NL, nc=NC, w=n1._w, b=n1._b)

        for s in sentences:
            dataset = ds.load_trainset([path.join(a.recdir, s+'.tfr')])
            example = dataset.make_one_shot_iterator().get_next()
            output = []
            while True:
                try:
                    output.append(n2.predict(session.run(example)[0],
                                             np.array(a.control)))
                except tf.errors.OutOfRangeError:
                    break
            x = np.concatenate(output)
            x = ax.destandardize(x, mean, stddev)

            w = 1



            v = x[:,-1]
            x[:,ax.LF0][v <= mean[-1]] = 0.0
            x[:,ax.LF0] = np.log(x[:,ax.LF0])
#            x[:,ax.LF0][x[:,ax.LF0] == -float('Inf')] = 0.0
            x = ax.window(x[:,:ax.AX_LEN], np.ones([w]) / float(w))

            x2 = lu.read_binfile(path.join(ACODDIR, s+'.aco'), dim=ds.AX_LEN)
            x2 = ax.window(x2[:,:ax.AX_LEN], np.ones([w]) / float(w))

            t = [n * 0.005 for n in range(x.shape[0])]

            pyplot.plot(t, [y if y > 0 else 1e-10 for y in x[:,ax.LF0]])
            pyplot.plot(t, [np.log(y) if y > 0 else 1e-10 for y in x2[:len(t),ax.LF0]])
            pyplot.savefig(path.join(a.outdir, s+'_f0.pdf'))
            pyplot.close()

            # pyplot.plot(t, [y for y in x[:,60]])
            # pyplot.plot(t, [y for y in x2[:len(t),60]])
            # pyplot.savefig(s+'_real-0.pdf')
            # pyplot.close()

            lu.write_binfile(x[:,ax.MAG], path.join(a.outdir, s+'.mag'))
            lu.write_binfile(x[:,ax.REAL], path.join(a.outdir, s+'.real'))
            lu.write_binfile(x[:,ax.IMAG], path.join(a.outdir, s+'.imag'))
            lu.write_binfile(x[:,ax.LF0], path.join(a.outdir, s+'.lf0'))

            try:
                call('{script} -s {sentence} '
                     '-v {vocdir} -o {outdir} '
                     '-m {magdim} -p {phadim}'.format(
                    script=path.join(SRCDIR, 'synthesize-wav.py'),
                    sentence=s,
                    vocdir=a.outdir,
                    outdir=a.outdir,
                    magdim=ax.MAG_LEN,
                    phadim=ax.REAL_LEN).split())
            except:
                pass
