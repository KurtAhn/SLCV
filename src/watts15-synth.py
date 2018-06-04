#!/usr/bin/env python
from __init__ import *
import sys
import os
sys.path.append(os.environ['MAGPHASE'])
import libutils as lu
import acoustic as ax
from subprocess import call
from os import path
from os import mkdir
import matplotlib.pyplot as pyplot
import tensorflow as tf
import numpy as np
from argparse import ArgumentParser


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-c', '--config', dest='config', required=True)
    p.add_argument('-m', '--model', dest='model', required=True)
    p.add_argument('-e', '--epoch', dest='epoch', type=int, required=True)
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-v', '--vector', dest='vector', type=float, nargs=NC, required=True)
    a = p.parse_args()

    load_config(a.config)

    import dataset as ds
    from watts15 import *
    from watts15.dnn import *

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f]

    mean = lu.read_binfile(path.join(STTDIR, 'mean'), dim=ds.AX_DIM)
    stddev = lu.read_binfile(path.join(STTDIR, 'stddev'), dim=ds.AX_DIM)

    outdir = SYNDIR
    for level in [a.model,
                  str(a.epoch),
                  ','.join(['{:.3f}'.format(e) for e in a.vector])]:
        outdir = path.join(outdir, level)
        try:
            mkdir(outdir)
        except FileExistsError:
            pass

    with tf.Session().as_default() as session:
        n1 = SLCV1(mdldir=path.join(MDLDIR, a.model), epoch=a.epoch)
        n2 = SLCV2(nl=NL, nc=NC, w=n1._w, b=n1._b)

        for s in sentences:
            dataset = ds.load_trainset([path.join(SYNDIR, s+'.tfr')])
            example = dataset.make_one_shot_iterator().get_next()
            output = []
            while True:
                try:
                    output.append(n2.predict(session.run(example)[0],
                                             np.array(a.vector)))
                except tf.errors.OutOfRangeError:
                    break
            x = np.concatenate(output)
            x = ax.destandardize(x, mean, stddev)

            w = 1

            v = x[:,-1]
            x = ax.window(x[:,:ax.AX_DIM], np.ones([w]) / float(w))
            x[:,ax.LF0][v <= mean[-1]] = 0.0
            print2(min(x[:,ax.LF0][x[:,ax.LF0] > 0.0]))
            x[:,ax.LF0] = np.log(x[:,ax.LF0])

            x2 = lu.read_binfile(path.join(ACO3DIR, s+'.aco'), dim=ds.AX_DIM)
            x2 = ax.window(x2[:,:ax.AX_DIM], np.ones([w]) / float(w))

            t = [n * 0.005 for n in range(x.shape[0])]

            pyplot.plot(t, [np.exp(y) if y > 0 else 1e-10 for y in x[:,ax.LF0]])
            pyplot.plot(t, [y if y > 0 else 1e-10 for y in x2[:len(t),ax.LF0]])
            pyplot.savefig(path.join(outdir, s+'_f0.pdf'))
            pyplot.close()

            lu.write_binfile(x[:,ax.MAG], path.join(outdir, s+'.mag'))
            lu.write_binfile(x[:,ax.REAL], path.join(outdir, s+'.real'))
            lu.write_binfile(x[:,ax.IMAG], path.join(outdir, s+'.imag'))
            lu.write_binfile(x[:,ax.LF0], path.join(outdir, s+'.lf0'))

            try:
                call('{script} -s {sentence} '
                     '-v {vocdir} -o {outdir} '
                     '-m {magdim} -p {phadim}'.format(
                    script=path.join(SRCDIR, 'synthesize-audio.py'),
                    sentence=s,
                    vocdir=outdir,
                    outdir=outdir,
                    magdim=ax.MAG_DIM,
                    phadim=ax.PHASE_DIM).split())
            except Exception as e:
                print2(e)
                print2('Error synthesizing', s)
                flush2()
