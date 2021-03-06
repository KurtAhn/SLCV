#!/usr/bin/env python
from __init__ import load_config
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.environ['MAGPHASE'])
import libutils as lu
from subprocess import call
from os import path, mkdir
import matplotlib.pyplot as pyplot
import tensorflow as tf
import numpy as np
np.warnings.filterwarnings('ignore')
from argparse import ArgumentParser


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-c', '--config', dest='config', required=True)
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-1', '--synthesizer', dest='synthesizer', required=True)
    p.add_argument('-v', '--vector', dest='vector', type=float, nargs='+', required=True)
    p.add_argument('-o', '--oracle', dest='oracle', action='store_true')
    p.add_argument('-d', '--directory', dest='directory', required=True)
    # p.add_argument('-p', '--plot-f0', dest='plot_f0', action='store_true')
    a = p.parse_args()

    load_config(a.config)
    from __init__ import *
    import acoustic as ax
    import dataset as ds
    from model import *

    if len(a.vector) != ds.NC:
        raise ValueError('Vector dimension must be {}'.format(ds.NC))

    model, epoch = a.synthesizer.split('/')

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f]

    mean = lu.read_binfile(path.join(ACOSDIR, 'mean'), dim=ds.NA)
    stddev = lu.read_binfile(path.join(ACOSDIR, 'stddev'), dim=ds.NA)

    outdir = SYNDIR
    for level in [model,
                  epoch,
                  a.directory]:
        outdir = path.join(outdir, level)
        try:
            mkdir(outdir)
        except FileExistsError:
            pass

    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with tf.Session(config=session_config).as_default() as session:
        model = Synthesizer(mdldir=path.join(MDLSDIR, model), epoch=int(epoch))
        session.run(tf.tables_initializer())

        for sentence in sentences:
            if a.oracle:
                vector = np.array(a.vector).reshape(1,-1) + \
                         model.embed([sentence])[0].reshape(1,-1)
            else:
                vector = np.array(a.vector).reshape(1,-1)
            # print2(vector)

            data = ds.load_synthesizer_dataset([sentence])\
                   .make_one_shot_iterator()
            example = data.get_next()

            outs = []
            while True:
                try:
                    x = session.run(example)[0].reshape(1,-1)
                    # print1(x[0,557:600])
                    out, = model.synth(x.reshape(1,-1),
                                       vector)
                    outs.append(out)
                except tf.errors.OutOfRangeError:
                    break

            x = np.concatenate(outs)
            v = x[:,-1]
            f = x[:,ax.LF0]
            x = x[:,:ax.AX_DIM-1]

            # u = np.mean(x, axis=0)
            # si = np.reciprocal(np.std(x, axis=0))
            # x = np.add(np.multiply(np.subtract(x, u), si), u)
            x = ax.destandardize(np.concatenate([x, f], axis=1),
                                 mean[:ax.AX_DIM], stddev[:ax.AX_DIM])
            # x[:,ax.LF0][x[:,ax.LF0] < 50.0] = 50.0
            if ds.EXP_F0:
                x[:,ax.LF0][v <= mean[-1]] = 0.0
                x[:,ax.LF0] = np.log(x[:,ax.LF0])
            else:
                x[:,ax.LF0][v <= mean[-1]] = float('-Inf')

            # if a.plot_f0:
            #     t = [n * 0.005 for n in range(x.shape[0])]
            #     x2 = lu.read_binfile(path.join(ACO3DIR, sentence+'.aco'), dim=ds.AX_DIM)
            #     x2[:,ax.LF0][x2[:,-1] == 0.0] = 0.0
            #     pyplot.plot(t, x2[:len(t),1])
            #     pyplot.plot(t, x[:,1])#np.exp(x[:,ax.LF0]))
            #     # pyplot.plot(t, [y if y > 0 else 0.0 for y in x2[:len(t),0]])
            #     pyplot.savefig(path.join(outdir, sentence+'_mag.pdf'))
            #     pyplot.close()

            lu.write_binfile(x[:,ax.MAG], path.join(outdir, sentence+'.mag'))
            lu.write_binfile(x[:,ax.REAL], path.join(outdir, sentence+'.real'))
            lu.write_binfile(x[:,ax.IMAG], path.join(outdir, sentence+'.imag'))
            lu.write_binfile(x[:,ax.LF0], path.join(outdir, sentence+'.lf0'))

            try:
                call('{script} -s {sentence} -o {outdir} -m {magdim} -p {phadim} -f merlin {const_rate}'\
                    .format(script=path.join(SRCDIR, 'wavify.py'),
                            sentence=sentence,
                            #vocdir=outdir,
                            outdir=outdir,
                            magdim=ax.MAG_DIM,
                            phadim=ax.PHASE_DIM,
                            const_rate='' if ds.CONST_RATE else '-v').split())
            except Exception as e:
                print2(e)
                print2('Error synthesizing', sentence)
                flush2()
