#!/usr/bin/env python
from __init__ import load_config
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.environ['MERLIN'])
from utils.compute_distortion import IndividualDistortionComp
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
    p.add_argument('-m', '--synthesizer', dest='synthesizer', required=True)
    p.add_argument('-e', '--synthesizer_epoch', dest='synthesizer_epoch', type=int, required=True)
    p.add_argument('-v', '--vector', dest='vector', type=float, nargs='+', required=True)
    p.add_argument('-M', dest='encoder', default=None)
    p.add_argument('-E', dest='encoder_epoch', type=int, default=None)
    p.add_argument('-p', '--plot-f0', dest='plot_f0', action='store_true')
    a = p.parse_args()

    load_config(a.config)
    from __init__ import *
    import acoustic as ax
    import dataset as ds
    from ahn18 import *

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f]

    mean = lu.read_binfile(path.join(STTDIR, 'mean'), dim=ds.AX_DIM)
    stddev = lu.read_binfile(path.join(STTDIR, 'stddev'), dim=ds.AX_DIM)

    outdir = SYNADIR
    for level in [a.synthesizer,
                  str(a.synthesizer_epoch),
                  '{}({})'.format(
                      '{}-{}+'.format(a.encoder, a.encoder_epoch) \
                      if a.encoder is not None else '',
                      ','.join(['{:.3f}'.format(e) for e in a.vector])
                  )]:
        outdir = path.join(outdir, level)
        try:
            mkdir(outdir)
        except FileExistsError:
            pass

    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    session_config.gpu_options.allow_growth = True

    if a.encoder is not None:
        with tf.Graph().as_default() as e_graph:
            e_example = load_encoder_dataset(sentences)\
                        .padded_batch(1, padded_shapes=([None,NE],[]))\
                        .make_one_shot_iterator()\
                        .get_next()
            with tf.Session(graph=e_graph, config=session_config).as_default() as e_session:
                encoder = Encoder(mdldir=path.join(MDLAEDIR, a.encoder),
                                  epoch=a.encoder_epoch)
        print2('Encoder loaded')

    with tf.Graph().as_default() as s_graph:
        with tf.Session(graph=s_graph, config=session_config).as_default() as s_session:
            synthesizer = Synthesizer(mdldir=path.join(MDLASDIR, a.synthesizer),
                                      epoch=a.synthesizer_epoch)
    print2('Synthesizer loaded')

    for sentence in sentences:
        if a.encoder is not None:
            with e_graph.as_default():
                with e_session.as_default():
                    vector = np.array(a.vector).reshape(1, -1) + \
                             encoder.encode(*e_session.run(e_example),
                                            targets=None,
                                            train=False)[0].reshape(1, -1)
                    print2(vector)
        else:
            vector = np.array(a.vector).reshape(1,-1)

        with s_graph.as_default():
            s_example = load_synthesizer_dataset([sentence])\
                        .make_one_shot_iterator()\
                        .get_next()

        outs = []
        while True:
            try:
                with s_graph.as_default():
                    with s_session.as_default():
                        out, = synthesizer.synth(s_session.run(s_example)[0].reshape(1,-1),
                                                 vector,
                                                 None)
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
        #     pyplot.plot(t, np.exp(x[:,ax.LF0]))
        #     x2 = lu.read_binfile(path.join(ACO3DIR, sentence+'.aco'), dim=ds.AX_DIM)
        #     x2[:,ax.LF0][x2[:,-1] == 0.0] = 0.0
        #     pyplot.plot(t, [y if y > 0 else 0.0 for y in x2[:len(t),ax.LF0]])
        #     pyplot.savefig(path.join(outdir, sentence+'_f0.pdf'))
        #     pyplot.close()

        lu.write_binfile(x[:,ax.MAG], path.join(outdir, sentence+'.mag'))
        lu.write_binfile(x[:,ax.REAL], path.join(outdir, sentence+'.real'))
        lu.write_binfile(x[:,ax.IMAG], path.join(outdir, sentence+'.imag'))
        lu.write_binfile(x[:,ax.LF0], path.join(outdir, sentence+'.lf0'))

        try:
            call('{script} -s {sentence} -o {outdir} -m {magdim} -p {phadim}'\
                .format(script=path.join(SRCDIR, 'synthesize-audio.py'),
                        sentence=sentence,
                        #vocdir=outdir,
                        outdir=outdir,
                        magdim=ax.MAG_DIM,
                        phadim=ax.PHASE_DIM).split())
        except Exception as e:
            print2(e)
            print2('Error synthesizing', sentence)
            flush2()
