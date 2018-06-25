#!/usr/bin/env python
from __init__ import load_config
from os import path, mkdir
import shutil
import numpy as np
from random import shuffle
import tensorflow as tf
from argparse import ArgumentParser


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-c', '--config', dest='config', required=True)
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-m', '--model', dest='model', required=True)
    p.add_argument('-e', '--epoch', dest='epoch', type=int, default=0)
    p.add_argument('-o', '--oracle', dest='oracle', required=True)
    p.add_argument('-b', '--nbatch', dest='nbatch', type=int, default=256)
    p.add_argument('-n', '--ndata', dest='ndata', type=int, default=None)
    p.add_argument('-x', '--split', dest='split', type=float, default=0.9)
    a = p.parse_args()

    load_config(a.config)
    from __init__ import *
    import acoustic as ax
    import dataset as ds
    from ahn18 import *
    import util

    mdldir = path.join(MDLASDIR, a.model)
    try:
        mkdir(mdldir)
    except FileExistsError:
        pass
    shutil.copy2(a.config, mdldir)

    log_path = path.join(mdldir, 'log.txt')
    with open(log_path, 'a') as f:
        f.write('----MODEL CONFIGURATION----\n')
        f.write('Control: {}\n'.format(NC))
        f.write('Acoustic model depth: {}\n'.format(DH))
        f.write('Acoustic nodes per layer: {}\n'.format(NH))
        f.write('Language model depth: {}\n'.format(DE))
        f.write('Language nodes per layer: {}\n'.format(NE))
        f.write('---------------------------\n')
        f.flush()

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f]

    if a.ndata is None:
        print2('Counting examples')
        n = ds.count_examples([path.join(DSATDIR, s+'.tfr')
                               for s in sentences])
    else:
        n = a.ndata
    n_t = int(a.split * n / a.nbatch)

    data = load_synthesizer_dataset(sentences, a.oracle)\
           .batch(a.nbatch)\
           .shuffle(buffer_size=1000,
                    reshuffle_each_iteration=False,
                    seed=SEED)\
           .make_initializable_iterator()
    example = data.get_next()

    with open(log_path, 'a') as f:
        f.write('----------DATASET----------\n')
        f.write('Size: {}\n'.format(n))
        f.write('Split: {}\n'.format(a.split))
        f.write('---------------------------\n')

    print2('Dataset created')

    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with tf.Session(config=session_config).as_default() as session:
        if a.epoch == 0:
            model = Synthesizer()
            session.run(tf.global_variables_initializer())
        else:
            model = Synthesizer(mdldir=mdldir, epoch=a.epoch)
        print2('Model created')

        saver = tf.train.Saver(max_to_keep=0)

        epoch = a.epoch+1
        v_loss = None
        while True:
            t_report = util.Report(epoch, mode='t')
            session.run(data.initializer)

            while t_report.iterations < n_t:
                try:
                    acoustics, loss = model.predict(*session.run(example),
                                                    train=True)
                    t_report.report(loss)
                except tf.errors.OutOfRangeError:
                    break
            print2()

            v_report = util.Report(epoch, mode='d')
            while True:
                try:
                    acoustics, loss = model.predict(*session.run(example),
                                                    train=False)
                    v_report.report(loss)
                except tf.errors.OutOfRangeError:
                    break
            print2()

            with open(log_path, 'a') as f:
                f.write("{},{:.3e},{:.3e}\n"\
                        .format(epoch, t_report.avg_loss, v_report.avg_loss))
                f.flush()

            model.save(saver, mdldir, epoch)
            epoch += 1

            if epoch > 30:
                break

            v_loss = v_report.avg_loss
