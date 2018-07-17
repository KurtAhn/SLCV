#!/usr/bin/env python
from __init__ import load_config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from os import path, mkdir
import shutil
from random import Random
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-c', '--config', dest='config', required=True)
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-m', '--model', dest='model', required=True)
    p.add_argument('-e', '--epoch', dest='epoch', type=int, default=0)
    p.add_argument('-E', '--until', dest='until', type=int, default=0)
    p.add_argument('-o', '--oracle', dest='oracle', required=True)
    p.add_argument('--batch', dest='batch', type=int, default=1)
    p.add_argument('--split', dest='split', type=float, default=0.9)
    p.add_argument('--learning-rate', dest='learning_rate', type=float, default=1e-4)
    p.add_argument('--min-learning-rate', dest='min_learning_rate', type=float, default=1e-8)
    p.add_argument('--decay-rate', dest='decay_rate', type=float, default=1.0)
    p.add_argument('--keep-prob', dest='keep_prob', type=float, default=1.0)
    p.add_argument('--valley', dest='valley', action='store_true')
    a = p.parse_args()

    if 0 < a.epoch and 0 < a.until and a.until <= a.epoch:
        raise ValueError('until must be greater than epoch')

    load_config(a.config)
    from __init__ import *
    import acoustic as ax
    import dataset as ds
    from ahn18 import *
    import util

    mdldir = path.join(MDLAEDIR, a.model)
    try:
        mkdir(mdldir)
    except FileExistsError:
        pass
    shutil.copy2(a.config, path.join(mdldir, 'config.json'))

    debug_log = path.join(mdldir, 'debug.txt')
    with open(debug_log, 'a') as f:
        f.write('---------------------------\n')
        f.write('NC: {}\n'.format(NC))
        f.write('NE: {}\n'.format(NE))
        f.write('NF: {}\n'.format(NF))
        f.write('DF: {}\n'.format(DF))
        f.write('NR: {}\n'.format(NR))
        f.write('DR: {}\n'.format(DR))
        f.write('cell type: {}\n'.format('LSTM' if USE_LSTM else 'GRU'))
        f.write('batch: {}\n'.format(a.batch))
        f.write('learning rate: {}\n'.format(a.learning_rate))
        f.write('minimum learning rate: {}\n'.format(a.min_learning_rate))
        f.write('decay rate: {}\n'.format(a.decay_rate))
        f.write('keep prob: {}\n'.format(a.keep_prob))
        f.flush()

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f]

    with open(path.join(ORCWDIR, a.oracle+'.orc'), 'rb') as f:
        oracle = np.load(f)

    data = load_encoder_dataset(sentences, oracle)\
           .padded_batch(a.batch,
                         padded_shapes=([None,NE],[],[NC]))\
           .shuffle(buffer_size=10000,
                    seed=SEED,
                    reshuffle_each_iteration=False)\
           .make_initializable_iterator()
    example = data.get_next()

    with open(debug_log, 'a') as f:
        f.write('size: {}\n'.format(len(sentences)))
        f.write('split: {}\n'.format(a.split))
        f.write('---------------------------\n')
    t_n = int(a.split * len(sentences) / a.batch)
    print2('Dataset created')

    error_log = path.join(mdldir, 'error.txt')

    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with tf.Session(config=session_config).as_default() as session:
        if a.epoch == 0:
            model = Encoder()
            session.run(tf.global_variables_initializer())
        else:
            model = Encoder(mdldir=mdldir, epoch=a.epoch)
        print2('Model created')

        saver = tf.train.Saver(max_to_keep=0)

        epoch = a.epoch+1
        v_loss = None
        while True:
            t_report = util.Report(epoch, mode='t')
            session.run(data.initializer)
            while t_report.iterations < t_n:
                try:
                    x = session.run(example)
                    controls, loss = model.encode(*x,
                                                  train=True,
                                                  learning_rate=a.learning_rate,
                                                  decay_rate=a.decay_rate,
                                                  # epochs=epoch-1,
                                                  keep_prob=a.keep_prob,
                                                  dataset_size=t_n)
                    t_report.report(loss)
                except tf.errors.OutOfRangeError:
                    break
            print2()

            v_report = util.Report(epoch, mode='d')
            while True:
                try:
                    controls, loss = model.encode(*session.run(example),
                                                  train=False)
                    v_report.report(loss)
                except tf.errors.OutOfRangeError:
                    break
            print2()

            with open(error_log, 'a') as f:
                f.write("{},{:.3e},{:.3e}\n"\
                        .format(epoch, t_report.avg_loss, v_report.avg_loss))
                f.flush()

            model.save(saver, mdldir, epoch)
            epoch += 1

            if v_loss is not None and a.valley and v_report.avg_loss >= v_loss:
                break
            if a.until > 0 and epoch > a.until:
                break

            v_loss = v_report.avg_loss
