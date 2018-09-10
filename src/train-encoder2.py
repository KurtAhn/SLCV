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
    p.add_argument('-p', '--synthesizer', dest='synthesizer', required=True)
    p.add_argument('-v', '--vocab', dest='vocab', required=True)
    p.add_argument('--batch', dest='batch', type=int, default=1)
    p.add_argument('--split', dest='split', type=float, default=0.9)
    p.add_argument('--reg-factor', dest='reg_factor', type=float, default=1e-5)
    p.add_argument('--learning-rate', dest='learning_rate', type=float, default=1e-4)
    p.add_argument('--min-learning-rate', dest='min_learning_rate', type=float, default=0)
    p.add_argument('--decay-rate', dest='decay_rate', type=float, default=1.0)
    p.add_argument('--clip-threshold', dest='clip_threshold', type=float, default=5.0)
    p.add_argument('--keep-prob', dest='keep_prob', type=float, default=1.0)
    p.add_argument('--valley', dest='valley', action='store_true')
    a = p.parse_args()

    if 0 < a.epoch and 0 < a.until and a.until <= a.epoch:
        raise ValueError('until must be greater than epoch')

    load_config(a.config)
    from __init__ import *
    import acoustic as ax
    import dataset as ds
    from model import *
    import util

    mdldir = path.join(MDLEDIR, a.model)
    try:
        mkdir(mdldir)
    except FileExistsError:
        pass
    shutil.copy2(a.config, path.join(mdldir, 'config.json'))

    debug_log = path.join(mdldir, 'debug.txt')
    with open(debug_log, 'a') as f:
        f.write('---------------------------\n')
        for k, v in vars(a).items():
            f.write('{}: {}\n'.format(k, v))
        f.flush()

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f]

    with open(path.join(EMBDIR, a.vocab+'.vcb')) as f:
        vocab = [l.rstrip() for l in f]

    with open(path.join(EMBDIR, a.vocab+'.dim')) as f:
        embed_shape = tuple(int(t) for t in next(f).rstrip().split())
    embed = np.memmap(path.join(EMBDIR, a.vocab+'.emb'),
                      mode='r', dtype='float', shape=embed_shape)

    with open(debug_log, 'a') as f:
        f.write('size: {}\n'.format(len(sentences)))
        f.write('split: {}\n'.format(a.split))
        f.write('---------------------------\n')
    t_size = int(a.split * len(sentences) / a.batch)

    error_log = path.join(mdldir, 'error.txt')

    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    session_config.gpu_options.allow_growth = True

    s_model, s_epoch = a.synthesizer.split('/')
    with tf.Graph().as_default() as s_graph:
        with tf.Session(config=session_config,
                        graph=s_graph) as s_session:
            synthesizer = Synthesizer(mdldir=path.join(MDLSDIR, s_model), epoch=int(s_epoch))
            s_session.run(tf.tables_initializer())
            P, = synthesizer.embed(sentences)

    with tf.Graph().as_default() as graph:
        with tf.Session(config=session_config,
                        graph=graph).as_default() as session:
            if a.epoch == 0:
                model = Encoder(vocab=vocab,
                                embed=embed,
                                mean=np.mean(P, axis=0),
                                stddev=np.std(P, axis=0))
                session.run(tf.global_variables_initializer())
                session.run(tf.tables_initializer())
            else:
                model = Encoder(mdldir=mdldir, epoch=a.epoch)
                session.run(tf.tables_initializer())
            print2('Model created')

            data = ds.load_encoder_dataset2(sentences, P)\
                   .padded_batch(a.batch,
                                 padded_shapes=([None],[],[model.nc]))\
                   .shuffle(buffer_size=10000,
                            seed=SEED,
                            reshuffle_each_iteration=False)\
                   .make_initializable_iterator()
            example = data.get_next()
            print2('Dataset created')

            saver = tf.train.Saver(max_to_keep=0)
            t_summary = tf.summary.FileWriter(path.join(mdldir, 'train'), graph=session.graph)
            v_summary = tf.summary.FileWriter(path.join(mdldir, 'valid'), graph=session.graph)

            repeat = 10
            epoch = a.epoch+1
            v_loss = None
            while True:
                t_report = util.Report(epoch, mode='t')
                session.run(data.initializer)
                while t_report.iterations < t_size * repeat:
                    try:
                        x = session.run(example)
                        # print2(*x)
                        for n in range(repeat):
                            controls, loss, summary, step = model.encode(
                                *x[:-1],
                                x[-1] + np.random.normal(0, 0.01, x[-1].shape),
                                train=True,
                                reg_factor=a.reg_factor,
                                learning_rate=a.learning_rate,
                                decay_rate=a.decay_rate,
                                clip_threshold=a.clip_threshold,
                                keep_prob=a.keep_prob,
                                dataset_size=t_size * repeat
                            )
                            # print2(controls, loss, summary, step)
                            # quit()
                            t_report.report(loss)
                            t_summary.add_summary(summary, step)
                    except tf.errors.OutOfRangeError:
                        break
                print2()

                v_report = util.Report(epoch, mode='d')
                while True:
                    try:
                        x = session.run(example)
                        controls, loss, summary, step = model.encode(
                            *x,
                            train=False)
                        v_report.report(loss)
                        v_summary.add_summary(summary, step)
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
