#!/usr/bin/env python
from __init__ import load_config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from os import path, mkdir
import shutil
import numpy as np
from numpy.core.umath_tests import inner1d
from random import Random
from argparse import ArgumentParser


# def memory():
#     import os, psutil
#     pid = os.getpid()
#     py = psutil.Process(pid)
#     use = py.memory_info()[0]/2.0**32
#     print2('memory use: ', use)


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-c', '--config', dest='config', required=True)
    p.add_argument('-m', '--model', dest='model', required=True)
    p.add_argument('-e', '--epoch', dest='epoch', type=int, default=0)
    p.add_argument('-E', '--until', dest='until', type=int, default=0)
    p.add_argument('-p', '--unpacker', dest='unpacker', default=None)
    p.add_argument('--batch', dest='batch', type=int, default=256)
    p.add_argument('--size', dest='size', type=int, default=None)
    p.add_argument('--reg-factor', dest='reg_factor', type=float, default=1e-5)
    p.add_argument('--split', dest='split', type=float, default=0.9)
    p.add_argument('--learning-rate', dest='learning_rate', type=float, default=1e-4)
    p.add_argument('--decay-rate', dest='decay_rate', type=float, default=1.0)
    p.add_argument('--min-learning-rate', dest='min_learning_rate', type=float, default=0)
    p.add_argument('--projection-factor', dest='projection_factor', type=float, default=1.0)
    p.add_argument('--output-factor', dest='output_factor', type=float, default=1.0)
    p.add_argument('--keep-prob', dest='keep_prob', type=float, default=1.0)
    a = p.parse_args()

    if 0 < a.epoch and 0 < a.until and a.until <= a.epoch:
        raise ValueError('until must be greater than epoch')

    load_config(a.config)
    from __init__ import *
    import acoustic as ax
    import dataset as ds
    from model import *
    import util

    mdldir = path.join(MDLSDIR, a.model)
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

    if a.size is None:
        print2('Counting examples')
        size = ds.count_examples([path.join(TFRSDIR, s+'.tfr')
                               for s in sentences])
    else:
        size = a.size
    t_size = int(a.split * size / a.batch)

    with open(debug_log, 'a') as f:
        f.write('size: {}\n'.format(size))
        f.write('split: {}\n'.format(a.split))
        f.write('---------------------------\n')

    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with tf.Graph().as_default() as graph:
        with tf.Session(config=session_config, graph=graph).as_default() as session:
            data = ds.load_synthesizer_dataset(Random(SEED).sample(sentences, len(sentences)))\
                   .batch(a.batch)\
                   .shuffle(buffer_size=100,
                            seed=SEED,
                            reshuffle_each_iteration=False)\
                   .make_initializable_iterator()
            example = data.get_next()
    print2('Dataset created')

    error_log = path.join(mdldir, 'error.txt')

    if a.unpacker is not None:
        u_model, u_epoch = a.unpacker.split('/')
        with tf.Graph().as_default() as u_graph:
            with tf.Session(config=session_config, graph=u_graph).as_default() as u_session:
                unpacker = Unpacker(mdldir=path.join(MDLUDIR, u_model), epoch=int(u_epoch))
                P = u_session.run(unpacker.P)
                print2(P.shape)
    else:
        P = None

    with graph.as_default():
        with session.as_default():
            if a.epoch == 0:
                model = Synthesizer(sentences=sentences, P=P)
                session.run(tf.global_variables_initializer())
                session.run(tf.tables_initializer())
            else:
                model = Synthesizer(mdldir=mdldir, epoch=a.epoch)
                session.run(tf.tables_initializer())
            print2('Model created')

            saver = tf.train.Saver(max_to_keep=0)
            t_summary = tf.summary.FileWriter(path.join(mdldir, 'train'), graph=session.graph)
            v_summary = tf.summary.FileWriter(path.join(mdldir, 'valid'), graph=session.graph)

            epoch = a.epoch+1
            v_loss = None
            while True:
                t_report = util.Report(epoch, mode='t')
                session.run(data.initializer)

                while t_report.iterations < t_size:
                    try:
                        out, loss, summary, step = model.train(
                            *session.run(example),
                            train=True,
                            reg_factor=a.reg_factor,
                            learning_rate=a.learning_rate,
                            decay_rate=a.decay_rate,
                            min_learning_rate=a.min_learning_rate,
                            projection_factor=a.projection_factor,
                            output_factor=a.output_factor,
                            dataset_size=t_size,
                            keep_prob=a.keep_prob
                        )
                        t_report.report(loss)
                        t_summary.add_summary(summary, global_step=step)
                    except tf.errors.OutOfRangeError:
                        break
                print2()


                v_report = util.Report(epoch, mode='v')
                while True:
                    try:
                        out, loss, summary, step = model.train(
                            *session.run(example),
                            train=False
                        )
                        v_report.report(loss)
                        v_summary.add_summary(summary, global_step=step)
                    except tf.errors.OutOfRangeError:
                        break
                print2()

                with open(error_log, 'a') as f:
                    f.write("{},{:.3e},{:.3e}\n"\
                            .format(epoch, t_report.avg_loss, v_report.avg_loss))
                    f.flush()

                model.save(saver, mdldir, epoch)
                epoch += 1

                v_loss = v_report.avg_loss

                if a.until > 0 and epoch > a.until:
                    break
