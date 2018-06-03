#!/usr/bin/env python
from __init__ import *
from os import path, mkdir
import numpy as np
from random import shuffle
from argparse import ArgumentParser


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-m', '--model', dest='model', required=True)
    p.add_argument('-e', '--epoch', dest='epoch', type=int, default=0)
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-b', '--nbatch', dest='nbatch', type=int, default=256)
    p.add_argument('-c', '--config', dest='config', required=True)
    a = p.parse_args()

    load_config(c.config)

    import acoustic as ax
    import dataset as ds
    from watts15 import *
    from watts15.dnn import *
    import util

    try:
        mkdir(path.join(MDLDIR, a.model))
    except FileExistsError:
        pass

    log_path = path.join(MDLDIR, a.model, 'log.txt')
    with open(log_path, 'a') as f:
        f.write('----MODEL CONFIGURATION----\n')
        f.write('Control: {}\n'.format(NC))
        f.write('Depth: {}\n'.format(DP))
        f.write('Nodes per layer: {}\n'.format(NH))
        f.write('---------------------------\n')
        f.flush()

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f]

    t_rec = [path.join(TRNDIR, s+'.tfr') for s in sentences]
    shuffle(t_rec)

    v_rec = [path.join(VALDIR, s+'.tfr') for s in sentences]

    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=True)
    session_config.gpu_options.allow_growth = True
    with tf.Session(config=session_config).as_default() as session:
        if a.epoch == 0:
            model = SLCV1(sentences=sentences,
                          nl=NL,
                          nc=NC,
                          nh=NH,
                          na=NA,
                          dp=DP,
                          rp=RP)
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())
        else:
            model = SLCV1(mdldir=path.join(MDLDIR, a.model), epoch=a.epoch)
            #session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())

        saver = tf.train.Saver(max_to_keep=0)

        print2('Model created')

        t_set = ds.load_trainset(t_rec)\
            .shuffle(buffer_size=100000)\
            .batch(a.nbatch)

        v_set = ds.load_trainset(v_rec).batch(a.nbatch)

        print2('Dataset created')

        epoch = a.epoch+1
        v_loss = None
        while True:
            print2('Training epoch', epoch)
            t_report = util.Report(epoch, mode='t')
            t_example = t_set.make_one_shot_iterator().get_next()
            while True:
                try:
                    loss = model.train(*session.run(t_example))
                    t_report.report(loss)
                except tf.errors.OutOfRangeError:
                    break
            print2()

            v_report = util.Report(epoch, mode='d')
            v_example = v_set.make_one_shot_iterator().get_next()
            while True:
                try:
                    loss, out = model.predict(*session.run(v_example))
                    v_report.report(loss)
                except tf.errors.OutOfRangeError:
                    break
            print2()

            with open(log_path, 'a') as f:
                f.write("{},{:.3e},{:.3e}\n"\
                        .format(epoch, t_report.avg_loss, v_report.avg_loss))
                f.flush()

            if epoch == 1:
                tf.train.export_meta_graph(
                    filename=path.join(MDLDIR, a.model, '_.meta')
                )
            saver.save(session,
                       path.join(MDLDIR, a.model, '_'),
                       global_step=epoch,
                       write_meta_graph=False)
            epoch += 1

            if epoch > 15 and \
               v_loss is not None and \
               v_loss < v_report.avg_loss:
                break
            v_loss = v_report.avg_loss
