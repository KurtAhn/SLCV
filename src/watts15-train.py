#!/usr/bin/env python
from __init__ import load_config
from os import path, mkdir
import shutil
import numpy as np
from random import shuffle
from argparse import ArgumentParser


# def memory():
#     import os, psutil
#     pid = os.getpid()
#     py = psutil.Process(pid)
#     use = py.memory_info()[0]/2.0**32
#     print2('memory use: ', use)


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-m', '--model', dest='model', required=True)
    p.add_argument('-e', '--epoch', dest='epoch', type=int, default=0)
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-b', '--nbatch', dest='nbatch', type=int, default=256)
    p.add_argument('-c', '--config', dest='config', required=True)
    a = p.parse_args()

    load_config(a.config)
    from __init__ import *
    import acoustic as ax
    import dataset as ds
    from watts15 import *
    from watts15.dnn import *
    import util

    mdldir = path.join(MDLDIR, a.model)
    try:
        mkdir(mdldir)
    except FileExistsError:
        pass
    shutil.copy2(a.config, mdldir)

    log_path = path.join(mdldir, 'log.txt')
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
                                    log_device_placement=False)
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
            model = SLCV1(mdldir=mdldir, epoch=a.epoch)
            session.run(tf.tables_initializer())

        saver = tf.train.Saver(max_to_keep=0)

        print2('Model created')

        t_data = ds.load_trainset(t_rec)\
                 .shuffle(buffer_size=100000)\
                 .batch(a.nbatch)\
                 .make_initializable_iterator()
        t_example = t_data.get_next()

        v_data = ds.load_trainset(v_rec)\
                 .batch(a.nbatch)\
                 .make_initializable_iterator()
        v_example = v_data.get_next()

        print2('Dataset created')

        epoch = a.epoch+1
        v_loss = None
        while True:
            print2('Training epoch', epoch)
            t_report = util.Report(epoch, mode='t')
            session.run(t_data.initializer)
            while True:
                try:
                    loss = model.train(*session.run(t_example))
                    t_report.report(loss)
                except tf.errors.OutOfRangeError:
                    break
            print2()

            v_report = util.Report(epoch, mode='d')
            session.run(v_data.initializer)
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
                    filename=path.join(mdldir, '_.meta')
                )
            saver.save(session,
                       path.join(mdldir, '_'),
                       global_step=epoch,
                       write_meta_graph=False)
            epoch += 1

            if epoch > 15 and \
               v_loss is not None and \
               v_loss < v_report.avg_loss:
                break
            v_loss = v_report.avg_loss
