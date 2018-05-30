from __init__ import *
from . import *
import tensorflow as tf
import numpy as np


class SLCV1:
    def __init__(self, **kwargs):
        if 'mdldir' in kwargs:
            self._restore(kwargs['mdldir'])
        else:
            self._create(**kwargs)

    def _restore(self, mdldir):
        g = tf.get_default_graph()
        tf.train.import_meta_graph(path.join(mdldir, '_.meta'))\
            .restore(tf.get_default_session(),
                     tf.train.latest_checkpoint(mdldir))
        for name in 'lsypj':
            setattr(self, '_'+name,
                    g.get_tensor_by_name('slcv1/'+name+':0'))
        self._w = []
        self._b = []
        while True:
            try:
                self._w.append(g.get_tensor_by_name(
                    'slcv1/w{}:0'.format(len(self._w))))
                self._b.append(g.get_tensor_by_name(
                    'slcv1/b{}:0'.format(len(self._b))))
            except KeyError:
                break
        self._c = tf.get_collection('c', scope='slcv1')[0]
        self._a = tf.get_collection('a', scope='slcv1')[0]
        self._f = tf.get_collection('f', scope='slcv1')[0]

    def _create(self, **kwargs):
        params = 'nl nc nh na dp rp sentences'.split()
        for f in params:
            if f not in kwargs:
                raise ValueError('Missing argument: {}'.format(f))
        nl, nc, nh, na, dp, rp, sentences = tuple(map(lambda k: kwargs[k],
                                                      params))

        init = lambda shape: tf.truncated_normal(shape, stddev=0.1)

        with tf.name_scope('slcv1') as scope:
            self._t = t = tf.contrib.lookup.index_table_from_tensor(
                mapping=tf.constant(sentences),
                name='t'
            )
            self._l = l = tf.placeholder('float', [None, nl], name='l')
            self._s = s = tf.placeholder('string', [None], name='s')
            self._y = y = tf.placeholder('float', [None, na], name='y')
            self._p = p = tf.Variable(init([len(sentences), nc]), name='p')
            self._c = c = tf.nn.embedding_lookup(p, t.lookup(s), name='c')
            tf.add_to_collection('c', self._c)
            w = [tf.Variable(init([nl+nc, nh]), name='w0')] + \
                [tf.Variable(init([nh, nh]), name='w{}'.format(n))
                 for n in range(1,dp-1)] + \
                [tf.Variable(init([nh, na]), name='w{}'.format(dp-1))]
            self._w = w
            b = [tf.Variable(init([1,nh]), name='b{}'.format(n))
                 for n in range(dp-1)] +\
                [tf.Variable(init([1,na]), name='b{}'.format(dp-1))]
            self._b = b
            h = tf.concat([l, c], axis=1)
            h = tf.nn.tanh(tf.matmul(h, w[0]) + b[0])
            for i in range(1,dp-1):
                h = tf.nn.tanh(tf.matmul(h, w[i]) + b[i])
            a = tf.matmul(h, w[-1])
            self._a = a = tf.add(a, b[-1], name='a')
            tf.add_to_collection('a', self._a)

            j = sum([rp * tf.nn.l2_loss(wi)
                     for wi in w[:-1]],
                    tf.reduce_mean(tf.square(a - y)))
            self._j = j = tf.identity(j, name='j')
            o = tf.train.AdamOptimizer()
            self._f = f = o.minimize(j, name='f')
            tf.add_to_collection('f', self._f)

    def train(self, l, s, y):
        _, j = tf.get_default_session().run(
            [self._f, self._j],
            feed_dict={
                self._l: l,
                self._s: s,
                self._y: y
            }
        )
        return j

    def predict(self, l, s, y):
        j, a = tf.get_default_session().run(
            [self._j, self._a],
            feed_dict={
                self._l: l,
                self._s: s,
                self._y: y
            }
        )
        return j, a

    def embed(self, s):
        c = tf.get_default_session().run(
            self._c,
            feed_dict={
                self._s: [s]
            }
        )
        return c[0]


class SLCV2:
    def __init__(self, **kwargs):
        if 'mdldir' in kwargs:
            self._restore(kwargs['mdldir'])
        else:
            self._create(**kwargs)

    def _restore(self, mdldir):
        g = tf.get_default_graph()
        tf.train.import_meta_graph(path.join(mdldir, '_.meta'))\
            .restore(tf.get_default_session(),
                     tf.train.latest_checkpoint(mdldir))
        for name in 'lcy':
            setattr(self, '_'+name,
                    g.get_tensor_by_name('slcv2/'+name+':0'))
        self._w = []
        self._b = []
        while True:
            try:
                self._w.append(g.get_tensor_by_name(
                    'slcv2/w{}:0'.format(len(self._w))))
                self._b.append(g.get_tensor_by_name(
                    'slcv2/b{}:0'.format(len(self._b))))
            except KeyError:
                break
        self._a = tf.get_collection('a', scope='slcv2')[0]

    def _create(self, **kwargs):
        g = tf.get_default_graph()
        params = 'nc nl w b'.split()
        for p in params:
            if p not in kwargs:
                raise ValueError('Missing argument: {}'.format(p))
        nc, nl, w, b = tuple(map(lambda k: kwargs[k], params))

        with tf.name_scope('slcv2'):
            self._l = l = tf.placeholder('float', [None, nl], name='l')
            self._c = c = tf.placeholder('float', [None, nc], name='c')
            self._w = [tf.assign(tf.Variable(np.zeros(wi.shape, dtype='float32')),
                                 wi, name='w{}'.format(i))
                       for i, wi in enumerate(w)]
            self._b = [tf.assign(tf.Variable(np.zeros(bi.shape, dtype='float32')),
                                 bi, name='b{}'.format(i))
                       for i, bi in enumerate(b)]
            h = tf.concat([l, c], axis=1)
            for wi, bi in zip(w[:-1], b[:-1]):
                h = tf.nn.tanh(tf.matmul(h, wi) + bi)
            a = tf.matmul(h, w[-1])
            self._a = a = tf.add(a, b[-1], name='a')
            tf.add_to_collection('a', self._a)

    def predict(self, l, c):
        return tf.get_default_session().run(
            self._a,
            feed_dict={
                self._l: [l],
                self._c: [c]
            }
        )
