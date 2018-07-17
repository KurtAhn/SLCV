from __init__ import *
import dataset as ds
import tensorflow as tf
try:
    TFRecordDataset = tf.data.TFRecordDataset
except AttributeError:
    from tensorflow.contrib.data import TFRecordDataset
import numpy as np
from model import Model


NL = ds.LX_DIM
NC = ds.NC
NH = cfg_syn.get('nh', 64)
NA = ds.AX_DIM
DH = cfg_syn.get('dh', 6)
DEVICE = cfg_syn.get('device', 'cpu')
DEVICE = '/gpu:0' if DEVICE == 'gpu' else '/cpu:0'


class Synthesizer(Model):
    def __init__(self, **kwargs):
        Model.__init__(self, 'Synthesizer', **kwargs)

    def _create(self, **kwargs):
        try:
            sentences = kwargs['sentences']
        except KeyError:
            raise ValueError('Missing argument: sentences')

        winit = tf.contrib.layers.xavier_initializer()
        wfunc = tf.nn.tanh
        pinit = winit

        with tf.device(DEVICE):
            with tf.name_scope(self.name) as scope:
                tf.placeholder('float', [None, NL], name='l')
                tf.placeholder('bool', [], name='b')
                tf.placeholder('string', [None], name='s')
                tf.placeholder('float', [None, NC], name='c')
                tf.placeholder('float', [None, NA], name='a_')

                tf.placeholder('float', name='reg_factor')
                tf.placeholder('float', name='keep_prob')
                self._optimizer = tf.train.AdamOptimizer(
                    tf.train.exponential_decay(
                        tf.placeholder('float', name='learning_rate'),
                        tf.Variable(0, trainable=False, dtype='int32', name='global_step'),
                        tf.placeholder('int32', name='dataset_size'),
                        tf.placeholder('float', name='decay_rate'),
                        False
                    )
                )
                tf.assign(self.global_step, self.global_step + 1)
                tf.placeholder('float', name='projection_factor')

                table = tf.contrib.lookup.index_table_from_tensor(
                    mapping=tf.constant(sentences),
                    num_oov_buckets=1,
                    name='T'
                )
                index = table.lookup(self.s)

                tf.Variable(pinit([len(sentences)+1, NC]), name='P')
                tf.nn.embedding_lookup(self.P, index, name='e')

                tf.Variable(winit([NL+NC,NH]), name='W0')
                for d in range(1,DH-1):
                    tf.Variable(winit([NH,NH]), name='W{}'.format(d))
                tf.Variable(winit([NH,NA]), name='W{}'.format(DH-1))

                for d in range(DH-1):
                    tf.Variable(winit([1,NH]), name='b{}'.format(d))
                tf.Variable(winit([1,NA]), name='b{}'.format(DH-1))

                h = tf.cond(self['b'], lambda: self['e'], lambda: self['c'])
                h = tf.concat([self['l'], h], axis=1)
                for d in range(DH-1):
                    Wd = 'W{}'.format(d)
                    bd = 'b{}'.format(d)
                    h = h @ self[Wd] + self[bd]
                    h = wfunc(h)
                    h = tf.nn.dropout(h, self['keep_prob'])
                tf.add(tf.matmul(h, self['W{}'.format(DH-1)]),
                       self['b{}'.format(DH-1)],
                       name='a')

                j = tf.reduce_mean(tf.abs(self.a - self.a_))
                tf.add(j, tf.add_n([tf.nn.l2_loss(self['W{}'.format(d)])
                                    for d in range(DH)]) * self.reg_factor,
                       name='j')
                # self.optimizer.minimize(self.j, name='o')
                self.optimizer.apply_gradients(
                    [(self.projection_factor * g if v == self.P else g, v)
                     for g, v in self.optimizer.compute_gradients(self.j)],
                    name='o'
                )

    def train(self, linguistics, sentences, targets, train=False, **kwargs):
        session = tf.get_default_session()
        dummy = np.zeros([linguistics.shape[0], NC], dtype=float)
        reg_factor = kwargs.get('reg_factor', 1e-5)
        learning_rate = kwargs.get('learning_rate', 1e-4)
        decay_rate = kwargs.get('decay_rate', 1.0)
        dataset_size = kwargs.get('dataset_size', 0)
        keep_prob = kwargs.get('keep_prob', 1.0)
        projection_factor = kwargs.get('projection_factor', 1.0)

        feed_dict = {
            self.b: True,
            self.l: linguistics,
            self.s: sentences,
            self.c: dummy
        }
        if train:
            return session.run(
                [self.a, self.j, self.o],
                feed_dict={
                    **feed_dict,
                    self.a_: targets,
                    self.reg_factor: reg_factor,
                    self.learning_rate: learning_rate,
                    self.decay_rate: decay_rate,
                    self.dataset_size: dataset_size,
                    self.keep_prob: keep_prob,
                    self.projection_factor: projection_factor
                }
            )[:2]
        elif targets is not None:
            return session.run(
                [self.a, self.j],
                feed_dict={
                    **feed_dict,
                    self.a_: targets,
                    self.reg_factor: reg_factor,
                    self.keep_prob: 1.0
                }
            )
        else:
            return session.run(
                [self.a],
                feed_dict={
                    **feed_dict,
                    self.keep_prob: 1.0
                }
            )

    def synth(self, linguistics, controls):
        return tf.get_default_session().run(
            [self.a],
            feed_dict={
                self.b: False,
                self.l: linguistics,
                self.s: np.array(['']*linguistics.shape[0]),
                self.c: controls,
                self.keep_prob: 1.0
            }
        )

    def embed(self, sentences):
        return tf.get_default_session().run(
            [self['e']],
            feed_dict={
                self['s']: sentences
            }
        )

    @property
    def optimizer(self):
        return self._optimizer


def load_dataset(sentences):
    return TFRecordDataset([path.join(DSATDIR, sentence+'.tfr')
                            for sentence in sentences])\
        .map(
            lambda record: \
                tf.parse_single_example(
                    record,
                    features={
                        's': tf.FixedLenFeature([], tf.string),
                        'l': tf.FixedLenFeature([ds.LX_DIM], tf.float32),
                        'a': tf.FixedLenFeature([ds.AX_DIM], tf.float32)
                    }
                )
        )\
        .map(
            lambda feature: (feature['l'], feature['s'], feature['a'])
        )
