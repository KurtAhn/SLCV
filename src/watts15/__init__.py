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
NC = cfg_net.get('nc', 2)
NH = cfg_net.get('nh', 64)
NA = ds.AX_DIM
DH = cfg_net.get('dh', 6)
RP = 1e-5
DEVICE = cfg_net.get('device', 'cpu')
DEVICE = '/gpu:0' if DEVICE == 'gpu' else '/cpu:0'


class Trainer(Model):
    def __init__(self, **kwargs):
        Model.__init__(self, 'watts15.Trainer', **kwargs)

    def _create(self, **kwargs):
        try:
            sentences = kwargs['sentences']
        except KeyError:
            raise ValueError('Missing argument: sentences')

        init = lambda shape: tf.truncated_normal(shape, stddev=0.1)

        with tf.device(DEVICE):
            with tf.name_scope(self.name) as scope:
                tf.placeholder('float', [None, NL], name='l')
                tf.placeholder('bool', [], name='b')
                tf.placeholder('string', [None], name='s')
                tf.placeholder('float', [None, NC], name='c')
                tf.placeholder('float', [None, NA], name='a_')
                self._optimizer = tf.train.AdamOptimizer()

                table = tf.contrib.lookup.index_table_from_tensor(
                    mapping=tf.constant(sentences),
                    num_oov_buckets=1,
                    name='T'
                )

                tf.Variable(init([len(sentences)+1, NC]), name='P')
                tf.nn.embedding_lookup(self['P'], table.lookup(self['s']), name='e')

                tf.Variable(init([NL+NC,NH]), name='W0')
                for d in range(1,DH-1):
                    tf.Variable(init([NH,NH]), name='W{}'.format(d))
                tf.Variable(init([NH,NA]), name='W{}'.format(DH-1))

                for d in range(DH-1):
                    tf.Variable(init([1,NH]), name='b{}'.format(d))
                tf.Variable(init([1,NA]), name='b{}'.format(DH-1))

                h = tf.cond(self['b'], lambda: self['e'], lambda: self['c'])
                h = tf.concat([self['l'], h], axis=1)
                for d in range(DH-1):
                    h = tf.nn.tanh(tf.add(tf.matmul(h, self['W{}'.format(d)]),
                                          self['b{}'.format(d)]))
                tf.add(tf.matmul(h, self['W{}'.format(DH-1)]),
                       self['b{}'.format(DH-1)],
                       name='a')

                j = sum([RP * tf.nn.l2_loss(self['W{}'.format(d)])
                         for d in range(DH)],
                        tf.reduce_mean(tf.square(self['a'] - self['a_'])))
                tf.identity(j, name='j')
                self.optimizer.minimize(self['j'], name='o')

    def train(self, linguistics, sentences, targets, train=False):
        session = tf.get_default_session()
        switch = np.ones([linguistics.shape[0]], dtype=bool)
        dummy = np.zeros([linguistics.shape[0], NC], dtype=float)
        if train:
            return session.run(
                [self['a'], self['j'], self['o']],
                feed_dict={
                    self['b']: True,
                    self['l']: linguistics,
                    self['s']: sentences,
                    self['c']: dummy,
                    self['a_']: targets
                }
            )[:2]
        elif targets is not None:
            return session.run(
                [self['a'], self['j']],
                feed_dict={
                    self['b']: True,
                    self['l']: linguistics,
                    self['s']: sentences,
                    self['c']: dummy,
                    self['a_']: targets
                }
            )
        else:
            return session.run(
                [self['a']],
                feed_dict={
                    self['b']: True,
                    self['l']: linguistics,
                    self['s']: sentences,
                    self['c']: dummy
                }
            )

    def synth(self, linguistics, controls):
        return tf.get_default_session().run(
            [self['a']],
            feed_dict={
                self['b']: False,
                self['l']: linguistics,
                self['s']: np.array(['']*linguistics.shape[0]),
                self['c']: controls
            }
        )

    def embed(self, sentences):
        return tf.get_default_session().run(
            [self._c],
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
                        'w': tf.FixedLenFeature([], tf.string),
                        'l': tf.FixedLenFeature([ds.LX_DIM], tf.float32),
                        'a': tf.FixedLenFeature([ds.AX_DIM], tf.float32)
                    }
                )
        )\
        .map(
            lambda feature: (feature['l'], feature['s'], feature['a'])
        )
