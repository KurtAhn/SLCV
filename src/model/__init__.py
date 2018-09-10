from __init__ import *
import dataset as ds
import tensorflow as tf
try:
    TFRecordDataset = tf.data.TFRecordDataset
except AttributeError:
    from tensorflow.contrib.data import TFRecordDataset
from os import path
import numpy as np


class Model:
    """
    Base class for Tensorflow networks.
    """
    def __init__(self, name, **kwargs):
        """
        Create a new model (no kwargs) or restore an existing model.

        name: Name of the subclass used for scoping
        **kwargs
            mdldir: Directory of an existing model description
            epoch: Subindex of the precise model version representing the
                number of epochs it was trained for
        """
        self._name = name
        if not kwargs:
            self._create(**kwargs)
        else:
            try:
                mdldir = kwargs.pop('mdldir')
                epoch = kwargs.pop('epoch', 0)
                self._restore(mdldir, epoch)
            except KeyError:
                raise ValueError('mdldir argument not provided')

    def _restore(self, mdldir, epoch):
        """
        Restore an existing model.

        mdldir: Directory with the description of the model to load;
            assumed to be a subdirectory of MDLDIR
        epoch: Number of epochs the model to load was trained for
        """
        g = tf.get_default_graph()
        meta = tf.train.import_meta_graph(path.join(mdldir, '_.meta'),
                                          clear_devices=True)
        if epoch:
            with open(path.join(mdldir, 'checkpoint'), 'w') as f:
                f.write('model_checkpoint_path: "' +
                        path.join(mdldir, '_-{}'.format(epoch)) + '"')
        meta.restore(tf.get_default_session(),
                     tf.train.latest_checkpoint(mdldir))

    def save(self, saver, mdldir, epoch):
        """
        Save a model to disk.

        saver: tf.Saver object
        mdldir: Directory to save the model in; assumed to be a
            subdirectory of MDLDIR
        epoch: Number of epochs the model was trained for (used to distinguish
            model descriptions under the same directory)
        """
        if epoch == 1:
            tf.train.export_meta_graph(
                filename=path.join(mdldir, '_.meta')
            )
        saver.save(tf.get_default_session(),
                   path.join(mdldir, '_'),
                   global_step=epoch,
                   write_meta_graph=False)

    @property
    def name(self):
        """
        Model name used for scoping.
        """
        return self._name

    @property
    def nl(self):
        """
        Linguistic feature dimension.
        """
        return ds.NL

    @property
    def nc(self):
        """
        Control vector dimension
        """
        return ds.NC

    def __getitem__(self, k):
        """
        Convenient method for accessing tensors and operations scoped by
        self.name.
        """
        g = tf.get_default_graph()
        try:
            return g.get_tensor_by_name('{}/{}:0'.format(self.name, k))
        except KeyError:
            try:
                return g.get_operation_by_name('{}/{}'.format(self.name, k))
            except KeyError:
                raise KeyError('Nonexistent name: {}/{}'.format(self.name, k))

    def __getattr__(self, a):
        """
        Convenient method for accessing tensors and operations scoped by
        self.name.
        """
        return self[a]


class Synthesizer(Model):
    """
    Corresponds to the acoustic model in the paper.

    At training, the model learns an embedding matrix that maps the ID of
    each training utterance to a low-dimensional vector, which helps explain
    variance in acoustic space not captured by linguistic features alone.
    (Linguistic features, Sentence ID) -> Acoustic features

    At synthesis, the embedding matrix is removed, and in place of its output,
    an arbitrary low-dimensional vector specified by the user is supplied.
    (Linguistic features, Control vector) -> Acoustic features
    """
    def __init__(self, **kwargs):
        Model.__init__(self, 'Synthesizer', **kwargs)

    def _create(self, **kwargs):
        """
        **kwargs
            sentences: IDs of all training sentences
            P: Initial value for the projection layer (optional)
        """
        try:
            sentences = kwargs['sentences']
        except KeyError:
            raise ValueError('Missing argument: sentences')
        P = kwargs.get('P', None)

        # Tanh activation for hidden layers (W's)
        wfunc = tf.nn.tanh
        winit = tf.contrib.layers.xavier_initializer()
        binit = tf.zeros_initializer()
        pinit = winit

        with tf.device(self.device):
            with tf.name_scope(self.name) as scope:
                # Linguistic features
                tf.placeholder('float', [None, self.nl], name='l')
                # Sentence ID (for training only)
                tf.placeholder('string', [None], name='s')
                # Control vector input (for synthesis only)
                tf.placeholder('float', [None, self.nc], name='c')
                # Acoustic feature target
                tf.placeholder('float', [None, self.na], name='a_')
                # True if training
                tf.placeholder('bool', name='t_mode')
                # True if validating
                tf.placeholder('bool', name='v_mode')
                # True if synthesizing
                tf.placeholder('bool', name='s_mode')
                # Regularization penalty
                tf.placeholder('float', name='reg_factor')
                # Dropout retain probability applied to each layer (not used in paper)
                tf.placeholder('float', name='keep_prob')
                # Initial learning rate before amplification
                tf.placeholder('float', name='learning_rate')
                # Number of examples in the training set
                tf.placeholder('int32', name='dataset_size')
                # Per epoch decay rate of the learning rate
                tf.placeholder('float', name='decay_rate')
                # Floor for the learning rate (not used in paper)
                tf.placeholder('float', name='min_learning_rate')
                # Scale learning rate for the projection layer (P)
                tf.placeholder('float', name='projection_factor')
                # Scale learning rate for the output layer (W[-1])
                tf.placeholder('float', name='output_factor')

                # Training example counter needed for learning rate decay
                tf.Variable(0, trainable=False, dtype='int32', name='t_step_0')
                tf.assign(self.t_step_0,
                          self.t_step_0 + tf.cast(self.t_mode, 'int32'), name='t_step')

                # Validation example counter for... stuff
                tf.Variable(0, trainable=False, dtype='int32', name='v_step_0')
                tf.assign(self.v_step_0,
                          self.v_step_0 + tf.cast(self.v_mode, 'int32'), name='v_step')

                tf.maximum(
                    tf.train.exponential_decay(
                        self.learning_rate,
                        self.t_step,
                        self.dataset_size,
                        self.decay_rate,
                        False
                    ),
                    self.min_learning_rate,
                    name='adjusted_learning_rate'
                )

                self._optimizer1 = tf.train.AdamOptimizer(
                    self.adjusted_learning_rate
                )

                self._optimizer2 = tf.train.AdamOptimizer(
                    self.adjusted_learning_rate * self.projection_factor
                )

                self._optimizer3 = tf.train.AdamOptimizer(
                    self.adjusted_learning_rate * self.output_factor
                )

                # Map sentence ID to index; final index reserved for unknown sentences
                table = tf.contrib.lookup.index_table_from_tensor(
                    mapping=tf.constant(sentences),
                    num_oov_buckets=1,
                    name='T'
                )
                index = table.lookup(self.s)

                # Projection layer can be imported from another model
                # In the paper, only Synthesizer-to-Unpacker transfer is discussed,
                # but the code allows for the vice versa case.
                if P is None:
                    tf.Variable(pinit([len(sentences)+1, self.nc]), name='P')
                else:
                    tf.Variable(P, name='P')
                tf.nn.embedding_lookup(self.P, index, name='e')

                # If synthesizing, skip embedding layer and use control vector directly.
                h = tf.cond(self.s_mode, lambda: self.c, lambda: self.e)
                h = tf.concat([self.l, h], axis=1)

                # Creating fully connected hidden layers with dropout.
                for d in range(self.dh):
                    Wd = 'W{}'.format(d)
                    bd = 'b{}'.format(d)

                    if d == self.dh - 1:
                        tf.Variable(winit([self.nh,self.na]), name=Wd)
                        tf.Variable(binit([1,self.na]), name=bd)
                        h = tf.add(h @ self[Wd], self[bd], name='a')
                    else:
                        if d == 0:
                            tf.Variable(winit([self.nl+self.nc,self.nh]), name=Wd)
                        else:
                            tf.Variable(winit([self.nh,self.nh]), name=Wd)
                        tf.Variable(binit([1,self.nh]), name=bd)
                        h = tf.nn.dropout(wfunc(h @ self[Wd] + self[bd]), self.keep_prob)

                # Mean absolute error objective with L2 regularization penalty
                # applied to all non-bias parameters (W's).
                j = tf.reduce_mean(tf.abs(self.a - self.a_))
                tf.add(j, tf.add_n([tf.nn.l2_loss(self['W{}'.format(d)])
                                    for d in range(self.dh)]) * self.reg_factor,
                       name='j')

                # Fancy juggling for applying different learning rates to different layers.
                variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
                v2 = [self.P.op.name]
                v3 = [self['W{}'.format(self.dh-1)].op.name,
                      self['b{}'.format(self.dh-1)].op.name]
                self.optimizer1.minimize(
                    j,
                    var_list=[
                        v
                        for v in variables
                        if v.op.name not in v2 + v3
                    ],
                    name='o1')
                self.optimizer2.minimize(
                    j,
                    var_list=[
                        v
                        for v in variables
                        if v.op.name in v2
                    ],
                    name='o2')
                self.optimizer3.minimize(
                    j,
                    var_list=[
                        v
                        for v in variables
                        if v.op.name in v3
                    ],
                    name='o3'
                )
                tf.group(self.o1, self.o2, self.o3, name='o')

                # So that we can see cool stuff with Tensorboard.
                # Oh, it all makes sense now:
                # Tensor = Water, Tensorflow -> Waterflow; Tensorboard -> Waterboard
                tf.summary.scalar('summary', self.j)

    def train(self, linguistics, sentences, targets, train=False, **kwargs):
        """
        Train (not really lol) the model with a batch.
        The only real difference with self.synth is that setence ID is used
        instead of control vector.
        Three modes of operations supported:
            - Training: provide all positional arguments and train=True
            - Validating: provide all positional arguments and train=False
            - Observing: set targets=None and train=False

        linguistics: Linguistic features
        sentences: Sentence IDs
        targets: Target acoustic parameters
        train: True if training False if validating
        **kwargs
            reg_factor: regularization penalty
            learning_rate: initial learning rate
            decay_rate: per-epoch learning rate decay
            min_learning_rate: learning rate floor
            dataset_size: size of the training set
            keep_prob: dropout retain probability
            projection_factor: scale the learning rate for the projection layer
            output_factor: scale the learning rate for the output layer
        """
        session = tf.get_default_session()
        # Something to plug the control vector input,
        # which is used only for synthesis.
        dummy = np.zeros([linguistics.shape[0], self.nc], dtype=float)
        reg_factor = kwargs.get('reg_factor', 0)
        learning_rate = kwargs.get('learning_rate', 1e-4)
        decay_rate = kwargs.get('decay_rate', 1.0)
        min_learning_rate = kwargs.get('min_learning_rate', 0)
        dataset_size = kwargs.get('dataset_size', 0)
        keep_prob = kwargs.get('keep_prob', 1.0)
        projection_factor = kwargs.get('projection_factor', 1.0)
        output_factor = kwargs.get('output_factor', 1.0)

        feed_dict = {
            self.l: linguistics,
            self.s: sentences,
            self.c: dummy,
            self.s_mode: False
        }
        if train:
            return session.run(
                [self.a, self.j, self.summary, self.t_step, self.o],
                feed_dict={
                    **feed_dict,
                    self.a_: targets,
                    self.reg_factor: reg_factor,
                    self.learning_rate: learning_rate,
                    self.decay_rate: decay_rate,
                    self.dataset_size: dataset_size,
                    self.min_learning_rate: min_learning_rate,
                    self.keep_prob: keep_prob,
                    self.projection_factor: projection_factor,
                    self.output_factor: output_factor,
                    self.t_mode: True,
                    self.v_mode: False
                }
            )[:4]
        elif targets is not None:
            # Validating
            return session.run(
                [self.a, self.j, self.summary, self.v_step],
                feed_dict={
                    **feed_dict,
                    self.a_: targets,
                    self.reg_factor: reg_factor,
                    self.keep_prob: 1.0,
                    self.t_mode: False,
                    self.v_mode: True
                }
            )
        else:
            # Not training or validating; just want to see the output
            return session.run(
                [self.a],
                feed_dict={
                    **feed_dict,
                    self.keep_prob: 1.0,
                    self.t_mode: False,
                    self.v_mode: False
                }
            )

    def synth(self, linguistics, controls):
        """
        Synthesize acoustic parameters using control vectors.
        As with self.train, arguments are in batches of frames.
        This method doesn't put the sort of globality restriction assumed
        in the paper.
        In other words, two frames from the same sentence don't need to be paired
        with the same control vector.

        linguistics: Linguistic features for frames
        controls: Arbitrary control vectors
        """
        return tf.get_default_session().run(
            [self.a],
            feed_dict={
                self.l: linguistics,
                self.s: np.array(['']*linguistics.shape[0]),
                self.c: controls,
                self.keep_prob: 1.0,
                self.s_mode: True,
                self.t_mode: False,
                self.v_mode: False
            }
        )

    def embed(self, sentences):
        """
        Find the embeddings for sentences using the projection layer.

        sentences: Array of sentence IDs
        """
        return tf.get_default_session().run(
            [self.e],
            feed_dict={
                self.s: sentences
            }
        )

    @property
    def optimizer1(self):
        return self._optimizer1

    @property
    def optimizer2(self):
        return self._optimizer2

    @property
    def optimizer3(self):
        return self._optimizer3

    @property
    def device(self):
        """
        Device type (CPU or GPU) to use for the model.
        """
        return '/'+cfg_syn.get('device', 'cpu')+':0'

    @property
    def nl(self):
        """
        Linguistic feature dimensions.
        """
        return ds.NL + 9

    @property
    def nh(self):
        """
        Number of nodes per hidden layer.
        """
        return cfg_syn.get('nh', 256)

    @property
    def dh(self):
        """
        Number of hidden layers.
        """
        return cfg_syn.get('dh', 6)

    @property
    def na(self):
        """
        Acoustic vector dimension.
        """
        return ds.NA


class Encoder(Model):
    def __init__(self, **kwargs):
        Model.__init__(self, 'Encoder', **kwargs)

    def _create(self, **kwargs):
        """
        **kwargs
            vocab: All legal words
            embed: Embeddings for all words in vocab, following the same order
                as vocab.
            mean: Training set style mean
            stddev: Training set style standard deviation
        """
        vocab = kwargs['vocab'] + ['']
        embed = np.concatenate([kwargs['embed'], np.zeros([1, self.ne])], axis=0)
        mean = kwargs['mean']
        stddev = kwargs['stddev']

        # Tanh activation for hidden layers (W's)
        wfunc = tf.nn.tanh
        winit = tf.contrib.layers.xavier_initializer()
        binit = tf.zeros_initializer()
        # ReLU activation for recurrent layers
        rfunc = tf.nn.relu
        rinit = tf.contrib.layers.variance_scaling_initializer()
        # Tanh activation for output projection layer
        pinit = winit

        with tf.device(self.device):
            with tf.name_scope(self.name) as scope:
                # Token sequence
                tf.placeholder('string', [None, None], name='w')
                # Token sequence length
                tf.placeholder('int32', [None], name='n')
                # Style target
                tf.placeholder('float', [None, self.nc], name='s_')
                # True if training
                tf.placeholder('bool', name='t_mode')
                # True if validating
                tf.placeholder('bool', name='v_mode')
                # True if synthesizing
                tf.placeholder('bool', name='s_mode')
                # Dropout retain probability
                tf.placeholder('float', name='keep_prob')
                # Regularization penalty applied to feedforward layers
                tf.placeholder('float', name='reg_factor')
                # Initial learning rate
                tf.placeholder('float', name='learning_rate')
                # Training set size
                tf.placeholder('int32', name='dataset_size')
                # Per-epoch learning rate decay
                tf.placeholder('float', name='decay_rate')
                # Learning rate floor
                tf.placeholder('float', name='min_learning_rate')
                # Gradient clipping
                tf.placeholder('float', name='clip_threshold')

                tf.Variable(0, trainable=False, dtype='int32', name='t_step_0')
                tf.assign(self.t_step_0,
                          self.t_step_0 + tf.cast(self.t_mode, 'int32'), name='t_step')

                tf.Variable(0, trainable=False, dtype='int32', name='v_step_0')
                tf.assign(self.v_step_0,
                          self.v_step_0 + tf.cast(self.v_mode, 'int32'), name='v_step')

                tf.maximum(
                    tf.train.exponential_decay(
                        self.learning_rate,
                        self.t_step,
                        self.dataset_size,
                        self.decay_rate,
                        False
                    ),
                    self.min_learning_rate,
                    name='adjusted_learning_rate'
                )

                self._optimizer = tf.train.AdamOptimizer(
                    self.adjusted_learning_rate
                )

                # For standardizing style output
                tf.constant(mean, name='mean')
                tf.constant(stddev, name='stddev')

                table = tf.contrib.lookup.index_table_from_tensor(
                    mapping=tf.constant(vocab),
                    name='T'
                )

                tf.Variable(embed, dtype='float', name='E', trainable=True)
                if self.df > 0:
                    tf.Variable(winit([self.ne,self.nf]), name='W0')
                    tf.Variable(binit([1,self.nf]), name='b0')
                    if self.df > 1:
                        for d in range(1, self.df):
                            tf.Variable(winit([self.nf,self.nf]), name='W{}'.format(d))
                            tf.Variable(binit([1,self.nf]), name='b{}'.format(d))

                h = tf.reshape(self.w, [-1])
                h = tf.nn.embedding_lookup(self.E, table.lookup(h))
                for d in range(self.df):
                    h = wfunc(h @ self['W{}'.format(d)] + self['b{}'.format(d)])
                shape = tf.shape(self.w)
                h = tf.reshape(h, tf.concat([shape, [self.ne]], axis=0))

                # Create bidirectional RNN
                self._brnn(rinit, rfunc, pinit)(h)
                #self._sbrnn(cell_type, rinit, rfunc, pinit)(h)
                # self._cnn(rfunc, rinit, rfunc, pinit)(h)
                tf.add(self.ss * self.stddev, self.mean, name='s')

                j = tf.reduce_mean(tf.abs(self.ss - (self.s_ - self.mean) / self.stddev))
                if self.df > 0:
                    tf.add(j, self.reg_factor *
                              tf.add_n([tf.nn.l2_loss(self['W{}'.format(d)])
                                        for d in range(self.df)]), name='j')
                else:
                    tf.identity(j, name='j')
                # self.optimizer.minimize(self.j, name='o')
                self.optimizer.apply_gradients(
                    [(tf.clip_by_value(g, -self.clip_threshold, self.clip_threshold), v)
                     for g, v in self.optimizer.compute_gradients(self.j)],
                    name='o'
                )

                tf.summary.scalar('summary', self.j)

    def _rnn(self, rinit, rfunc, pinit):
        """
        Create unidirectional RNN cells.
        Not used in the paper.

        rinit: Initialization for recurrent cells
        rfunc: Activation for recurrent cells
        pinit: Initialization for output projection layer
        """
        def wrapper(h):
            with tf.variable_scope('rnn', initializer=rinit):
                cells = tf.nn.rnn_cell.MultiRNNCell(
                    [tf.nn.rnn_cell.DropoutWrapper(
                        self.cell_type(num_units=NR, activation=rfunc),
                        output_keep_prob=self.keep_prob
                     )
                     for d in range(DR)],
                    state_is_tuple=True
                )

                y, q = tf.nn.dynamic_rnn(
                    cell=cells,
                    inputs=h,
                    sequence_length=self.n,
                    dtype='float',
                    time_major=False
                )

            if USE_LSTM:
                tf.identity(q[-1].h, name='q')
            else:
                tf.identity(y[-1], name='q')

            tf.Variable(pinit([NR, NC]), name='P')
            tf.matmul(self.q, self.P, name='s')
        return wrapper

    def _brnn(self, rinit, rfunc, pinit):
        """
        Create bidirectional RNN cells.

        rinit: Initialization for recurrent cells
        rfunc: Activation for recurrent cells
        pinit: Initialization for output projection layer
        """
        def wrapper(h):
            with tf.variable_scope('brnn', initializer=rinit):
                cells = tf.nn.rnn_cell.MultiRNNCell(
                    [tf.nn.rnn_cell.DropoutWrapper(
                            self.cell_type(num_units=self.nr, activation=rfunc),
                            output_keep_prob=self.keep_prob
                     )
                     for d in range(self.dr)],
                    state_is_tuple=True
                )

                y, q = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cells,
                    cell_bw=cells,
                    inputs=h,
                    sequence_length=self.n,
                    dtype='float',
                    time_major=False
                )

            if USE_LSTM:
                tf.concat([q[0][-1].h, q[1][-1].h], axis=1, name='q')
            else:
                tf.concat([q[0][-1], q[1][-1]], axis=1, name='q')

            tf.Variable(pinit([2*self.nr, self.nc]), name='P')
            tf.matmul(self.q, self.P, name='ss')
        return wrapper

    def _sbrnn(self, rinit, rfunc, pinit):
        """
        Create stacked bidirectional RNN cells.
        Not used in the paper.

        rinit: Initialization for recurrent cells
        rfunc: Activation for recurrent cells
        pinit: Initialization for output projection layer
        """
        def wrapper(h):
            with tf.variable_scope('sbrnn', initializer=rinit):
                cells = [
                    tf.nn.rnn_cell.DropoutWrapper(
                        self.cell_type(num_units=NR,
                                      activation=rfunc),
                        output_keep_prob=self.keep_prob
                    )
                    for d in range(DR)
                ]

                y, qf, qb = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    cells_fw=cells,
                    cells_bw=cells,
                    inputs=h,
                    sequence_length=self.n,
                    dtype='float',
                    time_major=False
                )

            if USE_LSTM:
                tf.concat([q[0][-1].h, q[1][-1].h], axis=1, name='q')
            else:
                tf.concat([q[0][-1], q[1][-1]], axis=1, name='q')

            tf.Variable(pinit([2*NR, NC]), name='P')
            tf.matmul(self.q, self.P, name='ss')
        return wrapper

    def _cnn(self, activation, rinit, rfunc, pinit):
        """
        Create CNN cells based on
        ("Learning Generic Sentence Representations
            Using Convolutional Neural Networks" by Gan et. al, 2017).
        Not used in the paper.

        rinit: Initialization for convolution cells
        rfunc: Activation for recurrent cells
        pinit: Initialization for output projection layer
        """
        def wrapper(h):
            # h = tf.expand_dims(h, -1)
            outputs = []
            n_max = 3
            for n in range(3, n_max+1):
                with tf.variable_scope('cnn{}'.format(n)):
                    l = h
                    for d in range(1):
                        l = tf.nn.convolution(
                            h,
                            filter=tf.Variable(rinit([n, NR, NR]), name='W'),
                            # stride=1,
                            #tf.Variable(rinit([n, NR, NR]), name='W'),
                            #strides=1,
                            padding='SAME'
                        )
                        l = rfunc(
                            tf.nn.bias_add(l,
                                           tf.Variable(tf.zeros([NR]), name='b')))
                    l = tf.nn.max_pool(
                        tf.expand_dims(l, -1),
                        ksize=[1, NR, NR, 1],
                        strides=[1, 1, 1, 1],
                        padding='SAME'
                    )
                    outputs.append(l)
            q = tf.reshape(tf.stack(outputs, axis=1), [-1, len(outputs)])
            q = tf.nn.dropout(q, self.keep_prob, name='q')
            tf.Variable(pinit([len(outputs), NC]), name='P')
            tf.matmul(q, self.P, name='ss')
        return wrapper

    def encode(self, tokens, lengths, targets, train=False, **kwargs):
        """
        Predict the style of a token sequence while:
            1. Training the model (w/ targets != None and train == True)
            2. Validating the model (w/ targets != None and train == False) or
            3. Doing nothing else (w/ targets == None and train == False)

        tokens: Token sequences
        lengths: Corresponding lengths of sequences
        targets: Target style vectors
        **kwargs
            reg_factor: Regularization penalty
            learning_rate: Initial learning rate
            min_learning_rate: Learning rate floor
            decay_rate: Per-epoch learning rate decay
            clip_threshold: Clip gradient at positive and negative values
            dataset_size: Number of examples in the training set
            keep_prob: Dropout retain probability
        """
        reg_factor = kwargs.get('reg_factor', 0)
        learning_rate = kwargs.get('learning_rate', 1e-4)
        min_learning_rate = kwargs.get('min_learning_rate', 0)
        decay_rate = kwargs.get('decay_rate', 1.0)
        clip_threshold = kwargs.get('clip_threshold', 5.0)
        dataset_size = kwargs.get('dataset_size', 7000)
        keep_prob = kwargs.get('keep_prob', 1.0)
        session = tf.get_default_session()
        if train:
            return session.run(
                [self.s, self.j, self.summary, self.t_step, self.o],
                feed_dict={
                    self.w: tokens,
                    self.n: lengths,
                    self.s_: targets,
                    self.reg_factor: reg_factor,
                    self.learning_rate: learning_rate,
                    self.min_learning_rate: min_learning_rate,
                    self.decay_rate: decay_rate,
                    self.clip_threshold: clip_threshold,
                    self.dataset_size: dataset_size,
                    self.keep_prob: keep_prob,
                    self.t_mode: True,
                    self.v_mode: False
                }
            )[:4]
        elif targets is not None:
            return session.run(
                [self.s, self.j, self.summary, self.v_step],
                feed_dict={
                    self.w: tokens,
                    self.n: lengths,
                    self.s_: targets,
                    self.keep_prob: 1.0,
                    self.reg_factor: reg_factor,
                    self.t_mode: False,
                    self.v_mode: True
                }
            )
        else:
            return session.run(
                [self.s],
                feed_dict={
                    self.w: tokens,
                    self.n: lengths,
                    self.keep_prob: 1.0,
                    self.t_mode: False,
                    self.v_mode: False
                }
            )

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def ne(self):
        """
        Embedding vector dimension
        """
        return ds.NE

    @property
    def nf(self):
        """
        Number of nodes per feedforward layer
        """
        return cfg_enc.get('nf', self.ne)

    @property
    def df(self):
        """
        Number of feedforward layers
        """
        return cfg_enc.get('df', 0)

    @property
    def nr(self):
        """
        Number of nodes per recurrent layer
        """
        return cfg_enc.get('nr', self.ne)

    @property
    def dr(self):
        """
        Number of recurrent layers
        """
        return cfg_enc.get('dr', 1)

    @property
    def cell_type(self):
        """
        Recurrent cell type. Either BasicLSTMCell or GRUCell
        """
        return tf.nn.rnn_cell.BasicLSTMCell if cfg_enc.get('lstm', True) \
               else tf.nn.rnn_cell.GRUCell

    @property
    def device(self):
        """
        Device type (CPU or GPU) to use for the model.
        """
        return '/' + cfg_enc.get('device', 'cpu') + ':0'


class Unpacker(Model):
    def __init__(self, **kwargs):
        Model.__init__(self, 'Unpacker', **kwargs)

    def _create(self, **kwargs):
        """
        **kwargs
            sentences: IDs of all training sentences
            P: Initial value for the projection layer (optional)
        """
        try:
            sentences = kwargs['sentences']
        except KeyError:
            raise ValueError('Missing argument: sentences')
        P = kwargs.get('P', None)

        winit = tf.contrib.layers.xavier_initializer()
        wfunc = tf.nn.tanh
        binit = winit
        pinit = winit

        with tf.device(self.device):
            with tf.name_scope(self.name) as scope:
                # Linguistic features
                tf.placeholder('float', [None, self.nl], name='l')
                # Sentence ID
                tf.placeholder('string', [None], name='s')
                # Control vector
                tf.placeholder('float', [None, self.nc], name='c')
                # Target duration
                tf.placeholder('float', [None, self.nt], name='t_')
                # True if training
                tf.placeholder('bool', name='t_mode')
                # True if validating
                tf.placeholder('bool', name='v_mode')
                # True if synthesizing
                tf.placeholder('bool', name='s_mode')
                # Regularization penalty factor
                tf.placeholder('float', name='reg_factor')
                # Dropout retain probability
                tf.placeholder('float', name='keep_prob')
                # Initial learning rate
                tf.placeholder('float', name='learning_rate')
                # Training set size
                tf.placeholder('int32', name='dataset_size')
                # Per-epoch learning rate decay
                tf.placeholder('float', name='decay_rate')
                # Learning rate floor
                tf.placeholder('float', name='min_learning_rate')
                # Scale projection layer learning rate
                tf.placeholder('float', name='projection_factor')
                # Scale output layer learning rate
                tf.placeholder('float', name='output_factor')

                tf.Variable(0, trainable=False, dtype='int32', name='t_step_0')
                tf.assign(self.t_step_0,
                          self.t_step_0 + tf.cast(self.t_mode, 'int32'), name='t_step')

                tf.Variable(0, trainable=False, dtype='int32', name='v_step_0')
                tf.assign(self.v_step_0,
                          self.v_step_0 + tf.cast(self.v_mode, 'int32'), name='v_step')

                tf.maximum(
                    tf.train.exponential_decay(
                        self.learning_rate,
                        self.t_step,
                        self.dataset_size,
                        self.decay_rate,
                        False
                    ),
                    self.min_learning_rate,
                    name='adjusted_learning_rate'
                )

                self._optimizer1 = tf.train.AdamOptimizer(
                    self.adjusted_learning_rate
                )

                self._optimizer2 = tf.train.AdamOptimizer(
                    self.adjusted_learning_rate * self.projection_factor
                )

                self._optimizer3 = tf.train.AdamOptimizer(
                    self.adjusted_learning_rate * self.output_factor
                )

                table = tf.contrib.lookup.index_table_from_tensor(
                    mapping=tf.constant(sentences),
                    num_oov_buckets=1,
                    name='T'
                )
                index = table.lookup(self.s)

                if P is None:
                    tf.Variable(pinit([len(sentences)+1, self.nc]), name='P')
                else:
                    tf.Variable(P, name='P')
                tf.nn.embedding_lookup(self.P, index, name='e')

                h = tf.cond(self.s_mode, lambda: self.c, lambda: self.e)
                h = tf.concat([self.l, h], axis=1)
                for d in range(self.dh):
                    Wd = 'W{}'.format(d)
                    bd = 'b{}'.format(d)

                    if d == self.dh - 1:
                        tf.Variable(winit([self.nh, self.nt]), name=Wd)
                        tf.Variable(binit([1, self.nt]), name=bd)
                        h = tf.add(h @ self[Wd], self[bd], name='t')
                    else:
                        if d == 0:
                            tf.Variable(winit([self.nl+self.nc,self.nh]), name=Wd)
                        else:
                            tf.Variable(winit([self.nh,self.nh]), name=Wd)
                        tf.Variable(binit([1,self.nh]), name=bd)
                        h = tf.nn.dropout(wfunc(h @ self[Wd] + self[bd]), self.keep_prob)

                j = tf.reduce_mean(tf.abs(self.t - self.t_))
                tf.add(j, tf.add_n([tf.nn.l2_loss(self['W{}'.format(d)])
                                    for d in range(self.dh)]) * self.reg_factor,
                       name='j')

                variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
                v2 = [self.P.op.name]
                v3 = [self['W{}'.format(self.dh-1)].op.name,
                      self['b{}'.format(self.dh-1)].op.name]
                self.optimizer1.minimize(
                    j,
                    var_list=[
                        v
                        for v in variables
                        if v.op.name not in v2 + v3
                    ],
                    name='o1')
                self.optimizer2.minimize(
                    j,
                    var_list=[
                        v
                        for v in variables
                        if v.op.name in v2
                    ],
                    name='o2')
                self.optimizer3.minimize(
                    j,
                    var_list=[
                        v
                        for v in variables
                        if v.op.name in v3
                    ],
                    name='o3'
                )

                tf.group(self.o1, self.o2, self.o3, name='o')

                tf.summary.scalar('summary', self.j)

    def train(self, linguistics, sentences, targets, train=False, **kwargs):
        session = tf.get_default_session()
        dummy = np.zeros([linguistics.shape[0], self.nc], dtype=float)
        reg_factor = kwargs.get('reg_factor', 0)
        learning_rate = kwargs.get('learning_rate', 1e-4)
        decay_rate = kwargs.get('decay_rate', 1.0)
        min_learning_rate = kwargs.get('min_learning_rate', 0)
        dataset_size = kwargs.get('dataset_size', 0)
        keep_prob = kwargs.get('keep_prob', 1.0)
        projection_factor = kwargs.get('projection_factor', 1.0)
        output_factor = kwargs.get('output_factor', 1.0)

        feed_dict = {
            self.l: linguistics,
            self.s: sentences,
            self.c: dummy,
            self.s_mode: False
        }
        if train:
            return session.run(
                [self.t, self.j, self.summary, self.t_step, self.o],
                feed_dict={
                    **feed_dict,
                    self.t_: targets,
                    self.reg_factor: reg_factor,
                    self.learning_rate: learning_rate,
                    self.decay_rate: decay_rate,
                    self.dataset_size: dataset_size,
                    self.min_learning_rate: min_learning_rate,
                    self.keep_prob: keep_prob,
                    self.projection_factor: projection_factor,
                    self.output_factor: output_factor,
                    self.t_mode: True,
                    self.v_mode: False
                }
            )[:4]
        elif targets is not None:
            return session.run(
                [self.t, self.j, self.summary, self.v_step],
                feed_dict={
                    **feed_dict,
                    self.t_: targets,
                    self.reg_factor: reg_factor,
                    self.keep_prob: 1.0,
                    self.t_mode: False,
                    self.v_mode: True
                }
            )
        else:
            return session.run(
                [self.t],
                feed_dict={
                    **feed_dict,
                    self.keep_prob: 1.0,
                    self.t_mode: False,
                    self.v_mode: False
                }
            )

    def synth(self, linguistics, controls):
        return tf.get_default_session().run(
            [self.t],
            feed_dict={
                self.l: linguistics,
                self.s: np.array(['']*linguistics.shape[0]),
                self.c: controls,
                self.keep_prob: 1.0,
                self.s_mode: True,
                self.t_mode: False,
                self.v_mode: False
            }
        )

    def embed(self, sentences):
        return tf.get_default_session().run(
            [self.e],
            feed_dict={
                self.s: sentences
            }
        )

    @property
    def optimizer1(self):
        return self._optimizer1

    @property
    def optimizer2(self):
        return self._optimizer2

    @property
    def optimizer3(self):
        return self._optimizer3

    @property
    def nh(self):
        """
        Number of nodes per hidden layer.
        """
        return cfg_unp.get('nh', 64)

    @property
    def dh(self):
        """
        Number of hidden layers.
        """
        return cfg_unp.get('dh', 3)

    @property
    def nt(self):
        """
        Duration output dimension.
        Typically five since that's how many states there are per phone.
        """
        return ds.NT

    @property
    def device(self):
        """
        Device type (CPU or GPU) to use for the model.
        """
        return '/'+cfg_unp.get('device', 'cpu')+':0'
