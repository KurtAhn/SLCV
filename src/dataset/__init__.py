from __init__ import *
import tensorflow as tf
try:
    TFRecordDataset = tf.data.TFRecordDataset
except AttributeError:
    from tensorflow.contrib.data import TFRecordDataset
import acoustic as ax
import numpy as np

# Exponentiate F0
EXP_F0 = cfg_data.get('exp', True)
# Use delta and delta-delta features
USE_DELTA = cfg_data.get('delta', True)
# Use constant frame rate
CONST_RATE = cfg_data.get('const', True)
# Control vector dimension
NC = cfg_data.get('nc', 2)
# Acoustic vector dimension (after delta)
NA = ax.AX_DIM * 3 + 1 if USE_DELTA else ax.AX_DIM + 1
# Linguistic feature dimension
NL = cfg_data.get('nl', 600)
# Word embedding dimension
NE = cfg_data.get('ne', 300)
# Number of states per phone
NT = 5
# Silent feature
SIL = cfg_data.get('silence-index', 160)
# Word feature (not used)
WIP = cfg_data.get('word-in-phrase-index', 582)
# Phrase feature (not used)
PIS = cfg_data.get('phrase-in-sentence-index', 593)


def count_examples(filepaths):
    """
    Count individual examples in TFRecord files.
    """
    n = 0
    for f in filepaths:
        for r in tf.python_io.tf_record_iterator(f):
            n += 1
    return n


def load_synthesizer_dataset(sentences):
    """
    Load dataset used for the synthesizer.
    (Linguistic features, Sentence ID) -> Acoustic features
    """
    return TFRecordDataset([path.join(TFRSDIR, sentence+'.tfr')
                            for sentence in sentences])\
        .map(
            lambda record: \
                tf.parse_single_example(
                    record,
                    features={
                        's': tf.FixedLenFeature([], tf.string),
                        'l': tf.FixedLenFeature([NL+9], tf.float32),
                        'a': tf.FixedLenFeature([NA], tf.float32)
                    }
                )
        )\
        .map(
            lambda feature: (feature['l'], feature['s'], feature['a'])
        )


def load_encoder_dataset(sentences, oracle=None):
    """
    Load semi-dataset used for the encoder.
    (Sentence ID, Word embeddings, Sequence length)
    Needds to be paired with target style vectors queried from a previously trained synthesizer.
    """
    dataset = TFRecordDataset([path.join(TFREDIR, sentence+'.tfr')
                               for sentence in sentences])\
        .map(
            lambda record: \
                tf.parse_single_example(
                    record,
                    features={
                        's': tf.FixedLenFeature([], tf.string),
                        'e': tf.FixedLenSequenceFeature([NE],
                                                        tf.float32,
                                                        allow_missing=True),
                        'n': tf.FixedLenFeature([], tf.int64)
                    }
                )
        )

    if oracle is None:
        return dataset.map(lambda feature: (feature['e'], feature['n']))
    else:
        indices = {s: n for n, s in enumerate(sentences)}
        return dataset.map(lambda feature: \
            (feature['e'],
             feature['n'],
             tf.py_func(
                lambda s: oracle[indices[s.decode('ascii')],:].reshape(NC),
                [feature['s']],
                tf.float32
             ))
        )


def load_encoder_dataset2(sentences, oracle=None):
    dataset = TFRecordDataset([path.join(TFREDIR, sentence+'.tfr')
                               for sentence in sentences])\
        .map(
            lambda record: \
                tf.parse_single_example(
                    record,
                    features={
                        's': tf.FixedLenFeature([], tf.string),
                        'w': tf.FixedLenSequenceFeature([], tf.string,
                                                        allow_missing=True),
                        'n': tf.FixedLenFeature([], tf.int64)
                    }
                )
        )

    if oracle is None:
        return dataset.map(lambda feature: (feature['w'], feature['n']))
    else:
        indices = {s: n for n, s in enumerate(sentences)}
        return dataset.map(lambda feature: \
            (feature['w'],
             feature['n'],
             tf.py_func(
                lambda s: oracle[indices[s.decode('ascii')],:].reshape(NC),
                [feature['s']],
                tf.float32
             ))
        )


def load_unpacker_dataset(sentences):
    """
    Load dataset used for the unpacker.
    (Linguistic features, Sentence ID) -> Acoustic features
    """
    return TFRecordDataset([path.join(TFRUDIR, sentence+'.tfr')
                            for sentence in sentences])\
        .map(
            lambda record: \
                tf.parse_single_example(
                    record,
                    features={
                        's': tf.FixedLenFeature([], tf.string),
                        'l': tf.FixedLenFeature([NL], tf.float32),
                        't': tf.FixedLenFeature([NT], tf.float32)
                    }
                )
        )\
        .map(
            lambda feature: (feature['l'], feature['s'], feature['t'])
        )
