#!/usr/bin/env python
from __init__ import load_config
import sys, os
sys.path.append(os.environ['MERLIN'])
from frontend.label_normalisation import HTSLabelNormalisation
from frontend.silence_remover import SilenceRemover, trim_silence
from frontend.min_max_norm import MinMaxNormalisation
from os import path
from argparse import ArgumentParser


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-c', '--config', dest='config', required=True)
    a = p.parse_args()

    load_config(a.config)
    from __init__ import *

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f if l]

    hts2 = [path.join(HTS2DIR, s+'.lab') for s in sentences]
    lab1 = [path.join(LAB1DIR, s+'.lab') for s in sentences]
    lab2 = [path.join(LAB2DIR, s+'.lab') for s in sentences]
    lab3 = [path.join(LAB3DIR, s+'.lab') for s in sentences]

    binarizer = HTSLabelNormalisation(question_file_name=path.join(RESDIR, '600.hed'))
    binarizer.perform_normalisation(hts2, lab1)

    remover = SilenceRemover(n_cmp=binarizer.dimension, silence_pattern=['*-#+*'])
    remover.remove_silence(lab1, hts2, lab2)

    normalizer = MinMaxNormalisation(feature_dimension=binarizer.dimension,
                                     min_value=0.01, max_value=0.99)
    normalizer.find_min_max_values(lab2)
    normalizer.normalise_data(lab2, lab3)
