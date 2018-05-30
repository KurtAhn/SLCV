#!/usr/bin/python3
from __init__ import *
import sys
sys.path.append('/home/kurt/Documents/etc/merlin/src')
from frontend.label_normalisation import HTSLabelNormalisation
from frontend.silence_remover import SilenceRemover, trim_silence
from frontend.min_max_norm import MinMaxNormalisation
from os import path
from argparse import ArgumentParser


if __name__ == '__main__':
    # 1. Vocode
    # 2. Realign
    # 3. Binarize <-
    # 4. Normalize
    # 5. TFR

    p = ArgumentParser()
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-a', '--alndir', dest='alndir', default=ALNRDIR)
    p.add_argument('-b', '--bindir', dest='bindir', default=LABDIR)
    p.add_argument('-t', '--trmdir', dest='trmdir', default=LABTDIR)
    p.add_argument('-n', '--nordir', dest='nordir', default=LABNDIR)
    a = p.parse_args()

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f if l]
    aln_files = [path.join(a.alndir, s+'.lab') for s in sentences]
    bin_files = [path.join(a.bindir, s+'.lab') for s in sentences]
    trm_files = [path.join(a.trmdir, s+'.lab') for s in sentences]
    nor_files = [path.join(a.nordir, s+'.lab') for s in sentences]

    binarizer = HTSLabelNormalisation(question_file_name=path.join(RESDIR, '600.hed'))
    binarizer.perform_normalisation(aln_files, bin_files)

    remover = SilenceRemover(n_cmp=binarizer.dimension, silence_pattern=['*-#+*'])
    remover.remove_silence(bin_files, aln_files, trm_files)

    normalizer = MinMaxNormalisation(feature_dimension=binarizer.dimension,
                                     min_value=0.01, max_value=0.99)
    normalizer.find_min_max_values(trm_files)
    normalizer.normalise_data(trm_files, nor_files)
