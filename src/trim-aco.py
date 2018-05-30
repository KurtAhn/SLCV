#!/usr/bin/python3
from __init__ import *
import sys
sys.path.append('/home/kurt/Documents/etc/merlin/src')
from frontend.label_normalisation import HTSLabelNormalisation
from frontend.silence_remover import SilenceRemover, trim_silence
from frontend.min_max_norm import MinMaxNormalisation
from os import path
import acoustic as ax
import dataset as ds
from argparse import ArgumentParser


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-a', '--alndir', dest='alndir', default=ALNDIR)
    p.add_argument('-r', '--refdir', dest='refdir', default=ACODIR)
    p.add_argument('-o', '--outdir', dest='outdir', default=ACOTDIR)
    a = p.parse_args()

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f]

    for x, n in {'mag': ax.MAG_LEN,
                 'real': ax.REAL_LEN,
                 'imag': ax.IMAG_LEN,
                 'lf0': ax.LF0_LEN}.items():
        remover = SilenceRemover(n_cmp=n,
                                 silence_pattern=['*-#+*'],
                                 label_type='hts')
        remover.remove_silence([path.join(a.refdir, s+'.'+x) for s in sentences],
                               [path.join(a.alndir, s+'.lab') for s in sentences],
                               [path.join(a.outdir, s+'.'+x) for s in sentences])
