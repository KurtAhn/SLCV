#!/usr/bin/python2
import sys, os
sys.path.append(os.environ['MAGPHASE'])
import magphase as mp
import libaudio as la
import libutils as lu
from os import path
from argparse import ArgumentParser


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-s', '--sentence', dest='sentence', required=True)
    p.add_argument('-o', '--outdir', dest='outdir', required=True)
    p.add_argument('-m', '--magdim', dest='magdim', type=int, required=True)
    p.add_argument('-p', '--phadim', dest='phadim', type=int, required=True)
    p.add_argument('-f', '--filter', dest='filter', default='no')
    p.add_argument('-v', '--variable', dest='const_rate', action='store_false')
    a = p.parse_args()

    mp.synthesis_from_acoustic_modelling(a.outdir,
                                         a.sentence,
                                         a.outdir,
                                         a.magdim,
                                         a.phadim,
                                         48000,
                                         pf_type=a.filter,
                                         b_const_rate=a.const_rate)
