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
    p.add_argument('-v', '--vocdir', dest='vocdir', required=True)
    p.add_argument('-o', '--outdir', dest='outdir', required=True)
    p.add_argument('-m', '--magdim', dest='magdim', type=int, required=True)
    p.add_argument('-p', '--phadim', dest='phadim', type=int, required=True)
    a = p.parse_args()

    # mp.synthesis_from_acoustic_modelling(a.outdir,
    #                                      a.sentence,
    #                                      a.outdir,
    #                                      a.magdim,
    #                                      a.phadim,
    #                                      48000,
    #                                      b_const_rate=False)

    mag = lu.read_binfile(path.join(a.vocdir, a.sentence+'.mag'), dim=a.magdim)
    real = lu.read_binfile(path.join(a.vocdir, a.sentence+'.real'), dim=a.phadim)
    imag = lu.read_binfile(path.join(a.vocdir, a.sentence+'.imag'), dim=a.phadim)
    lf0 = lu.read_binfile(path.join(a.vocdir, a.sentence+'.lf0'), dim=1)
    v_syn_sig = mp.synthesis_from_compressed(mag, real, imag, lf0, 48000, b_const_rate=False, b_out_hpf=False)
    la.write_audio_file(path.join(a.outdir,a.sentence+'.wav'), v_syn_sig, 48000)
    # print 'Wrote audio'
