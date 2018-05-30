#!/usr/bin/python3
from __init__ import *
import sys
sys.path.append('/home/kurt/Documents/etc/magphase/src')
import libutils as lu
from os import path
import acoustic as ax
import dataset as ds
import numpy as np
from argparse import ArgumentParser


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-r', '--refdir', dest='refdir', default=ACOTDIR)
    p.add_argument('-o', '--outdir', dest='outdir', default=ACODDIR)
    a = p.parse_args()

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f]

    for s in sentences:
        mag = lu.read_binfile(path.join(a.refdir, s+'.mag'), dim=ax.MAG_LEN)
        real = lu.read_binfile(path.join(a.refdir, s+'.real'), dim=ax.REAL_LEN)
        imag = lu.read_binfile(path.join(a.refdir, s+'.imag'), dim=ax.IMAG_LEN)
        lf0 = lu.read_binfile(path.join(a.refdir, s+'.lf0'), dim=ax.LF0_LEN)\
                .reshape([-1,ax.LF0_LEN])
        x = ax.acoustic(mag=mag, real=real, imag=imag, lf0=lf0)
        x[:,ax.LF0] = np.exp(x[:,ax.LF0])
        #x = ax.interpolate_f0(x)
        dx = ax.velocity(x)
        ddx = ax.acceleration(x)
        v = ax.voicing(x)
        x = np.concatenate([x, dx, ddx, v], axis=1)
        lu.write_binfile(x, path.join(a.outdir, s+'.aco'))
