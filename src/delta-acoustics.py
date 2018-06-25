#!/usr/bin/env python
from __init__ import load_config
import sys, os
sys.path.append(os.environ['MAGPHASE'])
import libutils as lu
from os import path
import numpy as np
from argparse import ArgumentParser


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-c', '--config', dest='config', required=True)
    a = p.parse_args()

    load_config(a.config)
    from __init__ import *
    import acoustic as ax

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f]

    for s in sentences:
        try:
            mag = lu.read_binfile(path.join(ACO2DIR, s+'.mag'), dim=ax.MAG_DIM)
            real = lu.read_binfile(path.join(ACO2DIR, s+'.real'), dim=ax.REAL_DIM)
            imag = lu.read_binfile(path.join(ACO2DIR, s+'.imag'), dim=ax.IMAG_DIM)
            lf0 = lu.read_binfile(path.join(ACO2DIR, s+'.lf0'), dim=ax.LF0_DIM)\
                    .reshape([-1,ax.LF0_DIM])
            x = ax.acoustic(mag=mag, real=real, imag=imag, lf0=lf0)
            x[:,ax.LF0] = np.exp(x[:,ax.LF0])
            v = ax.voicing(x)
            x = ax.interpolate_f0(x)
            dx = ax.velocity(x)
            ddx = ax.acceleration(x)
            x = np.concatenate([x, dx, ddx, v], axis=1)
            lu.write_binfile(x, path.join(ACO3DIR, s+'.aco'))
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            print2(e)
            pass
        else:
            print1(s)
