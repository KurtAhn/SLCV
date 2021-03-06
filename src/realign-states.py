#!/usr/bin/env python2
from __init__ import load_config
import sys, os
sys.path.append(os.environ['MAGPHASE'])
import magphase as mp
import libutils as lu
import libaudio as la
from os import path
from argparse import ArgumentParser


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-s', '--senlst', dest='senlst', required=True)
    p.add_argument('-c', '--config', dest='config', required=True)
    a = p.parse_args()

    load_config(a.config)
    from __init__ import *

    CONST_RATE = cfg_data.get('const', True)

    with open(a.senlst) as f:
        sentences = [l.rstrip() for l in f]

    if CONST_RATE:
        for s in sentences:
            os.symlink(path.join(HTS1DIR, s+'.lab'), path.join(HTS2DIR, s+'.lab'))
    else:
        for s in sentences:
            htsfile = path.join(HTS1DIR, s+'.lab')
            outfile = path.join(HTS2DIR, s+'.lab')

            try:
                shift = lu.read_binfile(path.join(ACO1DIR, s+'.shift'), dim=1)
                frames = mp.get_num_of_frms_per_state(shift, htsfile, 48000, False)
                la.convert_label_state_align_to_var_frame_rate(htsfile, frames, outfile)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                pass
            else:
                print1(s)
