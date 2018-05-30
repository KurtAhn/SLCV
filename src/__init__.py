import sys
from os import path


PRJDIR = path.dirname(path.dirname(path.realpath(sys.argv[0])))
DATDIR = path.join(PRJDIR, 'data')
MDLDIR = path.join(DATDIR, 'models')
ALNDIR = path.join(DATDIR, 'hts')
ALNRDIR = path.join(DATDIR, 'hts2')
LABDIR = path.join(DATDIR, 'lab')
LABTDIR = path.join(DATDIR, 'labt')
LABNDIR = path.join(DATDIR, 'labtn')
ACODIR = path.join(DATDIR, 'aco')
ACOTDIR = path.join(DATDIR, 'acot')
#ACODDIR = path.join(DATDIR, 'acotd')
ACODDIR = path.join(DATDIR, 'acotd-noint')
TXTDIR = path.join(DATDIR, 'txt')
TRNDIR = path.join(DATDIR, 'train')
TRNDIR = path.join(DATDIR, 'train-noint')
SYNDIR = path.join(DATDIR, 'synth')
STTDIR = path.join(DATDIR, 'stats')
STTDIR = path.join(DATDIR, 'stats-noint')
OUTDIR = path.join(DATDIR, 'out')
RESDIR = path.join(PRJDIR, 'res')
SRCDIR = path.join(PRJDIR, 'src')


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
