from __init__ import *
import dataset as ds


NL = ds.LX_DIM
NC = cfg_net.get('control-dim', 2)
NH = cfg_net.get('num-hidden-nodes', 256)
NA = ds.AX_DIM
DP = cfg_net.get('num-hidden-layers', 6)
RP = 1e-5
DEVICE = cfg_net.get('device', 'cpu')
DEVICE = '/gpu:0' if DEVICE == 'gpu' else '/cpu:0'
