

import numpy as np
import subprocess as sp
import shlex
import os
import json

OPTIONS_ONE = [
    "Heat",
    "Euler"
    "KS",
]

div = [2**k for k in range(11,21)]
blk = 256
dt = [0.2,0.2,0.005]

all_runs = dict()

t_fin = blk*10
freq


f = open(Varfile)
data = []
for k, line in enumerate(f):
    if i>1:
        #readit

#Do all the cases and save them acccording to Problem, scheme, div, Time
#
#ReadFile[:][0] is key [:][1:] is vals
