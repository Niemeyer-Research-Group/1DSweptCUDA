'''Docstring'''

import os
import os.path as op
import matplotlib.pyplot as plt
from cycler import cycler
import pandas as pd
import palettable.colorbrewer as pal
import numpy as np
import main_help as mh

#Test program against exact values.
class ExactTest(object):
    
    def __init__(self,problem,plotpath):
        self.problem  = problem
        self.plotpath = plotpath
        
    def exact(self,L):
        
    def rmse(self,)
        
#Test that classic and swept give the same results
class ConsistencyTest(object):
    
    def __init__(self,problem,plotpath):
        self.problem  = problem
        self.plotpath = plotpath