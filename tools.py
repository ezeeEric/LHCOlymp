import sys
from functools import wraps
import time
import numpy as np

#create global dictionary to store execution times of various methods
from collections import OrderedDict
timelogDict=OrderedDict()

def chronomat(method):
    """
    Method to measure execution time of single methods.
    Simply add the decorator @chronomat to your method to print methods execution time at the 
    end of the code run. Stores values in a global dictionary.
    """
    @wraps(method)
    def timeThis(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        timelogDict[method.__module__+'.'+method.__name__]='%2.2f ms' % ( (te - ts) * 1000)
        return result
    return timeThis

def accumulativeChronomat(method):
    """
    Method to measure execution time of single methods.
    Simply add the decorator @chronomat to your method to print methods execution time at the 
    end of the code run. Stores values in a global dictionary.
    """
    @wraps(method)
    def timeThis(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'acc_'+method.__module__+'.'+method.__name__ in timelogDict:
            timelogDict['acc_'+method.__module__+'.'+method.__name__]+=(te - ts) * 1000
        else:
            timelogDict['acc_'+method.__module__+'.'+method.__name__]=(te - ts) * 1000
        return result
    return timeThis

def pickleThis(pickleFile, obj):
    pass

def datasetInformation(dataFrame):
    
    pass


##########
def calcMjj(alljets):
    mjj={}
    for mytype in ['background','signal']:
        mjj[mytype]=[]
        for k in range(len(alljets[mytype])):
            E = alljets[mytype][k][0].e+alljets[mytype][k][1].e
            px = alljets[mytype][k][0].px+alljets[mytype][k][1].px
            py = alljets[mytype][k][0].py+alljets[mytype][k][1].py
            pz = alljets[mytype][k][0].pz+alljets[mytype][k][1].pz
            mjj[mytype]+=[(E**2-px**2-py**2-pz**2)**0.5]
            pass
        pass
    return mjj
