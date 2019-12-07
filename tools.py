import sys
from functools import wraps
import time
import numpy as np
from math import sqrt

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
def calcMjj(jets):
    E =  jets[0]['e'] +jets[1]['e']
    px = jets[0]['px']+jets[1]['px']
    py = jets[0]['py']+jets[1]['py']
    pz = jets[0]['pz']+jets[1]['pz']
    mjj=(E**2-px**2-py**2-pz**2)**0.5
    return mjj

def dEtajj(jets):
    return abs(jets[0]['eta']-jets[1]['eta'])

def dPhijj(jets):
    return abs(jets[0]['phi']-jets[1]['phi'])

def dRjj(jets):
    dEta=abs(jets[0]['eta']-jets[1]['eta'])
    dPhi=abs(jets[0]['phi']-jets[1]['phi'])
    return sqrt(dEta*dEta+dPhi*dPhi)

def dPtjj(jets):
    return abs(jets[0]['pt']-jets[1]['pt'])

def scalarSumAllJetPt(jets):
    return sum([jets[i]['pt'] for i in range(len(jets))])

def scalarSumDiJetPt(jets):
    return jets[0]['pt']+jets[1]['pt']

def vectorSumDiJetPt(jets):
    px = jets[0]['px']+jets[1]['px']
    py = jets[0]['py']+jets[1]['py']
    return sqrt(px*px+py*py)

def vectorSumAllJetPt(jets):
    px = sum([jets[i]['px'] for i in range(len(jets))])
    py = sum([jets[i]['py'] for i in range(len(jets))])
    return sqrt(px*px+py*py)



