import pandas
from pyjet import cluster,DTYPE_PTEPM

import numpy as np

from tools import chronomat


def createJetsWithTruth(events_combined, nEvts):
    leadpT = {}
    alljets = {}
    for mytype in ['background','signal']:
        leadpT[mytype]=[]
        alljets[mytype]=[]
        for i in range(nEvts): #len(events_combined)):
            if (i%(nEvts/10)==0):
                print(mytype,i)
                pass
            issignal = events_combined[i][2100]
            if (mytype=='background' and issignal):
                continue
            elif (mytype=='signal' and issignal==0):
                 continue
            pseudojets_input = np.zeros(len([x for x in events_combined[i][::3] if x > 0]), dtype=DTYPE_PTEPM)
            for j in range(700):
                if (events_combined[i][j*3]>0):
                    pseudojets_input[j]['pT'] = events_combined[i][j*3]
                    pseudojets_input[j]['eta'] = events_combined[i][j*3+1]
                    pseudojets_input[j]['phi'] = events_combined[i][j*3+2]
                    pass
                pass
            sequence = cluster(pseudojets_input, R=1.0, p=-1)
            jets = sequence.inclusive_jets(ptmin=20)
            leadpT[mytype] += [jets[0].pt]
            alljets[mytype] += [jets]
            pass
    return alljets,leadpT


#Now, let's cluster some jets!
@chronomat
def createJetCollections(events_combined, nEvts, truthBit=False, dataLabel="UnlabelledData"):
    if truthBit:
        alljets,leadpT=createJetsWithTruth(events_combined, nEvts)
    else:
        leadpT = {}
        alljets = {}
        leadpT[dataLabel]=[]
        alljets[dataLabel]=[]
        for i in range(nEvts): #len(events_combined)):
            if (i%(nEvts/10)==0):
                print(dataLabel,i)
                pass
            pseudojets_input = np.zeros(len([x for x in events_combined[i][::3] if x > 0]), dtype=DTYPE_PTEPM)
            for j in range(700):
                if (events_combined[i][j*3]>0):
                    pseudojets_input[j]['pT'] = events_combined[i][j*3]
                    pseudojets_input[j]['eta'] = events_combined[i][j*3+1]
                    pseudojets_input[j]['phi'] = events_combined[i][j*3+2]
                    pass
                pass
            sequence = cluster(pseudojets_input, R=1.0, p=-1)
            jets = sequence.inclusive_jets(ptmin=20)
            leadpT[dataLabel] += [jets[0].pt]
            alljets[dataLabel] += [jets]
            pass
    return alljets, leadpT
