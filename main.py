import matplotlib.pyplot as plt
import numpy as np
import pandas
import pickle

from tools import timelogDict,chronomat

from dataPreparation import createJetCollections
from tools import calcMjj

def plotter(allJets, leadJet):
    #Let's make some very simple plots.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    n,b,p = plt.hist(leadJet['background'], bins=50, facecolor='r', alpha=0.2,label='background')
    plt.hist(leadJet['signal'], bins=b, facecolor='b', alpha=0.2,label='signal')
    plt.xlabel(r'Leading jet $p_{T}$ [GeV]')
    plt.ylabel('Number of events')
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig("leadjetpt.pdf")
    
    mjj=calcMjj()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    n,b,p = plt.hist(mjj['background'], bins=50, facecolor='r', alpha=0.2,label='background')
    plt.hist(mjj['signal'], bins=b, facecolor='b', alpha=0.2,label='signal')
    plt.xlabel(r'$m_{JJ}$ [GeV]')
    plt.ylabel('Number of events')
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig("mjj.pdf")
    return

def printDSStats(allJets, leadJet):
  print(allJets)
  print(leadJet)

@chronomat
def main():
    nEvts=10
    fn =  './data/events_anomalydetection.h5'
    
    # Option 2: Only load the first 10k events for testing
    df_test = pandas.read_hdf(fn, stop=nEvts)
    f=df_test
    events_combined = f.T
    np.shape(events_combined)

    allJets,leadJet=createJetCollections(events_combined,nEvts)
   # plotter(allJets,leadJet)
    printDSStats(allJets,leadJet)    

    #outtag="jetCollection_%s_test"%(int(nEvts))
    #pklFile=open("./output/{0}.pkl".format(outtag),'wb')
    #pickle.dump( allJets  , pklFile)
    #pickle.dump( leadJet , pklFile)
    
    for key,val in timelogDict.items():
      print(key,val)
    return

if __name__=="__main__":
    main()




