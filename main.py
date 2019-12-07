import matplotlib.pyplot as plt
import numpy as np
import pandas
import pickle

from tools import timelogDict,chronomat

from eventClass import EventContainer

#from dataPreparation import createJetCollections
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
#    fig = plt.figure()
#    ax = fig.add_subplot(1, 1, 1)
#    n,b,p = plt.hist(mjj['background'], bins=50, facecolor='r', alpha=0.2,label='background')
#    plt.hist(mjj['signal'], bins=b, facecolor='b', alpha=0.2,label='signal')
#    plt.xlabel(r'$m_{JJ}$ [GeV]')
#    plt.ylabel('Number of events')
#    plt.legend(loc='upper right')
#    plt.show()
#    plt.savefig("mjj.pdf")
    return

@chronomat
def main(args=None):
    nEvts=args.numberOfEvents
    inFile=args.inputFile
    # Option 2: Only load the first 10k events for testing
    dataFrame = pandas.read_hdf(inFile, stop=nEvts)
    events = dataFrame.T
    np.shape(events)
    
    evtContainer=EventContainer()
    evtContainer.readEvents(events,nEvts,truthBit=args.hasTruthbit)
    print("Read {0} events".format(evtContainer.numEvents))
    pklFile=open(inFile.replace(".h5",".pkl"),'wb')
    pickle.dump( evtContainer , pklFile)
    pickle.dump( args, pklFile)
    pklFile.close()
    return

if __name__=="__main__":
    from argparse import ArgumentParser
    argParser = ArgumentParser(add_help=False)
    
    argParser.add_argument( '-i', '--inputFile',  type=str, default="./data/events_LHCO2020_backgroundMC_Pythia.h5")
    argParser.add_argument( '-truth',  '--hasTruthbit', action="store_true")
    argParser.add_argument( '-nevts',  '--numberOfEvents', type=int, default=-1)
    args = argParser.parse_args()
    main(args)
    
    for key,val in timelogDict.items():
      print(key,val)
    pass
