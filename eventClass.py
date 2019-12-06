import numpy as np
from pyjet import cluster,DTYPE_PTEPM

#container for all events
class EventContainer(object):
    def __init__(self):
        self.numEvents=-1
        self.numSignalEvents=-1
        self.numBackgroundEvents=-1
        self.allEvents=[]
        self.allSignalEvents=[]
        self.allBkgdEvents=[]
        return

    def __str__(self):
        return "Container entries: {0}".format(self.numEvents)
    
    def readEvents(self,events_combined,nEvts=-1,truthBit=True):
        eventTypes=["unlabelled"] if not truthBit else ["signal","background"]
        for i in range(nEvts):
            if (i%(nEvts/10)==0): print("Event: ",i)
            jEvt=JetEvent(i)
            #create collection of all jets
            jEvt.runDefaultJetClustering(eventTypes)
            #calculate kinematic and topological variables
            jEvt.calculateVariables()
        return 

    
#class object keeping all jets per event
class JetEvent(object):
    
    def __init__(self,idx=-1):
        self.idx=idx
        self.allJets=[]
        self.type=""
        self.isSignal=False
        return

    def __str__(self):
        return "Event index {0}".format(self.idx)
    
    def convertJetsToDType(self,jets=None):
        pseudojet_dtype  = [('jetidx', np.int),
                ('px', np.float64),
                      ('py', np.float64),
                      ('pz', np.float64),
                      ('eta', np.float64), 
                      ('phi', np.float64), 
                      ('mass', np.float64),
                      ('signal', bool)] # True for signal, False for background
        allJetsPerEvent_dType = np.zeros((len(jets), ), dtype=pseudojet_dtype)
        jet_idx = 0
        for jet in jets:
            allJetsPerEvent_dType[jet_idx] = (jet_idx, jet.px, jet.py, jet.pz, jet.eta, jet.phi, jet.mass, self.issignal)
            jet_idx += 1
        return allJetsPerEvent_dType

    def runDefaultJetClustering(self,eventTypes=[]):
        for evtType in eventTypes:
            if "signal" in evtType or "background" in evtType:
                issignal = events_combined[self.idx][2100]
                if (evtType=='background' and issignal) or (evtType=='signal' and issignal==0): continue
                self.issignal=issignal
            self.type=evtType
            #actual jet clustering, taken from LHCOlympics page
            pseudojets_input = np.zeros(len([x for x in events_combined[self.idx][::3] if x > 0]), dtype=DTYPE_PTEPM)
            for j in range(700):
                if (events_combined[self.idx][j*3]>0):
                    pseudojets_input[j]['pT'] = events_combined[self.idx][j*3]
                    pseudojets_input[j]['eta'] = events_combined[self.idx][j*3+1]
                    pseudojets_input[j]['phi'] = events_combined[self.idx][j*3+2]
                    pass
                pass
            sequence = cluster(pseudojets_input, R=1.0, p=-1)
            jets = sequence.inclusive_jets(ptmin=20)
            self.allJets = self.convertJetsToDType(jets)
            pass
        print(self.allJets)
        pass

    def calculateVariables(self):
        return

if __name__=="__main__":
    import pandas
    nEvts=10
    fn =  './data/events_anomalydetection.h5'
    # Option 2: Only load the first 10k events for testing
    df_test = pandas.read_hdf(fn, stop=nEvts)
    f=df_test
    events_combined = f.T
    np.shape(events_combined)

    evtContainer=EventContainer()
    evtContainer.readEvents(events_combined,nEvts)
    print(evtContainer)



# from O:
## How many total jets?
#num_clusters = len(alljets['background']) + len(alljets['signal'])
#
#print(len(alljets['background']))
#print(len(alljets['signal']))
#num_total_jets = sum([len(x) for x in alljets['background']]) + sum([len(x) for x in alljets['signal']])
#
## Collect the jet data
#jet_data = np.zeros((num_total_jets, ), dtype=pseudojet_dtype)
#
#cluster_idx = 0
#jet_idx = 0
#
#for jet_type in ['background', 'signal']:
#    for cluster in alljets[jet_type]:
#        for jet in cluster:
#            is_signal = True if jet_type == 'signal' else False
#            jet_data[jet_idx] = (cluster_idx, jet.px, jet.py, jet.pz, jet.eta, jet.phi, jet.mass, is_signal)
#            jet_idx += 1
#        cluster_idx += 1
##
