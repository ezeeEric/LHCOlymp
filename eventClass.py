import pandas as pd
import numpy as np
from pyjet import cluster,DTYPE_PTEPM
from tools import addSingleJetVariables

#container for all events
class EventContainer(object):
    def __init__(self):
        self.allEvents=[]
        self.allSignalEvents=[]
        self.allBkgdEvents=[]
        return

    def __str__(self):
        return "Container entries: {0}".format(self.numEvents)
    
    @property
    def numEvents(self):
        return len(self.allEvents)
    
    @property
    def numSignalEvents(self):
        return len(self.allSignalEvents)
    
    @property
    def numBackgroundEvents(self):
        return len(self.allBkgdEvents)

    def readEvents(self,events,nEvts=-1,truthBit=True):
        eventTypes=["unlabelled"] if not truthBit else ["signal","background"]
        nEvts = len(events.T) if int(nEvts)<1 else nEvts
        for i in range(nEvts):
            if (i%(int(nEvts/10))==0): print("Event: ",i)
            jEvt=JetEvent(i)
            #create collection of all jets
            jEvt.runDefaultJetClustering(events[i],eventTypes)
            #calculate kinematic and topological variables
            if len(jEvt.allJets)>1:
                jEvt.calculateVariables()
                #store events
                self.allEvents.append(jEvt)
                if "signal" in jEvt.type: self.allSignalEvents.append(jEvt)
                elif "background" in jEvt.type: self.allBkgdEvents.append(jEvt)
            else:
                print("WARNING: Event with less than 2 jets, not recorded")
        return 
    
    def toDataFrame(self):
        print("Converting to dataframe")
        #dict carrying variables of event in container to be transformed to DF
        #define additional entries of dataframe
        dictToFrame={
                "evtIdx":[],
                "isSignal":[],
                "type":[],
                "nJets":[]
                }
        counter=0
        for evt in self.allEvents:
            #if counter>2:
            #    break
            dictToFrame["evtIdx"]+=[evt.idx]
            dictToFrame["isSignal"]+=[int(evt.isSignal)]
            dictToFrame["type"]+=[evt.type]
            dictToFrame["nJets"]+=[len(evt.allJets)]
            for key,item in evt.variables.items():
                if key in dictToFrame.keys():
                    dictToFrame[key]+=[item]
                else:
                    dictToFrame[key]=[item]
            dictToFrame=addSingleJetVariables(evt.allJets,dictToFrame) 
            counter+=1
        #sanity check whether all variable fields are of same length. Looks very clumsy
        dummy=0
        firstIt=True
        for key,val in dictToFrame.items():
            if firstIt:
                dummy=len(val)
                continue
            else:
                if len(val)!=dummy:
                    raise
            firstIt=False
        df=pd.DataFrame(dictToFrame)
        return df

#class object keeping all jets per event
class JetEvent(object):
    
    def __init__(self,idx=-1):
        self.idx=idx
        self.allJets=[]
        self.type=""
        self.isSignal=False
        self.variables={}
        return

    def __str__(self):
        return "Event index {0}".format(self.idx)

#automatised variable discovery from fastjet::pseudojet type
#not used for now, as explicit is easier to read
#   dummyJet=[jet for jet in jets][0]
#   for key in dir(dummyJet):
#       if type(getattr(dummyJet,key)) is float:
#           print(key,getattr(dummyJet,key))
#           pseudojet_dtype.append((key,np.float64))
    def convertJetsToDType(self,jets=None):
        constituents_dtype  = [
                      ('constidx', np.int),
                      ('e', np.float64),
                      ('eta', np.float64), 
                      ('phi', np.float64), 
                      ('pt', np.float64), 
                      ('px', np.float64),
                      ('py', np.float64),
                      ('pz', np.float64),
                      ] # True for signal, False for background
        pseudojet_dtype  = [
                      ('jetidx', np.int),
                      ('e', np.float64),
                      ('et', np.float64),
                      ('eta', np.float64), 
                      ('mass', np.float64),
                      ('phi', np.float64), 
                      ('pt', np.float64), 
                      ('px', np.float64),
                      ('py', np.float64),
                      ('pz', np.float64),
                      ('signal', bool),
                      ('numConstituents', np.int),
                      ('constituents', pseudojet_const_dtype)
                      ] # True for signal, False for background
        

        #  mt() why not implemented?
        allJetsPerEvent_dType = np.zeros((len(jets), ), dtype=pseudojet_dtype)
        jet_idx = 0
        for jet in jets:
            allConstPerJetPerEvent_dType = np.zeros((len(jet), ), dtype=pseudojet_const_dtype)
            
            #pt sort this? remove entries?
            const_idx=0
            for const in jet:
                allConstPerJetPerEvent_dType[const_idx] = (const_idx, const.e, const.eta, const.phi, const.pt, const.px, const.py, const.pz )
                const_idx+=1
            allJetsPerEvent_dType[jet_idx] = (jet_idx, jet.e, jet.et, jet.eta, jet.mass, jet.phi, jet.pt, jet.px, jet.py, jet.pz, self.isSignal, len(jet), allConstPerJetPerEvent_dType)
            import sys
            sys.exit()
            jet_idx += 1
        return allJetsPerEvent_dType

    def runDefaultJetClustering(self,event, eventTypes=[]):
        for evtType in eventTypes:
            if "signal" in evtType or "background" in evtType:
                isSignal = event[2100]
                if (evtType=='background' and isSignal) or (evtType=='signal' and isSignal==0): continue
                self.isSignal=isSignal
            self.type=evtType
            #actual jet clustering, taken from LHCOlympics page
            pseudojets_input = np.zeros(len([x for x in event[::3] if x > 0]), dtype=DTYPE_PTEPM)
            for j in range(700):
                if (event[j*3]>0):
                    pseudojets_input[j]['pT'] = event[j*3]
                    pseudojets_input[j]['eta'] = event[j*3+1]
                    pseudojets_input[j]['phi'] = event[j*3+2]
                    pass
                pass
            sequence = cluster(pseudojets_input, R=1.0, p=-1)
            jets = sequence.inclusive_jets(ptmin=20)
            self.allJets = self.convertJetsToDType(jets)
            pass
        pass

    #calculate and append variables to event
    def calculateVariables(self):
        #invariant mass of leading and subleading jet
        from tools import calcMjj
        self.variables["mjj"]=calcMjj(self.allJets)
        #pseudorapidity gap leading jets
        from tools import dEtajj
        self.variables["dEtajj"]=dEtajj(self.allJets)
        #delta phi leading jets
        from tools import dPhijj
        self.variables["dPhijj"]=dPhijj(self.allJets)
        #delta R leading jets
        from tools import dRjj
        self.variables["dRjj"]=dRjj(self.allJets)
        #delta Pt leading jets
        from tools import dPtjj
        self.variables["dPtjj"]=dPtjj(self.allJets)
        #scalar sum pt all jets
        from tools import scalarSumAllJetPt
        self.variables["sumPtAllJets"]=scalarSumAllJetPt(self.allJets)
        #scalar sum pt dijets
        from tools import scalarSumDiJetPt
        self.variables["sumPtjj"]=scalarSumDiJetPt(self.allJets)
        #vector sum pt all jets
        from tools import vectorSumAllJetPt
        self.variables["vecSumPtAllJets"]=vectorSumAllJetPt(self.allJets)
        #vector sum pt dijets
        from tools import vectorSumDiJetPt
        self.variables["vecSumPtjj"]=vectorSumDiJetPt(self.allJets)
        
        self.variables["nJets"]=len(self.allJets)

#* hadron content - pt sum, harder hadrons?
#* nr of hadrons in all jet, lead jet
#* jet dR vs jet mass?
#* substructure quantities? N-subjettiness, Ratio of Energy Correlation functions D2
#* ...


        return

if __name__=="__main__":
    import pandas
    nEvts=20000
    fn =  './data/events_anomalydetection.h5'
    # Option 2: Only load the first 10k events for testing
    df_test = pandas.read_hdf(fn, stop=nEvts)
    f=df_test
    events = f.T
    np.shape(events)

    evtContainer=EventContainer()
    evtContainer.readEvents(events,nEvts)
    print(evtContainer.numEvents)
    import pickle
    pklFile=open("./wtruth_20k_constituents.pkl",'wb')
    pickle.dump( evtContainer , pklFile)



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
