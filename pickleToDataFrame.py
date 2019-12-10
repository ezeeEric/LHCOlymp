#quick test to read pickled event container
import pickle
from eventClass import EventContainer,JetEvent
import os
#data/events_LHCO2020_backgroundMC_Pythia.pkl
#data/events_anomalydetection.pkl

fileList=["events_anomalydetection.pkl","events_LHCO2020_BlackBox1.pkl","events_LHCO2020_backgroundMC_Pythia.pkl"]

pathPrefix="./data/"

for fP in fileList:
    f=open(os.path.join(pathPrefix,fP), "rb")
    evtCont=pickle.load(f)
    df=evtCont.toDataFrame()
    pklFile=open('./dataExamples/df_{0}'.format(fP),'wb')
    pickle.dump( df, pklFile)
    pklFile.close()
    f.close()
