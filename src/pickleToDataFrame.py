#quick test to read pickled event container
import pickle
from eventClass import EventContainer,JetEvent
import os
#data/events_LHCO2020_backgroundMC_Pythia.pkl
#data/events_anomalydetection.pkl

fileList=["events_anomalydetection.pkl","events_LHCO2020_BlackBox1.pkl","events_LHCO2020_backgroundMC_Pythia.pkl"]
fileList=["wtruth_20k_constituents.pkl"]
fileList=["wtruth_1k_subjettiness.pkl"]
pathPrefix="./"

for fP in fileList:
    f=open(os.path.join(pathPrefix,fP), "rb")
    evtCont=pickle.load(f)
    df=evtCont.toDataFrame()
    pklFile=open('./data/dataFrames/df_dev_const_{0}'.format(fP),'wb')
    pickle.dump( df, pklFile)
    pklFile.close()
    f.close()
