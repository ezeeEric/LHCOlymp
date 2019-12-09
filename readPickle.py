#quick test to read pickled event container
import pickle
from eventClass import EventContainer,JetEvent

#data/events_LHCO2020_backgroundMC_Pythia.pkl
#data/events_anomalydetection.pkl

#fP="./data/events_LHCO2020_BlackBox1.pkl"
fP="./data/events_LHCO2020_backgroundMC_Pythia.pkl"
fP="./wtruth_20k.pkl"
f=open(fP, "rb")
evtCont=pickle.load(f)
df=evtCont.toDataFrame()

pklFile=open('./testDF.pkl','wb')
pickle.dump( df, pklFile)
pklFile.close()
