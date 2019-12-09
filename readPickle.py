#quick test to read pickled event container
import pickle
from eventClass import EventContainer,JetEvent

fP="./test.pkl"

f=open(fP, "rb")
evtCont=pickle.load(f)
print(evtCont)
