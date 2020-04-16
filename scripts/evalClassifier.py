#!/usr/bin/env python 
import pickle
from src.classify import NNClassifier 



# Recreate the exact same model, including its weights and the optimizer

## load model props
with open("Model_NN_Properties_kFold_2.pkl", "rb") as pfile:
    props = pickle.load(pfile)
            
print(props)
props.pop("firstLoss", None)
model = NNClassifier(**props)
model.compile()
model.initVariables()
model.load_weights("Model_NN_Weights_kFold_2")
model.summary()

# # Show the model architecture
# new_model.summary()