#! /usr/bin/env python 

import os

import tensorflow as tf
## turn off warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
tf.executing_eagerly()

# import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout

import yaml

from src.dataset import Dataset
from src.classify import NNClassifier 


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

## load datasets
with open("/project/6024950/sbahrase/LHC/LHCOlymp/tstDB.yml", "rb") as yfile:
    db = yaml.load(yfile, Loader=yaml.UnsafeLoader)

for name, dsInfo in db.items():
    ds = Dataset(**dsInfo.__dict__)
    df = ds.load()


features = ["mjj", "dEtajj", "dPhijj", "dRjj"]
target = "isSignal"

layers = [
    Dense(512, activation='relu', input_shape=(4, )),
    Dropout(0.2),
    Dense(2, activation="softmax"),
]

compileParams = {
    "optimizer": "adam",
    "loss": tf.keras.losses.BinaryCrossentropy(), 
    "metrics": ['accuracy'],
}

model = NNClassifier(ds, features, layers=layers, compileParams=compileParams)

model.compile()
model.fit(epochs=10)

