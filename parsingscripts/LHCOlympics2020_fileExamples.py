#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division

import pandas
import matplotlib.pyplot as plt

import h5py


# In[2]:


# path to file
fn =  '/Users/gregor/Dropbox/AnomalyDetection/events_anomalydetection.h5'


# In[3]:


# Option 1: Load everything into memory
df = pandas.read_hdf(fn)
print(df.shape)
print("Memory in GB:",sum(df.memory_usage(deep=True)) / (1024**3))


# In[5]:


# Option 2: Only load the first 10k events for testing
df_test = pandas.read_hdf(fn,stop=10000)
print(df_test.shape)
print("Memory in GB:",sum(df_test.memory_usage(deep=True)) / (1024**3))


# In[6]:


# Option 3: Use generator to loop over the whole file
def generator(filename, chunksize=512,total_size=1100000):

    i = 0
    
    while True:

        yield pandas.read_hdf(filename,start=i*chunksize, stop=(i+1)*chunksize)

        i+=1
        if (i+1)*chunksize > total_size:
            i=0

gen = generator(fn)


# In[7]:


gen.next()

