#!/usr/bin/env python
# coding: utf-8

# # Boosted Decision Tree

# In[1]:


import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


# In[2]:


## load dataset
df = pd.read_pickle("../data/dataFrames/df_dev_const_wtruth_20k_subjettiness.pkl")


# In[3]:


params={'n_estimators': 100, 
        'learning_rate': 0.1, 
        'max_depth':10, 
        'min_samples_leaf':0.005, 
        'min_samples_split':0.01
       }
model=GradientBoostingClassifier(**params)


# In[4]:


def train_model(model, 
    train_df=None,
    features=[], 
    positive_weights=True, 
    balanced=True,
    outdir="", 
    weight_sample=False, 
    scale_features=True, 
    save_model=True, 
    overwrite=False):
    """ Train a classifier and produce some validation plots.
    Parameters
    ----------
    model: DecisionTree;
        sklearn classifier 
    train_df: pandas.DataFrame; 
        training dataframe  
    outdir: str;
        path to save the model and roc curve 
    weight_sample: bool;
        whether to use the sample weight or go with the sklearn's default balanced weight procedure for the classes. 
    save_model: bool;
        whether to save the trained model to disk or not

    Return
    ------
    trained model 
    """
    tr_df=train_df
    if weight_sample:
        balanced = False

    if train_df is None:
        tr_df = model.train_df
    if not features :
        features = model.features


    if balanced: 
        weight_sample = False   
        b_df = tr_df[tr_df["isSignal"]==0]
        s_df = tr_df[tr_df["isSignal"]==1]
        print("Balancing training classes (under-sampling); signal events:{} | bkg events: {}".format(s_df.shape[0], b_df.shape[0]))
        b_df = b_df.sample(s_df.shape[0], replace=True)
        tr_df_ud = pd.concat([b_df, s_df], axis=0)

    ## - - training arrays
    X_train = tr_df_ud[[ft for ft in features]]    
    Y_train = tr_df_ud["isSignal"]
   # if weight_sample:
    if False:
        print("Using event weight for training, events with negative weight are thrown away")        
        X_weight = tr_df_ud["weight"] 
        ## in order to avoid events with negative weight thrown away        
        if positive_weights: 
            X_weight = np.absolute(X_weight.values)

   # if scale_features:
    if False:
        print("Scaling features using StandardScaler ...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.values)

    is_trained = False       
    
    if not is_trained:
        ## train the model,
        model = model.fit(X_train.values, Y_train.values, sample_weight=X_weight if weight_sample else None)    
    return model


# In[5]:


trainedModel=train_model(model=model,train_df=df,features=['mjj','dEtajj'])


# In[6]:


X_train = df[['mjj','dEtajj']]
Y_train = df['isSignal']
trainedModel.score(X_train,Y_train)


# In[ ]:




