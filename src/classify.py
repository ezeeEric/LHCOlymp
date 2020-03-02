"""
## Ref: https://arxiv.org/pdf/1902.02634.pdf

- Split dataset into 5 subsets stratified by mres binning for subseti in subsets do
    - Set aside subseti as test data for subsetj in subsets, j ̸= i do
        - Validation data sideband = merge sideband bins of subsetj
        - Validation data signal-region = merge signal-region bins of subsetj
        - Training data sideband = merge sideband bins of remaining subsets Training data signal-region = merge signal-region bins of remaining subsets Assign signal-region data with label 1
    - Assign sideband data with label 0
    - Train twenty classifiers on training data, each with different random
    initialization
    modeli,j = best of the twenty models, as measured by performance on
    validation data
end 􏰆
modeli = j modeli,j/4

- Select Y % most signal-like data points of subseti, as determined by modeli. The
threshold on the neural network to achieve Y % is determined using all other bins
with large numbers of events and so the uncertainty on the value is negligible.
end
- Merge selected events from each subset into new mres histogram
- Fit smooth background distribution to mres distribution with the signal region masked Evaluate p-value of signal region excess using fitted background distribution
interpolated into the signal region.
"""

## stdlib
import os, sys, re, random
import pickle

##PyPI
import numpy as np
import pandas as pd
# import tensorflow
# ## supress tf warning messages
# tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import to_categorical 

from sklearn.model_selection import train_test_split


## local
from src.dataset import Dataset



##<<--------------------------------------------------------------------------------
## 
##<<--------------------------------------------------------------------------------
class kCallBack(Callback):
    """  Keras custom callback 
    """
    def __init__(self):
        super(kCallBack, self).__init__()


    def on_train_begin(self, logs=None):
        return super().on_train_begin(logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        return super().on_epoch_end(epoch, logs=logs)
    

##<<--------------------------------------------------------------------------------
## 
##<<--------------------------------------------------------------------------------
class NNClassifier(Sequential):
    """ Neural Network based classifier. 
    """

    def __init__(self,  dataset=None, features=[], name="NN", layers=[], outDir="./", kFolds=5, modelPrefix=None, compileParams={}):
        """
        Parameters
        ----------
        name: str, 
            a name for the model.
        layers: list of keras.Layer instances,
            to build the model.
        dataset: Dataset instance, see dataset.py,
            Container for training/validation/test data.

        outDir: str,
            the output diretory that models will be written to.        
        """

        ## instantiate the base class 
        super(NNClassifier, self).__init__(name=name, layers=layers)
        self.dataset = dataset
        self.features = features
        self.outDir = outDir
        self.kFolds = kFolds
        self.modelPrefix = modelPrefix
        self.compileParams = compileParams

    # def __repr__(self):
    #     pass

    def props(self):
        """ Selected model properties (to write to disk)
        """
        modelProps = {}
        modelProps["dataset"] = self.dataset
        modelProps["features"] = self.features
        modelProps["outDir"] = self.outDir
        modelProps["kFolds"] = self.kFolds
        modelProps["modelPrefix"] = self.modelPrefix
        modelProps["compileParams"] = self.compileParams

        return modelProps

    def add(self, layer):
        return super().add(layer)

    def compile(self):
        optimizer = self.compileParams.get("optimizer", "adam")
        super().compile(**self.compileParams)

    # def compile(self, optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None, **kwargs):
    #     return super().compile(optimizer, 
    #     loss=loss, metrics=metrics, loss_weights=loss_weights, 
    #     sample_weight_mode=sample_weight_mode, weighted_metrics=weighted_metrics, target_tensors=target_tensors, **kwargs)

    def fit(self, 
        batch_size=None, 
        epochs=1, 
        verbose=1, 
        callbacks=None, 
        # validation_split=0.0, 
        # validation_data=None, 
        shuffle=True, 
        # class_weight=None, 
        # sample_weight=None, 
        # initial_epoch=0, 
        # steps_per_epoch=None, 
        # validation_steps=None, 
        **kwargs):
        """
        1. the dataset is randomly partitioned bin-by-bin into kFolds groups. 
        2. for each group i ∈ {1, 2, 3, 4, 5} (the test set), an ensemble classifier is trained on the remaining groups j ̸= i. There are four ways to split the four remaining groups into three for training and one for validation. 
        3. For each of these four ways, many classifiers are trained and the one with best validation performance is selected. 
        4. The ensemble classifier is then formed by the average of the four selected classifiers (one for each way to assign the training/validation split). 
        5. Data are selected from each test group using a threshold cut from their corresponding ensemble classifier. The selected events are then merged into a single histogram.

        """

        ## get sideband and target regions (with assigned RegionLables)
        regionLabelName = "RegionLabel"
        sidebandRegion = self.dataset.sidebandRegion(regionLabelName=regionLabelName)
        targetRegion = self.dataset.targetRegion(regionLabelName=regionLabelName)

        ## merge into a single dataframe 
        dataFrame = pd.concat([sidebandRegion, targetRegion])

        ## get matrices ready for training 
        feats = [f for f in self.features]
        x = dataFrame[feats].values
        y = dataFrame[regionLabelName].values
        
        ## kFold cross validation 
        indices = range(y.shape[0])
        for k in range(2, 3):#1+self.kFolds):
            trainMask = []
            for idx in indices:
                if idx%k!=0:
                    trainMask += [idx]
            xTrain = x[trainMask]
            yTrain = y[trainMask]
            ## one-hot encoding
            yTrain = to_categorical(yTrain)
            
            ##@FIXME validation set 
            xVal = xTrain
            yVal = yTrain 

            history = super().fit(
                x=xTrain, 
                y=yTrain, 
                batch_size=batch_size, 
                epochs=epochs, 
                verbose=verbose, 
                callbacks=callbacks, 
                # validation_split=validation_split, 
                # validation_data=validation_data, 
                shuffle=shuffle, 
                # class_weight=class_weight, 
                # sample_weight=sample_weight, 
                # initial_epoch=initial_epoch, 
                # steps_per_epoch=steps_per_epoch, 
                # validation_steps=validation_steps, 
                **kwargs)

            print(self.summary())

            print(self.predict(xVal))

            ##@FIXME recommended safe way to save SubClassed Models
            ## https://colab.research.google.com/drive/172D4jishSgE3N7AO6U2OKAA_0wNnrMOq#scrollTo=OOSGiSkHTERy
            modelWeights = os.path.join(self.outDir, "Model_%s_Weights_kFold_%i"%(self.name, k))
            self.save_weights(modelWeights, save_format='tf')

            ## save the loss on the first batch to later assert that the optimizer state was preserved
            firstLoss = self.train_on_batch(xTrain[:batch_size], yTrain[:batch_size])

            """To restore your model, you will need access to the code that created the model object.
                Note that in order to restore the optimizer state and the state of any stateful metric, 
                you should compile the model (with the exact same arguments as before) and call it on some data before calling load_weights:
            """

            ## model Props
            modelProps = os.path.join(self.outDir, "Model_%s_Properties_kFold_%i.pkl"%(self.name, k))
            with open(modelProps, "wb") as pfile:
                props = self.props()
                props["firstLoss"] = firstLoss #<! add first loss 
                pickle.dump(props, pfile)
            
        return None


    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None):
        return super().evaluate(x=x, y=y, batch_size=batch_size, verbose=verbose, sample_weight=sample_weight, steps=steps)


    def initVariables(self):
        ## get sideband and target regions (with assigned RegionLables)
        regionLabelName = "RegionLabel"
        sidebandRegion = self.dataset.sidebandRegion(regionLabelName=regionLabelName)
        targetRegion = self.dataset.targetRegion(regionLabelName=regionLabelName)

        ## merge into a single dataframe 
        dataFrame = pd.concat([sidebandRegion, targetRegion])

        ## get matrices ready for training 
        feats = [f for f in self.features]
        x = dataFrame[feats].values
        y = dataFrame[regionLabelName].values
        y = to_categorical(y)

        # This initializes the variables used by the optimizers,
        # as well as any stateful metric variables
        self.train_on_batch(x[:1], y[:1])

        return 


##--------------------------------------------------------------------------
## util for parallel processing
##--------------------------------------------------------------------------
def trainModel(model, 
    train_df=None,
    features=[], 
    balanced=True,
    outdir="", 
    balanceClasses=False, 
    scaleFeatures=True,
    normalizeFeatures=True, 
    saveModel=True, 
    overWrite=False):
    """ Train a classifier and produce some validation plots.
    Parameters
    ----------
    model: DecisionTree;
        sklearn classifier 
    train_df: pandas.DataFrame; 
        training dataframe  
    outdir: str;
        path to save the model and roc curve 
    balanceClasses: bool;
        whether to use the sample weight or go with the sklearn's default balanced weight procedure for the classes. 
    save_model: bool;
        whether to save the trained model to disk or not

    Return
    ------
    trained model 
    """
    if balanceClasses:
        balanced = False

    if train_df is None:
        tr_df = model.train_df
    if not features :
        features = model.features

    if balanced: 
        balanceClasses = False   
        b_df = tr_df[tr_df["class_label"]==0]
        s_df = tr_df[tr_df["class_label"]==1]
        log.info("Balancing training classes (under-sampling); signal events:{} | bkg events: {}".format(s_df.shape[0], b_df.shape[0]))
        b_df = b_df.sample(s_df.shape[0], replace=True)
        tr_df_ud = pd.concat([b_df, s_df], axis=0)

    ## - - training arrays
    X_train = tr_df_ud[[ft.name for ft in features]]    
    Y_train = tr_df_ud["class_label"]
    if balanceClasses:
        log.info("Using event weight for training, events with negative weight are thrown away")        
        X_weight = tr_df_ud["weight"] 
        ## in order to avoid events with negative weight thrown away        
        if positive_weights: 
            log.info("using abs(weights)!")
            X_weight = np.absolute(X_weight.values)

    if scale_features:
        log.info("Scaling features using StandardScaler ...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.values)

    mpath = os.path.join(outdir, model.name)
    is_trained = False 
    if os.path.isfile(mpath):
        log.info("Loading %s model from disk ..."%mpath)
        with open(mpath, "r") as cache:
            model = pickle.load(cache)
            is_trained =  model.is_trained
            if is_trained:
                log.warning("The %s model is already trained! set overwrite=True, if you want to overwrite it"%mpath)
                if overwrite:
                    os.remove(mpath)
                    is_trained = False                                    

    if not is_trained:
        ## train the model,
        model = model.fit(X_train.values, Y_train.values, sample_weight=X_weight if balanceClasses else None)
        model.is_trained = True
        if save_model:
            mpath = os.path.join(outdir, model.name)
            with open(mpath, "w") as cache:
                pickle.dump(model, cache, protocol=2)
    
    log.info("Trained %s model"%(model.name))
    return model

