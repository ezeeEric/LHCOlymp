"""
Base class for building a dataset
"""
## stdlib
import os, re

## PyPI
import yaml
import numpy as np
import pandas as pd

## local
# from src.yamlUtils import Serializable

from yamlable import YamlAble, yaml_info


##<<--------------------------------------------------------------------------------
## 
##<<--------------------------------------------------------------------------------

@yaml_info(yaml_tag_ns="Dataset")
class Dataset(YamlAble):
    """
    """

    def __init__(self, name="DS", localityFeature=None, regionSpan=[], path=None, labelData=False, sidebandMargin=0.25, pathExt="h5"):
        """
        Parameters
        ----------
        localityFeature: str, 
            A feature that is used to create this dataset (region).
        regionSpan: list, 
            (min, max) for the localityFeature. 
        """

        self.name = name
        self.localityFeature = localityFeature
        self.regionSpan = regionSpan
        self.sidebandMargin = sidebandMargin
        self.labelData = labelData
        self.path = path        

        ## determine file format
        self.pathExt = None
        regX = re.compile("^.*\.(?P<ext>\w+)$")
        match = re.match(regX, self.path)
        if match:
            self.pathExt = match.group("ext")

    def __repr__(self):
        return ("%s(name=%r, localityFeature=%r, regionSpan=%r, sidebandMargin=%r, path=%r, pathExt=%r)")%(
            self.__class__.__name__, self.name, self.localityFeature, self.regionSpan, self.sidebandMargin, self.path, self.pathExt
        ) 


    def load(self):
        """ load the dataset from disk
        """
        if self.path==None or os.path.isfile(self.path)==False:
            raise IOError("Input is not provided")

        if self.pathExt=="h5" or self.pathExt=="hdf":
            dframe = pd.read_hdf(self.path)
        elif self.pathExt=="csv":
            dframe = pd.read_csv(self.path)
        elif self.pathExt=="pkl":
            dframe = pd.read_pickle(self.path)
        else:
            raise NotImplementedError("Reading stream of %s is not yet implemented!"%self.pathExt)
        
        return dframe 
                
    def write(self, dframe):
        """ write the dataset to disk
        """
        if self.pathExt=="h5" or self.pathExt=="hdf":
            dframe.to_hdf(self.path, key="df", mode="w")
        elif self.pathExt=="csv":
            dframe.to_csv(self.path)
        elif self.pathExt=="pkl":
            dframe.to_pickle(self.path)
        else:
            raise NotImplementedError("Writing stream of %s is not yet implemented!"%self.pathExt)
        

    def weights(self):
        """ weights for the target and sideband regions 
        """
        pass

    def balanceRegions(self):
        """ Balance weights for the target and sideband regions.
        """
        raise NotImplementedError
        
    def targetRegion(self, regionLabelName="RegionLabel"):
        """ get the target region ( if labeled data the class with the smaller number of data points) with slice number sliceNum
        """
        dataFrame = self.load()
        span = self.regionSpan[1] - self.regionSpan[0]

        rl = self.regionSpan[0] + span*self.sidebandMargin
        rh = self.regionSpan[1] - span*self.sidebandMargin

        targetDframe = dataFrame.loc[((rl <= dataFrame[self.localityFeature]) & (dataFrame[self.localityFeature] <= rh))].copy(deep=True)
        ## release memory
        del dataFrame

        colSize = targetDframe.shape[0]
        targetDframe[regionLabelName] = np.ones(colSize)

        return targetDframe 

    def sidebandRegion(self, regionLabelName="RegionLabel"):
        """ get the sideband region (usually around the target region with some margins) 
        """
        rl, rh = self.regionSpan[0], self.regionSpan[1]        
        span = self.regionSpan[1] - self.regionSpan[0]
        r1 = self.regionSpan[0] + span*self.sidebandMargin
        r2 = self.regionSpan[1] - span*self.sidebandMargin

        dataFrame = self.load()

        mask = ((rl <= dataFrame[self.localityFeature]) & (dataFrame[self.localityFeature]<= r1)\
            |(r2 <= dataFrame[self.localityFeature]) & (dataFrame[self.localityFeature] <= rh))
        
        sidebandDframe = dataFrame[mask].copy(deep=True)
        ## release memory
        del dataFrame

        colSize = sidebandDframe.shape[0]
        sidebandDframe[regionLabelName] = np.zeros(colSize)


        return sidebandDframe

    def splitTrainValData(self, valFrac=0.2):
        """ keep a fraction of valFrac for optimizing models.
        """
        raise NotImplementedError



##<<--------------------------------------------------------------------------------
## 
##<<--------------------------------------------------------------------------------
def datasetConstructor(loader, node):
    """ helper for setting up the yaml default ctor
    """
    kwargs = loader.construct_mapping(node)
    try:
        return Dataset(**kwargs)
    except:
        fields = '\n'.join('%s = %s' % item for item in kwargs.items())
        log.error("unable to load dataset %s with these fields:\n\n%s\n" %
                  (kwargs['name'], fields))
        raise

# yaml.add_constructor(u'!Dataset', datasetConstructor)
