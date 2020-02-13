"""
Base class for building a dataset
"""
import os, re

import yaml

from src.yaml_utils import Serializable



##<<--------------------------------------------------------------------------------
## 
##<<--------------------------------------------------------------------------------
class Dataset(Serializable):
    """
    """

    def __init__(self, name="DS", localityFeature=None, regionSpan=None, path=None, labelData=False, sideBandMargin=0.3):
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
        self.labelData = labelData
        self.path = path        

        ## determine file format
        self.pathExt = None
        regX = re.compile("^.*\.(?P<ext>\w+)$")
        match = re.match(regX, self.path)
        if match:
            self.pathExt = match.group("ext")

    def __repr__(self):
        return "(%s(name=%r, localityFeature=%r, regionSpan=%r, path=%r))"%(
            self.__class__.__name__, self.name, self.localityFeature, self.regionSpan, self.path)

    def info(self):
        """
        """
        pass

    def load(self, ftype="h5"):
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
        
    def targetRegion(self):
        """ get the target region ( if labeled data the class with the smaller number of data points) with slice number sliceNum
        """

        pass

    def getSidebandRegion(self, sliceNum):
        """ get the sideband region (usually around the target region with some margins) 
        """
        raise NotImplementedError

    def splitTrainValData(self, valFrac=0.2):
        """ keep a fraction of valFrac for optimizing models.
        """
        raise NotImplementedError

    def __repr__(self):
        pass



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

yaml.add_constructor(u'!Dataset', datasetConstructor)
