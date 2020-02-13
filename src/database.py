# stdlib
import os, sys, re, glob
from operator import itemgetter
import logging
import atexit
import fnmatch
from collections import namedtuple
from  multiprocessing import Pool, cpu_count

# PyPI
import yaml
import numpy as np
import pandas as  pd


# local
from src import log
from src.decorators import cached_property
from src.yaml_utils import Serializable
from src.dataset import Dataset


## Dataset types 
UNLABELED, SINGLE, LABELED = 0, 1, 2

## this files path
__HERE = os.path.dirname(os.path.abspath(__file__))


##<<--------------------------------------------------------------------------------
## 
class Database(dict):
    """ base class for Database utils

    Attributes
    ----------
    name: str, 
      
    verbose: bool,
      
    stream: out stream (default=sys.stdout)
    """

    def __init__(self, name='DB', dbFile=None, outDir="", version="", verbose=False, stream=sys.stdout, outExt=".h5",
        inputPath=None, localityFeature="", numSlices=1, localityFeatureRange=(), sideBandMargin=0.3, logLevel="INFO", ):
        """ 
        """

        super(Database, self).__init__()
        self.name = name
        self.verbose = verbose
        self.stream = stream
        self.version = version
        self.outExt = outExt
        self.dbFile = dbFile
        self.outDir = outDir
        self.localityFeature = localityFeature
        self.inputPath = inputPath
        self.numSlices = numSlices
        self.localityFeatureRange  = localityFeatureRange
        self.sideBandMargin = sideBandMargin

        ## determine file format
        self.pathExt = None
        regX = re.compile("^.*\.(?P<ext>\w+)$")
        match = re.match(regX, self.inputPath)
        if match:
            self.pathExt = match.group("ext")

        ## setup logging 
        log.setLevel(logLevel)
        log.debug("Initialzing Database ...")

        ## where to put the database yml file
        # if dbFile is None:
        #     dbFile = __HERE 
        # self.dbFile = os.path.join(dbFile, '%s%s.yml' % (self.name, self.version))
        if os.path.isfile(self.dbFile):
            with open(self.dbFile) as db:
                log.info("Loading database '%s' ..." % self.dbFile)
                d = yaml.load(db)
                if d:
                    self.update(d)
        self.modified = False


    def __setitem__(self, name, ds):
        """
        """
        super(Database, self).__setitem__(name, ds)

    def load(self):
        """ load the dataset from disk
        """
        if self.inputPath==None or os.path.isfile(self.inputPath)==False:
            raise IOError("Input is not provided!")

        
        if self.pathExt=="h5" or self.pathExt=="hdf":
            dframe = pd.read_hdf(self.inputPath, stop=100)
        elif self.pathExt=="csv":
            dframe = pd.read_csv(self.inputPath)
        elif self.pathExt=="pkl":
            dframe = pd.read_pickle(self.inputPath)
        else:
            raise NotImplementedError("Reading stream of %s is not yet implemented!"%self.pathExt)
        
        return dframe 

    def scan(self, dtype=LABELED):
        """ 
        """
        if len(self.localityFeatureRange) < 2:
            raise IndexError("Locality feature range is not provided properly!")
        
        _min, _max = self.localityFeatureRange
        bins = np.linspace(_min, _max, num=self.numSlices)

        ## Load the full data frame
        dframe = self.load()

        ## get slices 
        for i in range(len(bins)-1):
            rl, rh = bins[i], bins[i+1]
            filterStr = "{0} <= {1} <= {2}".format(rl, self.localityFeature, rh)
            subDframe = dframe[ ((rl <= dframe[self.localityFeature]) & (dframe[self.localityFeature] <= rh))]

            
            ## create dateset
            parent = self.inputPath.split("/")[-1]
            parent = parent.replace("."+self.pathExt, "")
            dsName = "DATASET_SLICE_{}_FROM_{}".format(i, parent)
            dsPath = os.path.join(self.outDir, dsName+self.outExt) 
            ds = Dataset(name=dsName, localityFeature=self.localityFeature, regionSpan=(rl, rh), path=dsPath, sideBandMargin=self.sideBandMargin)
            print(subDframe)
            ds.write(subDframe)
            self[dsName] = ds 

        self.modified = True


    def write(self):
        """ write database to yml file
        """
        if self.modified:            
            with open(self.dbFile, 'w') as db:
                log.info("Saving database '%s' ..." % self.name)
                yaml.dump(dict(self), db)
            

    def reset(self):
        """cleanup
        """
        return self.clear()

    def clear(self):
        """erase all datasets in database
        """
        log.info("Resetting database '%s' ..." % self.name)
        super(Database, self).clear()
        self.modified = True

