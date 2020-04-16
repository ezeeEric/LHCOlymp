""" main engine for features extraction, validation, dimensional reduction, etc.
"""

##<<--------------------------------------------------------------------------------
## 
##<<--------------------------------------------------------------------------------
class Feature:
    """ base class for Features.

    Attributes
    ----------
    name: str, varibale's name
    unit: str, variable's unit
    title: str, variable's title for x axis on histograms
    label: str, label to be shown on the plot(legend)
    binning: tupel, varibale's binning 
    
    Examples
    --------
    >>> tau_0_pt = Variable(
    "tau_0_pt",
    title='#font[52]{p}_{T}(#tau_{1}) [GeV]',
    binning=(20, 0, 400),
    unit='[GeV]',
    scale=0.001)

    """
    
    def __init__(
            self, name, 
            unit="",
            title=None,
            label=None, 
            scale=None,
            binning=None,
            bins=None,
            scale=1.,
            **kwargs):
        
        self.name = name
        self.title = title
        self.unit = unit
        self.scale = scale
        self.binning = binning
        self.bins = bins


    def __repr__(self):
        return "Feature:: name=%r, binning=%r, scale=%r, "%(self.name, self.binning, self.scale)



##<<--------------------------------------------------------------------------------
## 
##<<--------------------------------------------------------------------------------
class FeatureEngine:
    """
    """
    
    def __init__(self, datasets, featuresList=[], localityFeature=None, outDir="./", ):
        """
        Parameters
        ----------
        datasets: list of Dataset type, 
            see (dataset.py)
        featuresList: list of Feature type, 
            see class Feature        
        """
        self.datasets = datasets
        self.featuresList = featuresList
        self.localityFeature = localityFeature
        self.outDir = outDir

    def __repr__(self):
        raise NotImplementedError

    
    def scatterPlots(self, ncols=3, nrows=3):
        """
        """
        raise NotImplementedError

    def reduceDimension(self, targetDims=10):
        raise NotImplementedError

    def plotReducedDims(self):
        raise NotImplementedError

    def plotCorrelations(self):
        """
        """
        raise NotImplementedError
    
    def decorrelateLocalityFeature(self):
        """
        """
        raise NotImplementedError