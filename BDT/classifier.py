from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

import logging
log = logging.getLogger(__name__)


##----------------------------------------------------------------------------------
## Base classifier class
##----------------------------------------------------------------------------------
class BDTClassifier(GradientBoostingClassifier):
    
    def __init__(self, channel, 
                    name="SKGBoost", 
                    features=[], 
                    train_df=None, 
                    weight_sample=False,
                    is_trained=False,
                    kfolds=5,
                    fold_num=0,
                    **params):
        log.debug("Initializing BDTClassifier ...")
        self.kparams = params
        self.channel = channel
        self.name = name 
        self.features = features
        self.train_df = train_df
        self.weight_sample = weight_sample
        self.kfolds = kfolds
        self.fold_num = fold_num
        self.is_trained = is_trained
        self.hyperparams = params

        ## instantiate the base
        super(BDTClassifier, self).__init__(**params)

    @staticmethod
    def prepare_data(backgrounds, signals, features,
                     branches=[],
                     category=None,
                     channel="taujet",
                     treename="NOMINAL",
                     train_data="CLF_DATA.pkl",
                     data_lumi=36200, 
                     overwrite=False,
                     truth_match_tau=True):
        """Training input as pandas DataFrame
        """

        ## first check if all the samples are already available in the training Dataframe
        cached_dframe = None
        if os.path.isfile(train_data) and not overwrite:
            log.info("Reading training data from %s"%train_data)
            with open(train_data, "r") as cache:
                cached_dframe = cPickle.load(cache)
            for b in backgrounds:
                if not (b.name in cached_dframe.index):
                    log.warning("missing %s in %s Dataframe"%(b.name, train_data))
                    missing_bkgs += [b]
                    
            for s in signals:
                if not (s.name in cached_dframe.index):
                    log.warning("missing %s in %s Dataframe"%(s.name, train_data))
                    missing_sigs += [s]
            if not missing_sigs and not missing_bkgs:
                log.info("All requested samples are available in %s Dataframe"%train_data)
                return cached_dframe
            
        ## feature name as column label
        columns = {}
        for feat in features:
            columns[feat.tformula] = feat.name

        ## concat sig & bkg and index based on the sample name
        m_keys = [bkg.name for bkg in missing_bkgs] + [sig.name for sig in missing_sigs]
        new_dframe = pd.concat(bkg_dfs+sig_dfs, keys=m_keys, sort=False)

        if cached_dframe is not None:
            keys = [sb.name for sb in backgrounds+signals]
            dframe = pd.concat([new_dframe, cached_dframe], sort=False)
        else:
            dframe = new_dframe   

        if overwrite:
            log.warning("caching training data")
            os.system("rm %s"%train_data)
            
        with open(train_data, "a") as cache:
            cPickle.dump(dframe, cache, protocol=2)
            
        return dframe
    
    ##--------------------------------------------------------------------------
    ## utility function for optimizing hyprparameters of a model
    ##--------------------------------------------------------------------------
    def optimize(self, X_train=None, Y_train=None, X_weight=None, param_grid={},
                    outdir="", weight_sample=False, save_model=True, validation_plots=False):
        """ Tune model's hyper parameters and return the best performing alongside the model.
        Parameters
        ----------
        see train_model() 
        """
    
        if X_train is None:
            X_train = model.X_train
        if X_weight is None:    
            X_weight = model.X_weight
        if Y_train is None:        
            Y_train = model.Y_train 
    
        gb_clf = GradientBoostingClassifier()
    
        # parameters to be passed to the estimator's fit method (gb_clf)
        fit_params = {"sample_weight": X_weight if weight_sample else None}
    
        # run grid search
        grid_search = GridSearchCV(
            gb_clf, param_grid=param_grid, fit_params=dict(fit_params), cv=3, n_jobs=N_OPT_CORES, verbose=3, scoring="roc_auc",  return_train_score=False)
        start = time.time()
        grid_search.fit(X_train, Y_train)
    
        print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
            % (time.time() - start, len(grid_search.cv_results_['params'])))
    
        report_name = os.path.join(outdir, model.name.replace(".pkl", "_HyperParams.TXT"))
        with open(report_name, "w") as rfile:
            rfile.write("%s\n"%model.name.replace(".pkl", ""))
            rfile.write(report(grid_search.cv_results_))
    
        return grid_search.cv_results_


if __name__=="__main__":
    bdtClassifier=BDTClassifier(channel='test')
