# LHCOlymp

See [Overview Notebook](notebooks/overview.ipynb) for full details 
on [LHCOlympics2020](https://indico.cern.ch/event/809820/page/19002-lhcolympics2020) challenge. 

### Package Requirements (to be specified):
- pyjet
- tables
- sklearn
- tensorflow
- keras


## Current Dataset
### Event Samples

- 1M background events, 100k signal events with truth label:

```
events_anomalydetection.h5
```

- 1M QCD Multijet events (Pythia8)

```
events_LHCO2020_backgroundMC_Pythia
```

- 1M Data events with injected signal

```
events_LHCO2020_BlackBox1
```


### Anti-kt R=1.0 jet collections post-clustering
- stored in dedicated event format
[URL](https://cernbox.cern.ch/index.php/s/MJSWVW6I8iPRNrB)
or
```
/eos/user/e/edrechsl/LHCOlympics2020/191217_antikt_1M_wSubstructure
```

### DataFrames produced from above files
- use this if you don't want to add further variables
[URL](https://cernbox.cern.ch/index.php/s/4m2tVBp7GDaIMY8)
or 
```
/eos/user/e/edrechsl/LHCOlympics2020/dataFrames/191217_antikt_1M_wSubstructure
```

## Analysis Flow

### Data Preparation
#### 1. Hadrons to Events with Jet collections

- Use pyjet (fastjet) library to cluster hadrons in input files to jets

  - store events with clustered jet collections in `JetEvent` class objects
  - single jet is numpy dtype with limited, specified variables read out from FastJet::PSeudoJet object
  - **extend this dtype to add further variables from single jet**

- each event carries dict of compound variables, extending from single-jet properties

- all events are stored in single `EventContainer`

```python
#Example run command
#truth flag optional, only relevant if signal/background dataset
python main.py -i /project/6001411/edrechsl/lhcolympics/data/events_anomalydetection.h5 -truth
```

#### 2. Event Container to DataFrame

- `EventContainer` class transformed to pandas DataFrame
- built-in function `EventContainer.toDataFrame()`
- stores compound variables by default on top of various `JetEvent`-properties (like `isSignal`)
- **extend this function to add further variables from clustered jet collection**

```python
python pickleToDataFrame.py
```

#### 3. DataFrame to ML algorithm

See [DNN Notebook](notebooks/DNN_minimalExample.ipynb) for an example workflow.

