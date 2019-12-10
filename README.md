# LHCOlymp

See [Overview Notebook](notebooks/overview.ipynb) for full details 
on [LHCOlympics2020](https://indico.cern.ch/event/809820/page/19002-lhcolympics2020) challenge. 

### Package Requirements (to be specified):
- pyjet
- tables
- sklearn
- tensorflow
- keras


## Analysis Flow

### Data Preparation
#### 1. Hadrons to Events with Jet collections

- Use pyjet (fastjet) library to cluster hadrons in input files to jets

..- store events with clustered jet collections in `JetEvent` class objects
..- single jet is numpy dtype with limited, specified variables read out from FastJet::PSeudoJet object
..- **extend this dtype to add further variables from single jet**

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

