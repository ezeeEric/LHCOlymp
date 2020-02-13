#!/usr//bin/env python

""" Run script to build/update the Database
"""

## std
import os
from argparse import ArgumentParser

## local
from src.database import Database
from src.dataset import Dataset 

##<<------------------------------------------------------------------------
## setup flags
##<<------------------------------------------------------------------------
parser = ArgumentParser()
parser.add_argument("--input", "-i", 
    help="Path to the input files")
parser.add_argument("--locality-feat", "-l", 
    help="A feature to split regions along it")
parser.add_argument("--num-slices", "-n", default=10, 
    help="Number of regions along locality feature")
parser.add_argument("--locality-feat-range", "-r", type=float, nargs=2, 
    help="Range of locality feature")
parser.add_argument("--outdir", "-o", 
    help="Path to the output directory")
parser.add_argument("--db-file", "-d", 
    help="Path to the db file")

parser.add_argument("--reset", action="store_true", 
    help="Clean the database ?")

flags = parser.parse_args()

##<<------------------------------------------------------------------------
## build Database
dbFile = os.path.join(flags.outdir, flags.db_file)
db = Database(name='DB', dbFile=flags.db_file, outDir=flags.outdir, inputPath=flags.input, outExt=".pkl",
    localityFeature=flags.locality_feat, numSlices=flags.num_slices, localityFeatureRange=flags.locality_feat_range, sideBandMargin=0.3, logLevel="INFO")

##<<------------------------------------------------------------------------
## cleanup
if flags.reset:
    db.rest() 

##<<------------------------------------------------------------------------
## scan and update 
db.scan()

##<<------------------------------------------------------------------------
## write 
db.write()




