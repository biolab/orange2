import orange

# Definitely ugly, but I see no other workaround.
# When, e.g. data.io executes "from orange import ExampleTable"
# orange gets imported again since it is not in sys.modules
# before this entire file is executed
import sys
sys.modules["orange"] = orange

import data
import data.io
import data.sample
import data.feature

import network

import stat

import probability
import probability.estimate
import probability.distributions
import probability.evd

import classification
import classification.tree
import classification.rules
import classification.lookup
import classification.bayes
import classification.svm
import classification.logreg
import classification.knn
import classification.majority

import projection
import projection.mds
import projection.som

import ensemble
import ensemble.bagging
import ensemble.boosting
import ensemble.forest

import regression
import regression.mean

import associate

import preprocess
#import preprocess.value
#import preprocess.data

import distances

import wrappers

import featureConstruction
import featureConstruction.univariate
import featureConstruction.functionDecomposition

import evaluation
import evaluation.scoring
import evaluation.testing

import clustering

import misc
import misc.counters
import misc.render
import misc.selection
