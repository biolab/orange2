import orange

# Definitely ugly, but I see no other workaround.
# When, e.g. data.io executes "from orange import ExampleTable"
# orange gets imported again since it is not in sys.modules
# before this entire file is executed
import sys
sys.modules["orange"] = orange

import data
import data.io
import data.example
import data.sample
import data.value
import data.variable
import data.domain

import graph

import stat

import probability
import probability.estimate
import probability.distributions
import probability.evd

import classify
import classify.trees
import classify.rules
import classify.lookup
import classify.bayes
import classify.svm
import classify.logreg
import classify.knn
import classify.majority

import regress

import associate

import preprocess
#import preprocess.value
#import preprocess.data

import distances

import wrappers

import featureConstruction
import featureConstruction.univariate
import featureConstruction.functionDecomposition

import cluster
