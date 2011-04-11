import orange

# Definitely ugly, but I see no other workaround.
# When, e.g. data.io executes "from orange import ExampleTable"
# orange gets imported again since it is not in sys.modules
# before this entire file is executed
import sys
sys.modules["orange"] = orange

import warnings

def _import(name):
    try:
        __import__(name, globals(), locals(), [], -1)
    except:
        warnings.warn("Could not import: " + name, UserWarning, 2)

_import("data")
_import("data.io")
_import("data.sample")
_import("data.variable")

_import("network")

_import("stat")

_import("statistics")
_import("statistics.estimate")
_import("statistics.contingency")
_import("statistics.distribution")
_import("statistics.basic")
_import("statistics.evd")

_import("classification")
_import("classification.tree")

_import("classification.rules")

_import("classification.lookup")
_import("classification.bayes")
_import("classification.svm")
_import("classification.logreg")
_import("classification.knn")
_import("classification.majority")

_import("optimization")

_import("projection")
_import("projection.mds")
_import("projection.som")

_import("ensemble")
_import("ensemble.bagging")
_import("ensemble.boosting")
_import("ensemble.forest")

_import("regression")
_import("regression.mean")

_import("associate")

_import("preprocess")
#import preprocess.value
#import preprocess.data

_import("distances")

_import("wrappers")

_import("featureConstruction")
_import("featureConstruction.univariate")
_import("featureConstruction.functionDecomposition")

_import("evaluation")
_import("evaluation.scoring")
_import("evaluation.testing")

_import("clustering")
_import("clustering.kmeans")
_import("clustering.hierarchical")

import misc
import misc.counters
import misc.render
import misc.selection
