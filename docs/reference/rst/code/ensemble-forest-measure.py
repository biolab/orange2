# Description: Demonstrates the use of random forests from Orange.ensemble.forest module
# Category:    classification, ensembles
# Classes:     RandomForestLearner
# Uses:        iris.tab
# Referenced:  orngEnsemble.htm

import Orange
import random

files = [ "iris.tab" ]

for fn in files:
    print "\nDATA:" + fn + "\n"
    iris = Orange.data.Table(fn)

    measure = Orange.ensemble.forest.ScoreFeature(trees=100)

    #call by attribute index
    imp0 = measure(0, iris) 
    #call with a Descriptor
    imp1 = measure(iris.domain.attributes[1], iris)
    print "first: %0.2f, second: %0.2f\n" % (imp0, imp1)

    print "different random seed"
    measure = Orange.ensemble.forest.ScoreFeature(trees=100, 
            rand=random.Random(10))

    imp0 = measure(0, iris)
    imp1 = measure(iris.domain.attributes[1], iris)
    print "first: %0.2f, second: %0.2f\n" % (imp0, imp1)

    print "All importances:"
    for at in iris.domain.attributes:
        print "%15s: %6.2f" % (at.name, measure(at, iris))
