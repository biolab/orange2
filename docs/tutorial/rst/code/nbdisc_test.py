# Description: Test of naive bayesian classifier with entropy-based discretization (as defined in nbdisc.py)
# Category:    modelling
# Uses:        iris.tab
# Classes:     orngTest.crossValidation, orngStat.CA
# Referenced:  c_nb_disc.htm

import orange, orngTest, orngStat, nbdisc
data = orange.ExampleTable("iris")
results = orngTest.crossValidation([nbdisc.Learner()], data, folds=10)
print "Accuracy = %5.3f" % orngStat.CA(results)[0]
