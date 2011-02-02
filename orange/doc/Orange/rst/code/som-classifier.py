# Description: Self Organizing Maps on iris data set
# Category:    modelling
# Uses:        iris
# Referenced:  orngSOM.htm
# Classes:     orngSOM.SOMLearner

import Orange
import random
learner = Orange.projection.som.SOMSupervisedLearner(map_shape=(4, 4))
data = Orange.data.Table("iris.tab")
classifier = learner(data)
random.seed(50)
for d in random.sample(data, 5):
    print "%-15s originally %-15s" % (classifier(d), d.getclass())
