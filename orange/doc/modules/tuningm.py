import orange, orngTree, orngStat, orngWrap

learner = orngTree.TreeLearner()
data = orange.ExampleTable("voting")
tuner = orngWrap.TuneMParameters(object=learner,
                                 parameters=[("minSubset", [2, 5, 10, 20]),
                                             ("measure", [orange.MeasureAttribute_gainRatio(), orange.MeasureAttribute_gini()])],
                                 evaluate = orngStat.AUC)
classifier = tuner(data)
