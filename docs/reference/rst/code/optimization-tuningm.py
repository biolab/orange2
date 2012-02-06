import Orange

learner = Orange.classification.tree.TreeLearner()
voting = Orange.data.Table("voting")
tuner = Orange.optimization.TuneMParameters(object=learner,
             parameters=[("minSubset", [2, 5, 10, 20]),
                         ("measure", [Orange.core.MeasureAttribute_gainRatio(), 
                                      Orange.core.MeasureAttribute_gini()])],
             evaluate = Orange.evaluation.scoring.AUC)

classifier = tuner(voting)