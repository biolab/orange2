import Orange.data
import Orange.optimization
import Orange.classification
import Orange.evaluation.scoring as scoring
import Orange.core

learner = Orange.classification.tree.TreeLearner()
data = Orange.data.Table("voting")
tuner = Orange.optimization.TuneMParameters(object=learner,
             parameters=[("minSubset", [2, 5, 10, 20]),
                         ("measure", [Orange.core.MeasureAttribute_gainRatio(), 
                                      Orange.core.MeasureAttribute_gini()])],
             evaluate = Orange.evaluation.scoring.AUC)

classifier = tuner(data)
