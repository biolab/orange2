import Orange

learners = [ Orange.classification.bayes.NaiveLearner(name = "bayes"),
             Orange.classification.tree.TreeLearner(name="tree"),
             Orange.classification.majority.MajorityLearner(name="majrty")]

voting = Orange.data.Table("voting")
res = Orange.evaluation.testing.cross_validation(learners, voting)

vehicle = Orange.data.Table("vehicle")
resVeh = Orange.evaluation.testing.cross_validation(learners, vehicle)
