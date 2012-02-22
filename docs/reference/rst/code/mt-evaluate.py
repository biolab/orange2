import Orange

data = Orange.data.Table('multitarget-synthetic')

majority = Orange.multitarget.MultitargetLearner(
    Orange.classification.majority.MajorityLearner(), name='Majority')
tree = Orange.multitarget.tree.MultiTreeLearner(max_depth=3, name='MT Tree')
pls = Orange.multitarget.pls.PLSRegressionLearner(name='PLS')
earth = Orange.multitarget.earth.EarthLearner(name='Earth')

learners = [majority, tree, pls, earth]
res = Orange.evaluation.testing.cross_validation(learners, data)
rmse = Orange.evaluation.scoring.RMSE
scores = Orange.evaluation.scoring.mt_average_score(
            res, rmse, weights=[5,2,2,1])
print 'Weighted RMSE scores:'
print '\n'.join('%12s\t%.4f' % r for r in zip(res.classifier_names, scores))
