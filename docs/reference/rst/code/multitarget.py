import Orange
data = Orange.data.Table('multitarget-synthetic')
print 'Features:', data.domain.features
print 'Classes:', data.domain.class_vars
print 'First instance:', data[0]
print 'Actual classes:', data[0].get_classes()

majority = Orange.classification.majority.MajorityLearner()
mt_majority = Orange.multitarget.MultitargetLearner(majority)
c_majority = mt_majority(data)
print 'Majority predictions:\n', c_majority(data[0])

pls = Orange.multitarget.pls.PLSRegressionLearner()
c_pls = pls(data)
print 'PLS predictions:\n', c_pls(data[0])

mt_tree = Orange.multitarget.tree.MultiTreeLearner(max_depth=3)
c_tree = mt_tree(data)
print 'Multi-target Tree predictions:\n', c_tree(data[0])
