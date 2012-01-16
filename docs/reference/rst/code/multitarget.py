import Orange
data = Orange.data.Table('test-pls')
print 'Actual classes:\n', data[0].get_classes()

majority = Orange.classification.majority.MajorityLearner()
mt_majority = Orange.multitarget.MultitargetLearner(majority)
c_mtm = mt_majority(data)
print 'Majority predictions:\n', c_mtm(data[0])

mt_tree = Orange.multitarget.tree.MultiTreeLearner(max_depth=3)
c_mtt = mt_tree(data)
print 'Multi-target Tree predictions:\n', c_mtt(data[0])
