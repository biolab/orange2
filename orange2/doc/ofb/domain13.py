# Description: Adds two new numerical attributes to iris data set, and tests through cross validation if this helps in boosting classification accuracy
# Category:    modelling
# Uses:        iris
# Classes:     Domain, FloatVariable, MakeRandomIndicesCV, orngTest.testWithIndices
# Referenced:  domain.htm

import orange, orngTest, orngStat, orngTree
data = orange.ExampleTable('iris')

sa = orange.FloatVariable("sepal area")
sa.getValueFrom = lambda e, getWhat: e['sepal length'] * e['sepal width']
pa = orange.FloatVariable("petal area")
pa.getValueFrom = lambda e, getWhat: e['petal length'] * e['petal width']

newdomain = orange.Domain(data.domain.attributes+[sa, pa, data.domain.classVar])
newdata = data.select(newdomain)

learners = [orngTree.TreeLearner(mForPruning=2.0)]

indices = orange.MakeRandomIndicesCV(data, 10)
res1 = orngTest.testWithIndices(learners, data, indices)
res2 = orngTest.testWithIndices(learners, newdata, indices)

print "original: %5.3f, new: %5.3f" % (orngStat.CA(res1)[0], orngStat.CA(res2)[0])
