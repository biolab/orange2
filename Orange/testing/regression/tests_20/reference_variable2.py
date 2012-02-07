# Description: Shows how to construct new attributes from the existing and use it seamlessly in sampling and testing
# Category:    feature construction, constructive induction
# Classes:     Variable, MakeRandomIndices2
# Uses:        monk1
# Referenced:  Variable.htm

import orange, orngTree

data = orange.ExampleTable("monk1")

indices = orange.MakeRandomIndices2(data, p0=0.7)
trainData = data.select(indices, 0)
testData = data.select(indices, 1)

e2 = orange.EnumVariable("e2", values = ["not 1", "1"])
e2.getValueFrom = lambda example, returnWhat: \
                  orange.Value(e2, example["e"]=="1")

newDomain = orange.Domain([data.domain["a"], data.domain["b"], e2, data.domain.classVar])
newTrain = orange.ExampleTable(newDomain, trainData)

tree = orange.TreeLearner(newTrain)

orngTree.printTxt(tree)

for ex in testData[:10]:
    print ex.getclass(), tree(ex)

