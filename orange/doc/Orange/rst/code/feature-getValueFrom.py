# Description: Construction and computation of new features
# Category:    core
# Uses:        monks-1
# Referenced:  Orange.data.feature
# Classes:     Orange.data.feature.Discrete

import Orange

def checkE(inst, returnWhat):
    if inst["e"]=="1": 
        return e2("1")
    else:
        return e2("not 1") 

data = Orange.data.Table("monks-1")
e2 = Orange.data.feature.Discrete("e2", values=["not 1", "1"])    
e2.getValueFrom = checkE 

print Orange.core.MeasureAttribute_info(e2, data)

dist = Orange.core.Distribution(e2, data)
print dist 

# Split the data into training and testing set
indices = Orange.core.MakeRandomIndices2(data, p0=0.7)
trainData = data.select(indices, 0)
testData = data.select(indices, 1)

# Convert the training set to a new domain
newDomain = orange.Domain([data.domain["a"], data.domain["b"], e2, data.domain.classVar])
newTrain = orange.ExampleTable(newDomain, trainData)

# Construct a tree and classify unmodified instances
tree = orange.TreeLearner(newTrain)
for ex in testData[:10]:
    print ex.getclass(), tree(ex)
