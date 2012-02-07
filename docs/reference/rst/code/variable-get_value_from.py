# Description: Construction and computation of new features
# Category:    core
# Uses:        monks-1
# Referenced:  Orange.feature
# Classes:     Orange.feature.Discrete

import Orange

def checkE(inst, return_what):
    if inst["e"]=="1": 
        return e2("1")
    else:
        return e2("not 1") 

monks = Orange.data.Table("monks-1")
e2 = Orange.feature.Discrete("e2", values=["not 1", "1"])    
e2.get_value_from = checkE 

print Orange.feature.scoring.InfoGain(e2, monks)

dist = Orange.core.Distribution(e2, monks)
print dist 

# Split the data into training and testing set
indices = Orange.core.MakeRandomIndices2(monks, p0=0.7)
train_data = monks.select(indices, 0)
test_data = monks.select(indices, 1)

# Convert the training set to a new domain
new_domain = Orange.data.Domain([monks.domain["a"], monks.domain["b"], e2, monks.domain.class_var])
new_train = Orange.data.Table(new_domain, train_data)

# Construct a tree and classify unmodified instances
tree = Orange.core.TreeLearner(new_train)
for ex in test_data[:10]:
    print ex.getclass(), tree(ex)
