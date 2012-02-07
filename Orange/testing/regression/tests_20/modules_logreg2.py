# Description: Demonstrates the use of logistic regression
# Category:    classification, logistic regression
# Classes:     LogRegLearner
# Uses:        adult_sample.tab

import orange, orngLR

data = orange.ExampleTable("../datasets/adult_sample.tab")
lr = orngLR.LogRegLearner(data, removeSingular = 1)

for ex in data[:5]:
    print ex.getclass(), lr(ex)

out = ['']

# get the longest attribute name
longest=0
for at in lr.continuized_domain.features:
    if len(at.name)>longest:
        longest=len(at.name)

# print out the head
for i in range(len(lr.continuized_domain.features)):
    print lr.continuized_domain.features[i].name, lr.beta[i+1]
