# Description: Demonstrates the use of logistic regression
# Category:    classification, logistic regression
# Classes:     LogRegLearner
# Uses:        titanic.tab

import orange
import orngLR

data = orange.ExampleTable("titanic")
lr = orngLR.LogRegLearner(data) 

correct = 0
for ex in data:
    if lr(ex) == ex.getclass():
        correct += 1
        
print "Classification accuracy:", correct/len(data)
orngLR.printOUT(lr) 
