# Description: Demonstrates the use of logistic regression
# Category:    classification, logistic regression
# Classes:     LogRegLearner
# Uses:        adult_sample.tab

import orange, orngLR

data = orange.ExampleTable("../datasets/adult_sample.tab")
lr = orngLR.LogRegLearner(data, removeSingular = 1)

for ex in data[:5]:
    print ex.getclass(), lr(ex)
    
orngLR.printOUT(lr) 