# Description: Learn a naive Bayesian classifier, and measure classification accuracy on the same data set
# Category:    evaluation
# Uses:        voting.tab
# Referenced:  c_performance.htm

import orange
data = orange.ExampleTable("voting")
classifier = orange.BayesLearner(data)

# compute classification accuracy
correct = 0.0
for ex in data:
    if classifier(ex) == ex.getclass():
        correct += 1
print "Classification accuracy:", correct/len(data)
