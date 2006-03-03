# Description: Read data, build naive Bayesian classifier and classify first few instances
# Category:    modelling
# Uses:        voting.tab
# Referenced:  c_basics.htm

import orange
data = orange.ExampleTable("voting")
classifier = orange.BayesLearner(data)
for i in range(5):
    c = classifier(data[i])
    print "%d: %s (originally %s)" % (i+1, c, data[i].getclass())
