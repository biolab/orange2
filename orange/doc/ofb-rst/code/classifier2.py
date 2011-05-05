# Description: Read data, build naive Bayesian classifier, and output class probabilities for the first few instances
# Category:    modelling
# Uses:        voting.tab
# Referenced:  c_basics.htm

import orange
data = orange.ExampleTable("voting")
classifier = orange.BayesLearner(data)
print "Possible classes:", data.domain.classVar.values
print "Probabilities for democrats:"
for i in range(5):
    p = classifier(data[i], orange.GetProbabilities)
    print "%d: %5.3f (originally %s)" % (i+1, p[1], data[i].getclass())
