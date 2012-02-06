# Description: Read data, learn several models (bayes, kNN, decision tree) and for all models output class probabilities they return for first few instances
# Category:    modelling
# Uses:        voting.tab
# Classes:     MajorityLearner, BayesLearner, orngTree.TreeLearner, kNNLearner
# Referenced:  c_otherclass.htm

import orange, orngTree
data = orange.ExampleTable("voting")

# setting up the classifiers
majority = orange.MajorityLearner(data)
bayes = orange.BayesLearner(data)
tree = orngTree.TreeLearner(data, sameMajorityPruning=1, mForPruning=2)
knn = orange.kNNLearner(data, k=21)

majority.name="Majority"; bayes.name="Naive Bayes";
tree.name="Tree"; knn.name="kNN"

classifiers = [majority, bayes, tree, knn]

# print the head
print "Possible classes:", data.domain.classVar.values
print "Probability for republican:"
print "Original Class",
for l in classifiers:
    print "%-13s" % (l.name),
print

# classify first 10 instances and print probabilities
for example in data[:10]:
    print "(%-10s)  " % (example.getclass()),
    for c in classifiers:
        p = apply(c, [example, orange.GetProbabilities])
        print "%5.3f        " % (p[0]),
    print
