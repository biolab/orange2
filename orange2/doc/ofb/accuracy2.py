# Description: Set a number of learners, for each build a classifier from the data and determine classification accuracy
# Category:    evaluation
# Uses:        voting.tab
# Referenced:  c_performance.htm

import orange, orngTree

def accuracy(test_data, classifiers):
    correct = [0.0]*len(classifiers)
    for ex in test_data:
        for i in range(len(classifiers)):
            if classifiers[i](ex) == ex.getclass():
                correct[i] += 1
    for i in range(len(correct)):
        correct[i] = correct[i] / len(test_data)
    return correct

# set up the classifiers
data = orange.ExampleTable("voting")
bayes = orange.BayesLearner(data)
bayes.name = "bayes"
tree = orngTree.TreeLearner(data);
tree.name = "tree"
classifiers = [bayes, tree]

# compute accuracies
acc = accuracy(data, classifiers)
print "Classification accuracies:"
for i in range(len(classifiers)):
    print classifiers[i].name, acc[i]
