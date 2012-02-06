# Category:    evaluation
# Description: Set a number of learners, split data to train and test set, learn models from train set and estimate classification accuracy on the test set
# Uses:        voting.tab
# Classes:     MakeRandomIndices2
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
selection = orange.MakeRandomIndices2(data, 0.5)
train_data = data.select(selection, 0)
test_data = data.select(selection, 1)

bayes = orange.BayesLearner(train_data)
tree = orngTree.TreeLearner(train_data)
bayes.name = "bayes"
tree.name = "tree"
classifiers = [bayes, tree]

# compute accuracies
acc = accuracy(test_data, classifiers)
print "Classification accuracies:"
for i in range(len(classifiers)):
    print classifiers[i].name, acc[i]

