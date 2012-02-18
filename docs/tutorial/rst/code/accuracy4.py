# Description: Estimation of accuracy by random sampling.
# User can set what proportion of data will be used in training.
# Demonstration of use for different learners.
# Category:   evaluation
# Uses:        voting.tab
# Classes:     MakeRandomIndices2
# Referenced:  c_performance.htm

import orange, orngTree

def accuracy(test_data, classifiers):
    correct = [0.0] * len(classifiers)
    for ex in test_data:
        for i in range(len(classifiers)):
            if classifiers[i](ex) == ex.getclass():
                correct[i] += 1
    for i in range(len(correct)):
        correct[i] = correct[i] / len(test_data)
    return correct

def test_rnd_sampling(data, learners, p=0.7, n=10):
    acc = [0.0] * len(learners)
    for i in range(n):
        selection = orange.MakeRandomIndices2(data, p)
        train_data = data.select(selection, 0)
        test_data = data.select(selection, 1)
        classifiers = []
        for l in learners:
            classifiers.append(l(train_data))
        acc1 = accuracy(test_data, classifiers)
        print "%d: %s" % (i + 1, ["%.6f" % a for a in acc1])
        for j in range(len(learners)):
            acc[j] += acc1[j]
    for j in range(len(learners)):
        acc[j] = acc[j] / n
    return acc

orange.setrandseed(0)
# set up the learners
bayes = orange.BayesLearner()
tree = orngTree.TreeLearner();
#tree = orngTree.TreeLearner(mForPruning=2)
bayes.name = "bayes"
tree.name = "tree"
learners = [bayes, tree]

# compute accuracies on data
data = orange.ExampleTable("voting")
acc = test_rnd_sampling(data, learners)
print "Classification accuracies:"
for i in range(len(learners)):
    print learners[i].name, acc[i]
