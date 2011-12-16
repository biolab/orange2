# Category:    evaluation
# Description: Estimation of accuracy by cross validation. Demonstration of use for different learners.
# Uses:        voting.tab
# Classes:     MakeRandomIndicesCV
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

def cross_validation(data, learners, k=10):
    acc = [0.0]*len(learners)
    selection= orange.MakeRandomIndicesCV(data, folds=k)
    for test_fold in range(k):
        train_data = data.select(selection, test_fold, negate=1)
        test_data = data.select(selection, test_fold)
        classifiers = []
        for l in learners:
            classifiers.append(l(train_data))
        acc1 = accuracy(test_data, classifiers)
        print "%d: %s" % (test_fold+1, acc1)
        for j in range(len(learners)):
            acc[j] += acc1[j]
    for j in range(len(learners)):
        acc[j] = acc[j]/k
    return acc

orange.setrandseed(0)
# set up the learners
bayes = orange.BayesLearner()
tree = orngTree.TreeLearner(mForPruning=2)

bayes.name = "bayes"
tree.name = "tree"
learners = [bayes, tree]

# compute accuracies on data
data = orange.ExampleTable("voting")
acc = cross_validation(data, learners, k=10)
print "Classification accuracies:"
for i in range(len(learners)):
    print learners[i].name, acc[i]
