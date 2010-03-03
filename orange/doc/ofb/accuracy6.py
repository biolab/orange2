# Description: Leave-one-out method for estimation of classification accuracy. Demonstration of use for different learners
# Category:    evaluation
# Uses:        voting.tab
# Referenced:  c_performance.htm

import orange, orngTree

def leave_one_out(data, learners):
    acc = [0.0]*len(learners)
    selection = [1] * len(data)
    last = 0
    for i in range(len(data)):
        print 'leave-one-out: %d of %d' % (i, len(data))
        selection[last] = 1
        selection[i] = 0
        train_data = data.select(selection, 1)
        for j in range(len(learners)):
            classifier = learners[j](train_data)
            if classifier(data[i]) == data[i].getclass():
                acc[j] += 1
        last = i

    for j in range(len(learners)):
        acc[j] = acc[j]/len(data)
    return acc

orange.setrandseed(0)    
# set up the learners
bayes = orange.BayesLearner()
tree = orngTree.TreeLearner(minExamples=10, mForPruning=2)
bayes.name = "bayes"
tree.name = "tree"
learners = [bayes, tree]

# compute accuracies on data
data = orange.ExampleTable("voting")
acc = leave_one_out(data, learners)
print "Classification accuracies:"
for i in range(len(learners)):
    print learners[i].name, acc[i]
