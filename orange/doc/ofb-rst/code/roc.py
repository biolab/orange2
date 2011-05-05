# Description: Implementation of AUC (area under ROC curve) statistics, test of different methods through 10-fold cross validation (warning: for educational purposes only, use orngEval for estimation of AUC and similar statistics)
# Category:    evaluation
# Uses:        voting.tab
# Classes:     MakeRandomIndicesCV
# Referenced:  c_performance.htm


import orange, orngTree

def aroc(data, classifiers):
    ar = []
    for c in classifiers:
        p = []
        for d in data:
            p.append(c(d, orange.GetProbabilities)[0])
        correct = 0.0; valid = 0.0
        for i in range(len(data)-1):
            for j in range(i+1,len(data)):
                if data[i].getclass() <> data[j].getclass():
                    valid += 1
                    if p[i] == p[j]:
                        correct += 0.5
                    elif data[i].getclass() == 0:
                        if p[i] > p[j]:
                            correct += 1.0
                    else:
                        if p[j] > p[i]:
                            correct += 1.0
        ar.append(correct / valid)
    return ar

def cross_validation(data, learners, k=10):
    ar = [0.0]*len(learners)
    selection = orange.MakeRandomIndicesCV(data, folds=k)
    for test_fold in range(k):
        train_data = data.select(selection, test_fold, negate=1)
        test_data = data.select(selection, test_fold)
        classifiers = []
        for l in learners:
            classifiers.append(l(train_data))
        result = aroc(test_data, classifiers)
        for j in range(len(learners)):
            ar[j] += result[j]
    for j in range(len(learners)):
        ar[j] = ar[j]/k
    return ar

orange.setrandseed(0)    
# set up the learners
bayes = orange.BayesLearner()
tree = orngTree.TreeLearner(mForPruning=2)
maj = orange.MajorityLearner()
bayes.name = "bayes"
tree.name = "tree"
maj.name = "majority"
learners = [bayes, tree, maj]

# compute accuracies on data
data = orange.ExampleTable("voting")
acc = cross_validation(data, learners, k=10)
print "Area under ROC:"
for i in range(len(learners)):
    print learners[i].name, "%.2f" % acc[i]
