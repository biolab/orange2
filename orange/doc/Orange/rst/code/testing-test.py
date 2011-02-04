import Orange
import orngStat
import random

table = Orange.data.Table("voting")

bayes = Orange.core.BayesLearner(name="bayes")
tree = Orange.core.TreeLearner(name="tree")
majority = Orange.classification.majority.MajorityLearner(name="default")
learners = [bayes, tree, majority]

def printResults(res):
    CAs = orngStat.CA(res, reportSE=1)
    for name, ca in zip(res.classifierNames, CAs):
        print "%s: %5.3f+-%5.3f" % (name, ca[0], 1.96 * ca[1]),
    print

print "\nproportionsTest that will always give the same results"
for i in range(3):
    res = Orange.evaluation.testing.proportionTest(learners, table, 0.7)
    printResults(res)

print "\nproportionsTest that will give different results, \
but the same each time the script is run"
myRandom = Orange.core.RandomGenerator()
for i in range(3):
    res = Orange.evaluation.testing.proportionTest(learners, table, 0.7,
        randomGenerator=myRandom)
    printResults(res)
# End

print "\nproportionsTest that will give different results each time it is run"
for i in range(3):
    res = Orange.evaluation.testing.proportionTest(learners, table, 0.7,
        randseed=random.randint(0, 100))
    printResults(res)
# End

print "\nproportionsTest + storing classifiers"
res = Orange.evaluation.testing.proportionTest(learners, table, 0.7, 100,
    storeClassifiers=1)
print "#iter %i, #classifiers %i" % \
    (len(res.classifiers), len(res.classifiers[0]))

print "\nGood old 10-fold cross validation"
res = Orange.evaluation.testing.crossValidation(learners, table)
printResults(res)

print "\nLearning curve"
prop = Orange.core.frange(0.2, 1.0, 0.2)
res = Orange.evaluation.testing.learningCurveN(learners, table, folds=5,
    proportions=prop)
for i in range(len(prop)):
    print "%5.3f:" % prop[i],
    printResults(res[i])
# End

print "\nLearning curve with pre-separated data"
indices = Orange.core.MakeRandomIndices2(table, p0=0.7)
train = table.select(indices, 0)
test = table.select(indices, 1)
res = Orange.evaluation.testing.learningCurveWithTestData(learners, train,
    test, times=5, proportions=prop)
for i in range(len(prop)):
    print "%5.3f:" % prop[i],
    printResults(res[i])
# End

print "\nLearning and testing on pre-separated data"
res = Orange.evaluation.testing.learnAndTestOnTestData(learners, train, test)
printResults(res)

print "\nLearning and testing on the same data"
res = Orange.evaluation.testing.learnAndTestOnLearnData(learners, table)
printResults(res)
