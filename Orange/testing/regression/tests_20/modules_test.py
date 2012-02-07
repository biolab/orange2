import orange, orngTest, orngStat
import random

data = orange.ExampleTable("voting")

bayes = orange.BayesLearner(name = "bayes")
tree = orange.TreeLearner(name = "tree")
majority = orange.MajorityLearner(name = "default")
learners = [bayes, tree, majority]
names = [x.name for x in learners]

def printResults(res):
    CAs = orngStat.CA(res, reportSE=1)
    for i in range(len(names)):
        print "%s: %5.3f+-%5.3f   " % (names[i], CAs[i][0], 1.96*CAs[i][1]),
    print

print "\nproportionsTest that will always give the same results"
for i in range(3):
    res = orngTest.proportionTest(learners, data, 0.7)
    printResults(res)

print "\nproportionsTest that will give different results, but the same each time the script is run"
myRandom = orange.RandomGenerator()
for i in range(3):
    res = orngTest.proportionTest(learners, data, 0.7, randomGenerator = myRandom)
    printResults(res)

if not vars().has_key("NO_RANDOMNESS"):
    print "\nproportionsTest that will give different results each time it is run"
    for i in range(3):
        res = orngTest.proportionTest(learners, data, 0.7, randseed = random.randint(0, 100))
        printResults(res)

print "\nproportionsTest + storing classifiers"
res = orngTest.proportionTest(learners, data, 0.7, 100, storeClassifiers = 1)
print "#iter %i, #classifiers %i" % (len(res.classifiers), len(res.classifiers[0]) if len(res.classifiers) > 0 else -1)
print

##print "\nLearning with 100% class noise"
##classnoise = orange.Preprocessor_addClassNoise(proportion=1.0)
##res = orngTest.proportionTest(learners, data, 0.7, 100, pps = [("L", classnoise)])
##printResults(res)

print "\nGood old 10-fold cross validation"
res = orngTest.crossValidation(learners, data)
printResults(res)


print "\nLearning curve"
prop = orange.frange(0.2, 1.0, 0.2)
res = orngTest.learningCurveN(learners, data, folds = 5, proportions = prop)
for i in range(len(prop)):
    print "%5.3f:" % prop[i],
    printResults(res[i])

print "\nLearning curve with pre-separated data"
indices = orange.MakeRandomIndices2(data, p0 = 0.7)
train = data.select(indices, 0)
test = data.select(indices, 1)
res = orngTest.learningCurveWithTestData(learners, train, test, times = 5, proportions = prop)
for i in range(len(prop)):
    print "%5.3f:" % prop[i],
    printResults(res[i])


print "\nLearning and testing on pre-separated data"
res = orngTest.learnAndTestOnTestData(learners, train, test)
printResults(res)

print "\nLearning and testing on the same data"
res = orngTest.learnAndTestOnLearnData(learners, data)
printResults(res)
