import orange
from orngMisc import demangleExamples, getobjectname, classChecksum
import exceptions, whrandom, cPickle, os, os.path

#### Some private stuff

def encodePP(pps):
    pps=""
    for pp in pps:
        objname = getobjectname(pp[1], "")
        if len(objname):
            pps+="_"+objname
        else:
            return "*"
    return pps



#### Data structures

class TestedExample:
    def __init__(self, iterationNumber=None, actualClass=None, n=0, weight=1.0):
        self.classes = [None]*n
        self.probabilities = [None]*n
        self.iterationNumber = iterationNumber
        self.actualClass= actualClass
        self.weight = weight
    
    def addResult(self, aclass, aprob):
        if type(aclass.value)==float:
            self.classes.append(float(aclass))
            self.probabilities.append(aprob)
        else:
            self.classes.append(int(aclass))
            self.probabilities.append(list(aprob))

    def setResult(self, i, aclass, aprob):
        if type(aclass.value)==float:
            self.classes[i] = float(aclass)
            self.probabilities[i] = aprob
        else:
            self.classes[i] = int(aclass)
            self.probabilities[i] = list(aprob)


class ExperimentResults:
    def __init__(self, iterations, learners, weights, baseClass, **argkw):
        self.numberOfIterations = iterations
        self.numberOfLearners = learners
        self.results = []
        self.classifiers = []
        self.loaded = None
        self.baseClass = baseClass
        self.weights = weights
        self.__dict__.update(argkw)

    def loadFromFiles(self, learners, filename):
        self.loaded = []
      
        for i in range(len(learners)):
            f = None
            try:
                f = open(".\\cache\\"+filename % getobjectname(learners[i], "*"), "rb")
                d = cPickle.load(f)
                for ex in range(len(self.results)):
                    tre = self.results[ex]
                    if (tre.actualClass, tre.iterationNumber) != d[ex][0]:
                        raise SystemError, "mismatching example tables or sampling"
                    self.results[ex].setResult(i, d[ex][1][0], d[ex][1][1])
                self.loaded.append(1)
            except exceptions.Exception:
                self.loaded.append(0)
            if f:
                f.close()
                
        return not 0 in self.loaded                
                

    def saveToFiles(self, learners, filename):
        for i in range(len(learners)):
            if self.loaded[i]:
                continue
            
            fname=".\\cache\\"+filename % getobjectname(learners[i], "*")
            if not "*" in fname:
                if not os.path.isdir("cache"):
                    os.mkdir("cache")
                f=open(fname, "wb")
                pickler=cPickle.Pickler(f, 1)
                pickler.dump([(  (x.actualClass, x.iterationNumber), (x.classes[i], x.probabilities[i])  ) for x in self.results])
                f.close()



#### Experimental procedures
def leaveOneOut(learners, examples, pps=[], **argkw):
    return testWithIndices(learners, examples, range(len(examples)), pps)


def proportionTest(learners, examples, learnProp, times=10, strat=orange.MakeRandomIndices.StratifiedIfPossible, pps=[], **argkw):
    examples, weight = demangleExamples(examples)
    pick = orange.MakeRandomIndices2(stratified = strat, p0 = learnProp)
    testResults = ExperimentResults(times, len(learners), weight!=0, examples.domain.classVar.baseValue)
    for time in range(times):
        indices = pick(examples)
        learnset = examples.selectref(indices, 0)
        testset = examples.selectref(indices, 1)
        learnAndTestOnTestData(learners, (learnset, weight), (testset, weight), testResults, time, pps)
    return testResults


def crossValidation(learners, examples, folds=10, strat=orange.MakeRandomIndices.StratifiedIfPossible, pps=[], **argkw):
    (examples, weight) = demangleExamples(examples)
    return apply(testWithIndices, (learners, (examples, weight), orange.MakeRandomIndicesCV(examples, folds, stratified = strat), "*", pps), argkw)


def learningCurveN(learners, examples, folds=10, strat=orange.MakeRandomIndices.StratifiedIfPossible, proportions=orange.frange(0.1), pps=[], **argkw):
    if strat:
        cv=orange.MakeRandomIndicesCV(folds = folds, stratified = strat)
        pick=orange.MakeRandomIndices2(stratified = strat)
    else:
        cv=orange.RandomIndicesCV(folds = folds, stratified = strat)
        pick=orange.RandomIndices2(stratified = strat)
    return apply(learningCurve, (learners, examples, cv, pick, proportions, pps), argkw)


def learningCurve(learners, examples, cv=None, pick=None, proportions=orange.frange(0.1), pps=[], **argkw):
    verb = argkw.get("verbose", 0)
    cache = argkw.get("cache", 1)

    for pp in pps:
        if pp[0]!="L":
            raise SystemError, "cannot preprocess testing examples"
    
    if not cv:
        cv = orange.MakeRandomIndicesCV(folds=10, stratified=orange.MakeRandomIndices.StratifiedIfPossible, randseed=0)
    if not pick:
        pick = orange.MakeRandomIndices2(stratified=orange.MakeRandomIndices.StratifiedIfPossible, randseed=0)

    examples, weight = demangleExamples(examples)
    folds = cv(examples)
    ccsum = classChecksum(examples)
    ppsp = encodePP(pps)
    nLrn = len(learners)

    allResults=[]
    for p in proportions:
        verbose_print(verb, "Proportion: %5.3f" % p)

        if (cv.randseed<0) or (pick.randseed<0):
            cache = 0
        else:
            fnstr = "{learningCurve}_%s_%s_%s_%s%s-%s" % ("%s", p, cv.randseed, pick.randseed, ppsp, ccsum)
            if "*" in fnstr:
                cache = 0

        testResults = ExperimentResults(cv.folds, len(learners), weight!=0, examples.domain.classVar.baseValue)
        testResults.results = [TestedExample(folds[i], int(examples[i].getclass()), nLrn, examples[i].getweight())
                               for i in range(len(examples))]

        if cache and testResults.loadFromFiles(learners, fnstr):
            verbose_print(verb, "  loaded from cache")
        else:
            for fold in range(cv.folds):
                verbose_print(verb, "  fold %d" % fold)
                
                # learning
                learnset = examples.selectref(cv, fold, negate=1)
                learnset = learnset.selectref(pick(learnset, p0=p), 0)
                if not len(learnset):
                    continue
                
                for pp in pps:
                    learnset = pp[1](learnset)

                classifiers = [None]*nLrn
                for i in range(nLrn):
                    if not testResults.loaded[i]:
                        classifiers[i] = learners[i](learnset, weight)

                # testing
                for i in range(len(examples)):
                    if (folds[i]==fold):
                        ex = examples[i]
                        for cl in range(nLrn):
                            if not testResults.loaded[cl]:
                                cls, pro = classifiers[cl](examples[i], orange.GetBoth)
                                testResults.results[i].setResult(cl, cls, pro)
            if cache:
                testResults.saveToFiles(learners, fnstr)

        allResults.append(testResults)
        
    return allResults


def learningCurveWithTestData(learners, learnset, testset, times=10, proportions=orange.frange(0.1), strat=orange.MakeRandomIndices.StratifiedIfPossible, pps=[], **argkw):
    verb = argkw.get("verbose", 0)

    learnset, learnweight = demangleExamples(learnset)
    testweight = demangleExamples(testset)[1]
    
    pick = orange.MakeRandomIndices2(stratified = strat)
    allResults=[]
    for p in proportions:
        print_verbose(verb, "Proportion: %5.3f" % p)
        testResults = ExperimentResults(times, len(learners), testweight!=0, testset.domain.classVar.baseValue)
        testResults.results = []
        
        for t in range(times):
            print_verbose(verb, "  repetition %d" % t)
            learnAndTestWithTestData(learners, (learnset.selectref(pick(learnset, p), 0), learnweight), testset, testResults, t)

        allResults.append(testResults)
        
    return allResults

   
def testWithIndices(learners, examples, indices, indicesrandseed="*", pps=[], **argkw):
    verb = argkw.get("verbose", 0)
    cache = argkw.get("cache", 1)
    storeclassifiers = argkw.get("storeclassifiers", 0) or argkw.get("storeClassifiers", 0)
    cache = cache and not storeclassifiers

    examples, weight = demangleExamples(examples)
    nLrn = len(learners)
    
    for pp in pps:
        if pp[0]!="L":
            raise SystemError, "cannot preprocess testing examples"

    nIterations = max(indices)+1
    testResults = ExperimentResults(nIterations, len(learners), weight!=0, examples.domain.classVar.baseValue)
    testResults.results = [TestedExample(indices[i], int(examples[i].getclass()), nLrn, examples[i].getweight(weight))
                           for i in range(len(examples))]

    ccsum = classChecksum(examples)
    ppsp = encodePP(pps)
    fnstr = "{TestWithIndices}_%s_%s%s-%s" % ("%s", indicesrandseed, ppsp, ccsum)
    if "*" in fnstr:
        cache = 0

    if cache and testResults.loadFromFiles(learners, fnstr):
        verbose_print(verb, "  loaded from cache")
    else:
        for fold in range(nIterations):
            # learning
            learnset = examples.selectref(indices, fold, negate=1)
            if not len(learnset):
                continue
            
            for pp in pps:
                learnset = pp[1](learnset)

            classifiers = [None]*nLrn
            for i in range(nLrn):
                if not cache or not testResults.loaded[i]:
                    classifiers[i] = learners[i](learnset, weight)
            if storeclassifiers:    
                testResults.classifiers.append(classifiers)

            # testing
            for i in range(len(examples)):
                if (indices[i]==fold):
                    ex = examples[i]
                    for cl in range(nLrn):
                        if not cache or not testResults.loaded[cl]:
                            cr = classifiers[cl](examples[i], orange.GetBoth)
                            testResults.results[i].setResult(cl, cr[0], cr[1])
        if cache:
            testResults.saveToFiles(learners, fnstr)
        
    return testResults
   




def learnAndTestOnTestData(learners, learnset, testset, testResults=None, iterationNumber=0, pps=[], **argkw):
    learnset, learnweight = demangleExamples(learnset)
    testset, testweight = demangleExamples(testset)
    
    for pp in pps:
        if pp[0]=="B":
            learnset = pp[0](learnset)
            testset = pp[0](testset)

    for pp in pps:
        if pp[0]=="L":
            learnset = pp[0](learnset)
        elif pp[0]=="T":
            testset = pp[0](testset)
            
    return testOnData([learner(learnset, learnweight) for learner in learners],
                            (testset, testweight), testResults, iterationNumber)


def learnAndTestOnLearnData(learners, learnset, testResults=None, iterationNumber=0, pps=[], **argkw):
    learnset, learnweight = demangleExamples(learnset)

    hasLorT = 0    
    for pp in pps:
        if pp[0]=="B":
            learnset = pp[0](learnset)
        else:
            hasLorT = 1

    if hasLorT:
        testset = orange.ExampleTable(learnset)
        for pp in pps:
            if pp[0]=="L":
                learnset = pp[0](learnset)
            elif pp[0]=="T":
                testset = pp[0](testset)
    else:
        testset = learnset    

    classifiers = [learner(learnset, learnweight) for learner in learners]
    return testOnData(classifiers, (testset, learnweight), testResults, iterationNumber)


def testOnData(classifiers, testset, testResults=None, iterationNumber=0, **argkw):
    testset, testweight = demangleExamples(testset)

    if not testResults:
        testResults=ExperimentResults(1, len(classifiers), testweight!=0, testset.domain.classVar.baseValue)
    
    for ex in testset:
        te = TestedExample(iterationNumber, int(ex.getclass()), 0, ex.getweight(testweight))
        
        for classifier in classifiers:
            cr = classifier(ex, orange.GetBoth)
            te.addResult(cr[0], cr[1])
        testResults.results.append(te)
        
    return testResults
