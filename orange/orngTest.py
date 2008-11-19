import orange
from orngMisc import demangleExamples, getobjectname, printVerbose
import exceptions, cPickle, os, os.path

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

class ExperimentResults(object):
    def __init__(self, iterations, classifierNames, classValues, weights, baseClass=-1, **argkw):
        self.classValues = classValues
        self.classifierNames = classifierNames
        self.numberOfIterations = iterations
        self.numberOfLearners = len(classifierNames)
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

    def remove(self, index):
        """remove one learner from evaluation results"""
        for r in self.results:
            del r.classes[index]
            del r.probabilities[index]
        del self.classifierNames[index]
        self.numberOfLearners -= 1

    def add(self, results, index, replace=-1):
        """add evaluation results (for one learner)"""
        if len(self.results)<>len(results.results):
            raise SystemError, "mismatch in number of test cases"
        if self.numberOfIterations<>results.numberOfIterations:
            raise SystemError, "mismatch in number of iterations (%d<>%d)" % \
                  (self.numberOfIterations, results.numberOfIterations)
        if len(self.classifiers) and len(results.classifiers)==0:
            raise SystemError, "no classifiers in results"

        if replace < 0 or replace >= self.numberOfLearners: # results for new learner
            self.classifierNames.append(results.classifierNames[index])
            self.numberOfLearners += 1
            for i,r in enumerate(self.results):
                r.classes.append(results.results[i].classes[index])
                r.probabilities.append(results.results[i].probabilities[index])
            if len(self.classifiers):
                for i in range(self.numberOfIterations):
                    self.classifiers[i].append(results.classifiers[i][index])
        else: # replace results of existing learner
            self.classifierNames[replace] = results.classifierNames[index]
            for i,r in enumerate(self.results):
                r.classes[replace] = results.results[i].classes[index]
                r.probabilities[replace] = results.results[i].probabilities[index]
            if len(self.classifiers):
                for i in range(self.numberOfIterations):
                    self.classifiers[replace] = results.classifiers[i][index]

#### Experimental procedures

def leaveOneOut(learners, examples, pps=[], indicesrandseed="*", **argkw):
    """leave-one-out evaluation of learners on a data set"""
    (examples, weight) = demangleExamples(examples)
    return testWithIndices(learners, examples, range(len(examples)), indicesrandseed, pps, **argkw)
    # return testWithIndices(learners, examples, range(len(examples)), pps=pps, argkw)

# apply(testWithIndices, (learners, (examples, weight), indices, indicesrandseed, pps), argkw)


def proportionTest(learners, examples, learnProp, times=10,
                   strat=orange.MakeRandomIndices.StratifiedIfPossible,
                   pps=[], callback=None, **argkw):
    """train-and-test evaluation (train on a subset, test on remaing examples)"""
    # randomGenerator is set either to what users provided or to orange.RandomGenerator(0)
    # If we left it None or if we set MakeRandomIndices2.randseed, it would give same indices each time it's called
    randomGenerator = argkw.get("indicesrandseed", 0) or argkw.get("randseed", 0) or argkw.get("randomGenerator", 0)
    pick = orange.MakeRandomIndices2(stratified = strat, p0 = learnProp, randomGenerator = randomGenerator)
    
    examples, weight = demangleExamples(examples)
    if examples.domain.classVar.varType == orange.VarTypes.Discrete:
        values = list(examples.domain.classVar.values)
        basevalue = examples.domain.classVar.baseValue
    else:
        basevalue = values = None
        classVar = examples.domain.classVar
        if examples.domain.classVar.varType == orange.VarTypes.Discrete:
            values = classVar.values.native()
            baseValue = classVar.baseValue
        else:
            values = None
            baseValue = -1
        testResults = ExperimentResults(times, [l.name for l in learners], values, weight!=0, baseValue)

        # 
        # testResults = ExperimentResults(times, [l.name for l in learners],
        #                            values, weight!=0, basevalue)
        
    for time in range(times):
        indices = pick(examples)
        learnset = examples.selectref(indices, 0)
        testset = examples.selectref(indices, 1)
        learnAndTestOnTestData(learners, (learnset, weight), (testset, weight), testResults, time, pps, **argkw)
        if callback: callback()
    return testResults

def crossValidation(learners, examples, folds=10,
                    strat=orange.MakeRandomIndices.StratifiedIfPossible,
                    pps=[], indicesrandseed="*", **argkw):
    """cross-validation evaluation of learners"""
    (examples, weight) = demangleExamples(examples)
    if indicesrandseed!="*":
        indices = orange.MakeRandomIndicesCV(examples, folds, randseed=indicesrandseed, stratified = strat)
    else:
        randomGenerator = argkw.get("randseed", 0) or argkw.get("randomGenerator", 0)
        indices = orange.MakeRandomIndicesCV(examples, folds, stratified = strat, randomGenerator = randomGenerator)
    return testWithIndices(learners, (examples, weight), indices, indicesrandseed, pps, **argkw)


def learningCurveN(learners, examples, folds=10,
                   strat=orange.MakeRandomIndices.StratifiedIfPossible,
                   proportions=orange.frange(0.1), pps=[], **argkw):
    """construct a learning curve for learners"""
    seed = argkw.get("indicesrandseed", -1) or argkw.get("randseed", -1)
    if seed:
        randomGenerator = orange.RandomGenerator(seed)
    else:
        randomGenerator = argkw.get("randomGenerator", orange.RandomGenerator())
        
    if strat:
        cv=orange.MakeRandomIndicesCV(folds = folds, stratified = strat, randomGenerator = randomGenerator)
        pick=orange.MakeRandomIndices2(stratified = strat, randomGenerator = randomGenerator)
    else:
        cv=orange.RandomIndicesCV(folds = folds, stratified = strat, randomGenerator = randomGenerator)
        pick=orange.RandomIndices2(stratified = strat, randomGenerator = randomGenerator)
    return apply(learningCurve, (learners, examples, cv, pick, proportions, pps), argkw)


def learningCurve(learners, examples, cv=None, pick=None, proportions=orange.frange(0.1), pps=[], **argkw):
    verb = argkw.get("verbose", 0)
    cache = argkw.get("cache", 0)
    callback = argkw.get("callback", 0)

    for pp in pps:
        if pp[0]!="L":
            raise SystemError, "cannot preprocess testing examples"

    if not cv or not pick:    
        seed = argkw.get("indicesrandseed", -1) or argkw.get("randseed", -1)
        if seed:
            randomGenerator = orange.RandomGenerator(seed)
        else:
            randomGenerator = argkw.get("randomGenerator", orange.RandomGenerator())
        if not cv:
            cv = orange.MakeRandomIndicesCV(folds=10, stratified=orange.MakeRandomIndices.StratifiedIfPossible, randomGenerator = randomGenerator)
        if not pick:
            pick = orange.MakeRandomIndices2(stratified=orange.MakeRandomIndices.StratifiedIfPossible, randomGenerator = randomGenerator)

    examples, weight = demangleExamples(examples)
    folds = cv(examples)
    ccsum = hex(examples.checksum())[2:]
    ppsp = encodePP(pps)
    nLrn = len(learners)

    allResults=[]
    for p in proportions:
        printVerbose("Proportion: %5.3f" % p, verb)

        if (cv.randseed<0) or (pick.randseed<0):
            cache = 0
        else:
            fnstr = "{learningCurve}_%s_%s_%s_%s%s-%s" % ("%s", p, cv.randseed, pick.randseed, ppsp, ccsum)
            if "*" in fnstr:
                cache = 0

        conv = examples.domain.classVar.varType == orange.VarTypes.Discrete and int or float
        testResults = ExperimentResults(cv.folds, [l.name for l in learners], examples.domain.classVar.values.native(), weight!=0, examples.domain.classVar.baseValue)
        testResults.results = [TestedExample(folds[i], conv(examples[i].getclass()), nLrn, examples[i].getweight(weight))
                               for i in range(len(examples))]

        if cache and testResults.loadFromFiles(learners, fnstr):
            printVerbose("  loaded from cache", verb)
        else:
            for fold in range(cv.folds):
                printVerbose("  fold %d" % fold, verb)
                
                # learning
                learnset = examples.selectref(folds, fold, negate=1)
                learnset = learnset.selectref(pick(learnset, p0=p), 0)
                if not len(learnset):
                    continue
                
                for pp in pps:
                    learnset = pp[1](learnset)

                classifiers = [None]*nLrn
                for i in range(nLrn):
                    if not cache or not testResults.loaded[i]:
                        classifiers[i] = learners[i](learnset, weight)

                # testing
                for i in range(len(examples)):
                    if (folds[i]==fold):
                        # This is to prevent cheating:
                        ex = orange.Example(examples[i])
                        ex.setclass("?")
                        for cl in range(nLrn):
                            if not cache or not testResults.loaded[cl]:
                                cls, pro = classifiers[cl](ex, orange.GetBoth)
                                testResults.results[i].setResult(cl, cls, pro)
                if callback: callback()
            if cache:
                testResults.saveToFiles(learners, fnstr)

        allResults.append(testResults)
        
    return allResults


def learningCurveWithTestData(learners, learnset, testset, times=10,
                              proportions=orange.frange(0.1),
                              strat=orange.MakeRandomIndices.StratifiedIfPossible, pps=[], **argkw):
    verb = argkw.get("verbose", 0)

    learnset, learnweight = demangleExamples(learnset)
    testweight = demangleExamples(testset)[1]
    
    randomGenerator = argkw.get("indicesrandseed", 0) or argkw.get("randseed", 0) or argkw.get("randomGenerator", 0)
    pick = orange.MakeRandomIndices2(stratified = strat, randomGenerator = randomGenerator)
    allResults=[]
    for p in proportions:
        printVerbose("Proportion: %5.3f" % p, verb)
        testResults = ExperimentResults(times, [l.name for l in learners],
                                        testset.domain.classVar.values.native(),
                                        testweight!=0, testset.domain.classVar.baseValue)
        testResults.results = []
        
        for t in range(times):
            printVerbose("  repetition %d" % t, verb)
            learnAndTestOnTestData(learners, (learnset.selectref(pick(learnset, p), 0), learnweight),
                                   testset, testResults, t)

        allResults.append(testResults)
        
    return allResults

   
def testWithIndices(learners, examples, indices, indicesrandseed="*", pps=[], callback=None, **argkw):
    verb = argkw.get("verbose", 0)
    cache = argkw.get("cache", 0)
    storeclassifiers = argkw.get("storeclassifiers", 0) or argkw.get("storeClassifiers", 0)
    cache = cache and not storeclassifiers

    examples, weight = demangleExamples(examples)
    nLrn = len(learners)

    if not examples:
        raise SystemError, "Test data set with no examples"
    if not examples.domain.classVar:
        raise "Test data set without class attribute"
    
##    for pp in pps:
##        if pp[0]!="L":
##            raise SystemError, "cannot preprocess testing examples"

    nIterations = max(indices)+1
    if examples.domain.classVar.varType == orange.VarTypes.Discrete:
        values = list(examples.domain.classVar.values)
        basevalue = examples.domain.classVar.baseValue
    else:
        basevalue = values = None

    conv = examples.domain.classVar.varType == orange.VarTypes.Discrete and int or float        
    testResults = ExperimentResults(nIterations, [getobjectname(l) for l in learners], values, weight!=0, basevalue)
    testResults.results = [TestedExample(indices[i], conv(examples[i].getclass()), nLrn, examples[i].getweight(weight))
                           for i in range(len(examples))]

    if argkw.get("storeExamples", 0):
        testResults.examples = examples
        
    ccsum = hex(examples.checksum())[2:]
    ppsp = encodePP(pps)
    fnstr = "{TestWithIndices}_%s_%s%s-%s" % ("%s", indicesrandseed, ppsp, ccsum)
    if "*" in fnstr:
        cache = 0

    if cache and testResults.loadFromFiles(learners, fnstr):
        printVerbose("  loaded from cache", verb)
    else:
        for fold in range(nIterations):
            # learning
            learnset = examples.selectref(indices, fold, negate=1)
            if not len(learnset):
                continue
            testset = examples.selectref(indices, fold, negate=0)
            if not len(testset):
                continue
            
            for pp in pps:
                if pp[0]=="B":
                    learnset = pp[1](learnset)
                    testset = pp[1](testset)

            for pp in pps:
                if pp[0]=="L":
                    learnset = pp[1](learnset)
                elif pp[0]=="T":
                    testset = pp[1](testset)
                elif pp[0]=="LT":
                    (learnset, testset) = pp[1](learnset, testset)

            if not learnset:
                raise SystemError, "no training examples after preprocessing"

            if not testset:
                raise SystemError, "no test examples after preprocessing"

            classifiers = [None]*nLrn
            for i in range(nLrn):
                if not cache or not testResults.loaded[i]:
                    classifiers[i] = learners[i](learnset, weight)
            if storeclassifiers:    
                testResults.classifiers.append(classifiers)

            # testing
            tcn = 0
            for i in range(len(examples)):
                if (indices[i]==fold):
                    # This is to prevent cheating:
                    ex = orange.Example(testset[tcn])
                    ex.setclass("?")
                    tcn += 1
                    for cl in range(nLrn):
                        if not cache or not testResults.loaded[cl]:
                            cr = classifiers[cl](ex, orange.GetBoth)                                      
                            if cr[0].isSpecial():
                                raise "Classifier %s returned unknown value" % (classifiers[cl].name or ("#%i" % cl))
                            testResults.results[i].setResult(cl, cr[0], cr[1])
            if callback:
                callback()
        if cache:
            testResults.saveToFiles(learners, fnstr)
        
    return testResults


def learnAndTestOnTestData(learners, learnset, testset, testResults=None, iterationNumber=0, pps=[], **argkw):
    storeclassifiers = argkw.get("storeclassifiers", 0) or argkw.get("storeClassifiers", 0)
    storeExamples = argkw.get("storeExamples", 0)

    learnset, learnweight = demangleExamples(learnset)
    testset, testweight = demangleExamples(testset)
    storeclassifiers = argkw.get("storeclassifiers", 0) or argkw.get("storeClassifiers", 0)
    
    for pp in pps:
        if pp[0]=="B":
            learnset = pp[1](learnset)
            testset = pp[1](testset)

    for pp in pps:
        if pp[0]=="L":
            learnset = pp[1](learnset)
        elif pp[0]=="T":
            testset = pp[1](testset)
        elif pp[0]=="LT":
            learnset, testset = pp[1](learnset, testset)
            
    classifiers = [learner(learnset, learnweight) for learner in learners]
    for i in range(len(learners)): classifiers[i].name = getattr(learners[i], 'name', 'noname')
    testResults = testOnData(classifiers, (testset, testweight), testResults, iterationNumber, storeExamples)
    if storeclassifiers:
        testResults.classifiers.append(classifiers)
    return testResults


def learnAndTestOnLearnData(learners, learnset, testResults=None, iterationNumber=0, pps=[], **argkw):
    storeclassifiers = argkw.get("storeclassifiers", 0) or argkw.get("storeClassifiers", 0)
    storeExamples = argkw.get("storeExamples", 0)

    learnset, learnweight = demangleExamples(learnset)

    hasLorT = 0    
    for pp in pps:
        if pp[0]=="B":
            learnset = pp[1](learnset)
        else:
            hasLorT = 1

    if hasLorT:
        testset = orange.ExampleTable(learnset)
        for pp in pps:
            if pp[0]=="L":
                learnset = pp[1](learnset)
            elif pp[0]=="T":
                testset = pp[1](testset)
            elif pp[0]=="LT":
                learnset, testset = pp[1](learnset, testset)
    else:
        testset = learnset    

    classifiers = [learner(learnset, learnweight) for learner in learners]
    for i in range(len(learners)): classifiers[i].name = getattr(learners[i], "name", "noname")
    testResults = testOnData(classifiers, (testset, learnweight), testResults, iterationNumber, storeExamples)
    if storeclassifiers:
        testResults.classifiers.append(classifiers)
    return testResults


def testOnData(classifiers, testset, testResults=None, iterationNumber=0, storeExamples = False, **argkw):
    testset, testweight = demangleExamples(testset)

    if not testResults:
        classVar = testset.domain.classVar
        if testset.domain.classVar.varType == orange.VarTypes.Discrete:
            values = classVar.values.native()
            baseValue = classVar.baseValue
        else:
            values = None
            baseValue = -1
        testResults=ExperimentResults(1, [l.name for l in classifiers], values, testweight!=0, baseValue)

    examples = getattr(testResults, "examples", False)
    if examples and len(examples):
        # We must not modify an example table we do not own, so we clone it the
        # first time we have to add to it
        if not getattr(testResults, "examplesCloned", False):
            testResults.examples = orange.ExampleTable(testResults.examples)
            testResults.examplesCloned = True
        testResults.examples.extend(testset)
    else:
        # We do not clone at the first iteration - cloning might never be needed at all...
        testResults.examples = testset
    
    conv = testset.domain.classVar.varType == orange.VarTypes.Discrete and int or float
    for ex in testset:
        te = TestedExample(iterationNumber, conv(ex.getclass()), 0, ex.getweight(testweight))

        for classifier in classifiers:
            # This is to prevent cheating:
            ex2 = orange.Example(ex)
            ex2.setclass("?")
            cr = classifier(ex2, orange.GetBoth)
            te.addResult(cr[0], cr[1])
        testResults.results.append(te)
        
    return testResults
