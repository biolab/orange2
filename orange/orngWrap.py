import orange

# The class needs to be given
#   object     - the learning algorithm to be fitted
#   evaluate   - statistics to evaluate (default: orngStat.CA)
#   folds      - the number of folds for internal cross validation
#   compare    - function to compare (default: cmp - the bigger the better)
#   returnWhat - tells whether to return values of parameters, a fitted
#                learner, the best classifier or None. "object" is left
#                with optimal parameters in any case
class TuneParameters(orange.Learner):
    returnNone=0
    returnParameters=1
    returnLearner=2
    returnClassifier=3
    
    def __new__(cls, examples = None, weightID = 0, **argkw):
        self = orange.Learner.__new__(cls, **argkw)
        self.__dict__.update(argkw)
        if examples:
            return self.__call__(examples, weightID)
        else:
            return self

    def findobj(self, name):
        import string
        names=string.split(name, ".")
        lastobj=self.object
        for i in names[:-1]:
            lastobj=getattr(lastobj, i)
        return lastobj, names[-1]
        
# Same arguments as TuneParameters, plus:
#   parameter  - a string or a list of strings with parameter(s) to fit
#   values     - possible values of the parameter
#                (eg <object>.<parameter> = <value>[i])
class Tune1Parameter(TuneParameters):
    def __call__(self, table, weight=None, verbose=0):
        import orngTest, orngStat, orngMisc

        verbose = verbose or getattr(self, "verbose", 0)
        evaluate = getattr(self, "evaluate", orngStat.CA)
        folds = getattr(self, "folds", 5)
        compare = getattr(self, "compare", cmp)
        returnWhat = getattr(self, "returnWhat", Tune1Parameter.returnClassifier)

        if (type(self.parameter)==list) or (type(self.parameter)==tuple):
            to_set = [self.findobj(ld) for ld in self.parameter]
        else:
            to_set = [self.findobj(self.parameter)]

        cvind = orange.MakeRandomIndicesCV(table, folds)
        findBest = orngMisc.BestOnTheFly(seed = table.checksum(), callCompareOn1st = True)
        tableAndWeight = weight and (table, weight) or table
        for par in self.values:
            for i in to_set:
                setattr(i[0], i[1], par)
            res = evaluate(orngTest.testWithIndices([self.object], tableAndWeight, cvind))
            findBest.candidate((res, par))
            if verbose==2:
                print '*** orngWrap  %s: %s:' % (par, res)

        bestpar = findBest.winner()[1]
        for i in to_set:
            setattr(i[0], i[1], bestpar)

        if verbose:
            print "*** Optimal parameter: %s = %s" % (self.parameter, bestpar)

        if returnWhat==Tune1Parameter.returnNone:
            return None
        elif returnWhat==Tune1Parameter.returnParameters:
            return bestpar
        elif returnWhat==Tune1Parameter.returnLearner:
            return self.object
        else:
            classifier = self.object(table)
            classifier.setattr("fittedParameter", bestpar)
            return classifier


# Same arguments as TuneParameters, plus
#   parameters - a list of tuples with parameters to be fitted and the
#                corresponding possible values, [(parameter(s), values), ...]
#                (eg <object>.<parameter[j]> = <value[j]>[i])
class TuneMParameters(TuneParameters):
    def __call__(self, table, weight=None, verbose=0):
        import orngTest, orngStat, orngMisc

        evaluate = getattr(self, "evaluate", orngStat.CA)
        folds = getattr(self, "folds", 5)
        compare = getattr(self, "compare", cmp)
        verbose = verbose or getattr(self, "verbose", 0)
        returnWhat = getattr(self, "returnWhat", Tune1Parameter.returnClassifier)
        progressCallback = getattr(self, "progressCallback", lambda i: None)
        
        to_set = []
        parnames = []
        for par in self.parameters:
            if (type(par[0])==list) or (type(par[0])==tuple):
                to_set.append([self.findobj(ld) for ld in par[0]])
                parnames.append(par[0])
            else:
                to_set.append([self.findobj(par[0])])
                parnames.append([par[0]])


        cvind = orange.MakeRandomIndicesCV(table, folds)
        findBest = orngMisc.BestOnTheFly(seed = table.checksum(), callCompareOn1st = True)
        tableAndWeight = weight and (table, weight) or table
        numOfTests = sum([len(x[1]) for x in self.parameters])
        milestones = set(range(0, numOfTests, max(numOfTests / 100, 1)))
        for itercount, valueindices in enumerate(orngMisc.LimitedCounter([len(x[1]) for x in self.parameters])):
            values = [self.parameters[i][1][x] for i,x in enumerate(valueindices)]
            for pi, value in enumerate(values):
                for i, par in enumerate(to_set[pi]):
                    setattr(par[0], par[1], value)
                    if verbose==2:
                        print "%s: %s" % (parnames[pi][i], value)
                        
            res = evaluate(orngTest.testWithIndices([self.object], tableAndWeight, cvind))
            if itercount in milestones:
                progressCallback(100.0 * itercount / numOfTests)
            
            findBest.candidate((res, values))
            if verbose==2:
                print "===> Result: %s\n" % res

        bestpar = findBest.winner()[1]
        if verbose:
            print "*** Optimal set of parameters: ",
        for pi, value in enumerate(bestpar):
            for i, par in enumerate(to_set[pi]):
                setattr(par[0], par[1], value)
                if verbose:
                    print "%s: %s" % (parnames[pi][i], value),
        if verbose:
            print

        if returnWhat==Tune1Parameter.returnNone:
            return None
        elif returnWhat==Tune1Parameter.returnParameters:
            return bestpar
        elif returnWhat==Tune1Parameter.returnLearner:
            return self.object
        else:
            classifier = self.object(table)
            classifier.fittedParameters = bestpar
            return classifier




class ThresholdLearner(orange.Learner):
    def __new__(cls, examples = None, weightID = 0, **kwds):
        self = orange.Learner.__new__(cls, **kwds)
        self.__dict__.update(kwds)
        if examples:
            return self.__call__(examples, weightID)
        else:
            return self

    def __call__(self, examples, weightID = 0):
        if not hasattr(self, "learner"):
            raise "learner not set"
        
        classifier = self.learner(examples, weightID)
        threshold, optCA, curve = orange.ThresholdCA(classifier, examples, weightID)
        if getattr(self, "storeCurve", 0):
            return ThresholdClassifier(classifier, threshold, curve = curve)
        else:
            return ThresholdClassifier(classifier, threshold)

class ThresholdClassifier(orange.Classifier):
    def __init__(self, classifier, threshold, **kwds):
        self.classifier = classifier
        self.threshold = threshold
        self.__dict__.update(kwds)

    def __call__(self, example, what = orange.Classifier.GetValue):
        probs = self.classifier(example, self.GetProbabilities)
        if what == self.GetProbabilities:
            return probs
        value = orange.Value(self.classifier.classVar, probs[1]>self.threshold)
        if what == orange.Classifier.GetValue:
            return value
        else:
            return (value, probs)

def ThresholdLearner_fixed(learner, threshold, examples = None, weightId = 0, **kwds):
    lr = apply(ThresholdLearner_fixed_Class, (learner, threshold), kwds)
    if examples:
        return lr(examples, weightId)
    else:
        return lr
    
class ThresholdLearner_fixed(orange.Learner):
    def __new__(cls, examples = None, weightID = 0, **kwds):
        self = orange.Learner.__new__(cls, **kwds)
        self.__dict__.update(kwds)
        if examples:
            return self.__call__(examples, weightID)
        else:
            return self

    def __call__(self, examples, weightID = 0):
        if not hasattr(self, "learner"):
            raise "learner not set"
        if not hasattr(self, "threshold"):
            raise "threshold not set"
        if len(examples.domain.classVar.values)!=2:
            raise "ThresholdLearner handles binary classes only"
        
        return ThresholdClassifier(self.learner(examples, weightID), self.threshold)


class PreprocessedLearner(object):
    def __init__(self, preprocessor = None, learner = None):
        if isinstance(preprocessor, list):
            self.preprocessors = preprocessor
        elif preprocessor is not None:
            self.preprocessors = [preprocessor]
        else:
            self.preprocessors = []
        #self.preprocessors = [orange.Preprocessor_addClassNoise(proportion=0.8)]
        if learner:
            self.wrapLearner(learner)
        
    def processData(self, data, weightId = None):
        import orange
        hadWeight = hasWeight = weightId is not None
        for preprocessor in self.preprocessors:
            t = preprocessor(data, weightId) if hasWeight else preprocessor(data)
            if isinstance(t, tuple):
                data, weightId = t
                hasWeight = True
            else:
                data = t
        if hadWeight:
            return data, weightId
        else:
            return data

    def wrapLearner(self, learner):
        class WrappedLearner(learner.__class__):
            preprocessor = self
            wrappedLearner = learner
            name = getattr(learner, "name", "")
            def __call__(self, data, weightId=0, getData = False):
                t = self.preprocessor.processData(data, weightId or 0)
                processed, procW = t if isinstance(t, tuple) else (t, 0)
                classifier = self.wrappedLearner(processed, procW)
                if getData:
                    return classifier, processed
                else:
                    return classifier # super(WrappedLearner, self).__call__(processed, procW)
        return WrappedLearner()