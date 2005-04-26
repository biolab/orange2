# orngWrap.Tune1Parameter(object=bayes, parameter='m', values=[0,10,20,30])
import orange

class TuneParameters:
    returnNone=0
    returnParameters=1
    returnLearner=2
    returnClassifier=3
    
    def findobj(self, name):
        import string
        names=string.split(name, ".")
        lastobj=self.object
        for i in names[:-1]:
            lastobj=getattr(lastobj, i)
        return lastobj, names[-1]
        

# The class needs to be given the following attributes
#   object     - the learning algorithm to be fitter
#   parameter  - a string or a list of strings with parameter(s) to fit
#   values     - possible values of the parameter
#                (eg <object>.<parameter> = <value>[i])
#   evaluate   - statistics to evaluate (default: orngStat.CA)
#   compare    - function to compare (default: cmp - the bigger the better)
#   returnWhat - tells whether to return values of parameters, a fitted
#                learner, the best classifier or None. "object" is left
#                with optimal parameters in any case
class Tune1Parameter(TuneParameters):
    def __init__(self, **keyw):
        self.__dict__=keyw

    def __call__(self, table, weight=None, folds=5, verbose=0):
        import types, whrandom
        import orange, orngTest, orngStat

        verbose = verbose or getattr(self, "verbose", 0)
        
        if (type(self.parameter)==types.ListType) or (type(self.parameter)==types.TupleType):
            to_set = [self.findobj(ld) for ld in self.parameter]
        else:
            to_set = [self.findobj(self.parameter)]

        cvind = orange.MakeRandomIndicesCV(table, folds)
        bestres = None

        evaluate = getattr(self, "evaluate", orngStat.CA)
        compare = getattr(self, "compare", cmp)

        # now, we need to set <lastobj>.<fieldname> to values in self.parameter[1]
        for par in self.values:
            for i in to_set:
                setattr(i[0], i[1], par)

            if weight:
                res = evaluate(orngTest.testWithIndices([self.object], (table, weight), cvind))
            else:
                res = evaluate(orngTest.testWithIndices([self.object], (table), cvind))
            if verbose==2:
                print 'orngWrap:\n', par, res

            if not bestres:
                bestres, bestpar, wins = res, par, 1
            else:
                r=compare(res, bestres)
                if (r>0):
                    bestres, bestpar, wins = res, par, 1
                elif (r==0):
                    if not whrandom.randint(0, wins):
                        bestres, bestpar = res, par
                    wins=wins+1

        returnWhat = getattr(self, "returnWhat", Tune1Parameter.returnClassifier)

        if verbose:
            print "*** Optimal parameter: %s = %s" % (self.parameter, bestpar)
            
        for i in to_set:
            setattr(i[0], i[1], bestpar)

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


# The class needs to be given the following attributes
#   object     - the learning algorithm to be fitter
#   parameters - a list of tuples with parameters to be fitted and the
#                corresponding possible values, [(parameter(s), values), ...]
#                (eg <object>.<parameter[j]> = <value[j]>[i])
#   evaluate   - statistics to evaluate (default: orngStat.CA)
#   compare    - function to compare (default: cmp - the bigger the better)
#   returnWhat - tells whether to return values of parameters, a fitted
#                learner, the best classifier or None. "object" is left
#                with optimal parameters in any case
class TuneMParameters(TuneParameters):
    def __init__(self, **keyw):
        self.__dict__ = keyw

    def __call__(self, table, weight=None, folds=5, verbose=0):
        import types, whrandom
        import orange, orngTest, orngStat, orngMisc

        verbose = verbose or getattr(self, "verbose", 0)
        
        to_set = []
        parnames = []
        for par in self.parameters:
            if (type(par[0])==types.ListType) or (type(par[0])==types.TupleType):
                to_set.append([self.findobj(ld) for ld in par[0]])
                parnames.append(par[0])
            else:
                to_set.append([self.findobj(par[0])])
                parnames.append([par[0]])

        cvind = orange.MakeRandomIndicesCV(table, folds)
        bestres = None

        evaluate = getattr(self, "evaluate", orngStat.CA)
        compare = getattr(self, "compare", cmp)

        for valueindices in orngMisc.LimitedCounter([len(x[1]) for x in self.parameters]):
            values = [self.parameters[i][1][x] for i,x in enumerate(valueindices)]
            for pi, value in enumerate(values):
                for i, par in enumerate(to_set[pi]):
                    setattr(par[0], par[1], value)
                    if verbose==2:
                        print "%s: %s" % (parnames[pi][i], value)
                        
            if weight==None:
                res = evaluate(orngTest.testWithIndices([self.object], (table), cvind))
            else:
                res = evaluate(orngTest.testWithIndices([self.object], (table, weight), cvind))
            if verbose==2:
                print "===> Result: %s\n" % res

            if not bestres:
                bestres, bestpar, wins = res, values, 1
            else:
                r = compare(res, bestres)
                if r > 0:
                    bestres, bestpar, wins = res, values, 1
                elif r == 0:
                    if not whrandom.randint(0, wins):
                        bestres, bestpar = res, values
                    wins = wins+1

        returnWhat = getattr(self, "returnWhat", Tune1Parameter.returnClassifier)

        if verbose:
            print "*** Optimal set of parameters: ",
            for pi, value in enumerate(bestpar):
                for i, par in enumerate(to_set[pi]):
                    setattr(par[0], par[1], value)
                    if verbose:
                        print "%s: %s" % (parnames[pi][i], value),
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



def ThresholdLearner(learner, examples = None, weightId = 0, **kwds):
    lr = apply(ThresholdLearner_Class, (learner, ), kwds)
    if examples:
        return lr(examples, weightId)
    else:
        return lr

class ThresholdLearner_Class(orange.Learner):
    def __init__(self, learner, **kwds):
        self.learner = learner
        for k, e in kwds.items():
            self.setattr(k, e)

    def __call__(self, examples, weightID = 0):
        if not hasattr(self, "learner"):
            raise "learner not set"
        
        classifier = self.learner(examples, weightID)
        threshold, optCA, curve = orange.ThresholdCA(classifier, examples, weightID)
#        print "Threshold %5.3f, CA %5.3f" % (threshold, optCA)
        if getattr(self, "storeCurve", 0):
            return ThresholdClassifier(classifier, threshold, curve = curve)
        else:
            return ThresholdClassifier(classifier, threshold)

class ThresholdClassifier(orange.Classifier):
    def __init__(self, classifier, threshold, **kwds):
        self.classifier = classifier
        self.threshold = threshold
        for k, e in kwds.items():
            self.setattr(k, e)

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
    
class ThresholdLearner_fixed_Class(orange.Learner):
    def __init__(self, learner, threshold, **kwds):
        self.learner = learner
        self.threshold = threshold
        for k, e in kwds.items():
            self.setattr(k, e)

    def __call__(self, examples, weightID = 0):
        if not hasattr(self, "learner"):
            raise "learner not set"
        if len(examples.domain.classVar.values)!=2:
            raise "ThresholdLearner handles binary classes only"
        
        return ThresholdClassifier(self.learner(examples, weightID), self.threshold)
