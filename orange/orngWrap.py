# orngWrap.Tune1Parameter(object=bayes, parameter='m', values=[0,10,20,30])
import orange

class Tune1Parameter:
    returnNone=0
    returnParameters=1
    returnLearner=2
    returnClassifier=3
    
    def __init__(self, **keyw):
        self.__dict__=keyw

    def findobj(self, name):
        import string
        names=string.split(name, ".")
        lastobj=self.object
        for i in names[:-1]:
            lastobj=getattr(lastobj, i)
        return lastobj, names[-1]
        
    def __call__(self, table, weight=None, folds=5, verbose=0):
        import types, whrandom
        import orange, orngTest, orngStat
        
        if (type(self.parameter)==types.ListType) or (type(self.parameter)==types.TupleType):
            to_set=[self.findobj(ld) for ld in self.parameter]
        else:
            to_set=[self.findobj(self.parameter)]

        cvind = orange.MakeRandomIndicesCV(table, folds)
        bestres = None

        evaluate = getattr(self, "evaluate", orngStat.CA)
        compare = getattr(self, "compare", cmp)

        # now, we need to set <lastobj>.<fieldname> to values in self.parameter[1]
        for par in self.values:
            for i in to_set:
                setattr(i[0], i[1], par)
            # self.evaluate could be, for example, orngEval.CA
            if weight==None:
                res=evaluate(orngTest.testWithIndices([self.object], (table), cvind))
            else:
                res=evaluate(orngTest.testWithIndices([self.object], (table, weight), cvind))
            if verbose:
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

        returnWhat = getattr(self, "returnWhat", Tune1Parameter.returnNone)

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
            classifier.fittedParameter = bestpar
            return classifier
##            setattr(self, "fittedParameter", bestpar)

def ThresholdLearner(learner, examples = None, weightId = 0, **kwds):
    lr = apply(ThresholdLearner_Class, (learner, ), kwds)
    if examples:
        return lr(examples, weightId)
    else:
        return lr

class ThresholdLearner_Class(orange.Learner):
    def __init__(self, learner, **kwds):
        self.learner = learner
        self.__dict__.update(kwds)

    def __call__(self, examples, weightID = 0):
        if not hasattr(self, "learner"):
            raise "learner not set"
        if len(examples.domain.classVar.values)!=2:
            raise "ThresholdLearner handles binary classes only"
        
        classifier = self.learner(examples, weightID)
        
        probs = {}
        N = N0 = 0
        for ex in examples:
            wei = ex.getweight(weightID)
            prob1 = classifier(ex, orange.Classifier.GetProbabilities)[1]
            probs.setdefault(prob1, 0.)
            if int(ex.getclass()):
                probs[prob1]-=wei
            else:
                probs[prob1]+=wei
                N0 += wei
            N += wei

        optcorr = optthresh = 0
        corr = N - N0
        pitems = probs.items()
        pitems.sort(lambda x,y:cmp(x[0], y[0]))
        for i in range(len(pitems)-1):
            corr += pitems[i][1]
            #print corr, pitems[i][0]
            if (corr<optcorr):
                continue
            tthresh = (pitems[i][0] + pitems[i+1][0])/2
            if (corr>optcorr) or (abs(tthresh-0.5) < abs(optthresh-0.5)):
                optcorr, optthresh = corr, tthresh

        print "Threshold %5.3f, CA %5.3f" % (optthresh, optcorr/N)
        return ThresholdClassifier(classifier, optthresh)

class ThresholdClassifier(orange.Classifier):
    def __init__(self, classifier, threshold):
        self.classifier = classifier
        self.threshold = threshold

    def __call__(self, example, what = orange.Classifier.GetValue):
        probs = self.classifier(example, self.GetProbabilities)
        if what == self.GetProbabilities:
            return probs
        value = orange.Value(self.classifier.classVar, probs[1]>self.threshold)
        if what == orange.Classifier.GetValue:
            return value
        else:
            return (value, probs)
