import orange, random, math, orngStat, orngTest

def LWRLearner(data = None, weightID = 0, **argkw):
    x = LWRLearner_class(**argkw)
    return data and x(data, weightID) or x

class LWRLearner_class:
    def __init__(self, **argkw):
        self.__dict__.update(argkw)
        self.argkw = argkw

    def __call__(self, data, weightID):
        if getattr(self, "pythonclass", 0):
            return LWRClassifier(data, weightID, **self.argkw)
        else:
            return self.instance()(data, weightID)

    def instance(self):
        learner = orange.LWRLearner()

        learner.distanceConstructor = getattr(self, "distanceConstructor", orange.ExamplesDistanceConstructor_Euclidean())
        if hasattr(self, "linRegLearner"):
            learner.linRegLearner = self.linRegLearner
        else:
            learner.linRegLearner = orange.LinRegLearner()
            for attrname in ["iterativeSelection", "multinomialTreatment", "Fin", "Fout"]:
                if hasattr(self, attrname):
                    setattr(learner.linRegLearner, attrname, getattr(self, attrname))
        for attrname in ["k", "rankWeight"]:
            if hasattr(self, attrname):
                setattr(learner, attrname, getattr(self, attrname))
        return learner

    
class LWRClassifier:
    def __init__(self, data, weightID=0, **argkw):
        self.k = 10
        self.__dict__.update(argkw)
        self.data = data
        self.weightID = weightID
        self.distWeightID = orange.newmetaid()
        self.findNearest = orange.FindNearestConstructor_BruteForce(
            data, weightID, self.distWeightID,
            distanceConstructor = getattr(self, "distanceConstructor", orange.ExamplesDistanceConstructor_Euclidean()))
        self.linregLearner = orange.LinRegLearner(iterativeSelection = getattr(self, "iterativeSelection", 3))
        
    def __call__(self, example, returnWhat=orange.Classifier.GetValue):
        nearest = self.findNearest(self.k, example)
        if self.k==1:
            pred = nearest[0].getclass()
        else:
            sigma = -math.log(1000) / nearest[-1][self.distWeightID]**2
            for n in nearest:
                n[self.distWeightID] = math.exp(n[self.distWeightID]**2 * sigma)
            lrclassifier = self.linregLearner(nearest, self.distWeightID)
            print " "*10,
            for attr in lrclassifier.domain:
                print "%s: %5.3f" % (attr.name, lrclassifier.coefficients[attr]),
            print
            pred = lrclassifier(example)
        if returnWhat==orange.Classifier.GetValue:
            return pred
        prob = orange.ContDistribution({float(pred):1.0})
        if returnWhat==orange.Classifier.GetProbabilities:
            return prob
        else:
            return pred, prob
