# Description: An implementation of bagging (only bagging class is defined here)
# Category:    modelling
# Referenced:  c_bagging.htm

import orange, random

def Learner(examples=None, **kwds):
    learner = apply(Learner_Class, (), kwds)
    if examples:
        return learner(examples)
    else:
        return learner

class Learner_Class:
    def __init__(self, learner, t=10, name='bagged classifier'):
        self.t = t
        self.name = name
        self.learner = learner

    def __call__(self, examples, weight=None):
        r = random.Random()
        r.seed(0)

        n = len(examples)
        classifiers = []
        for i in range(self.t):
            selection = []
            for j in range(n):
                selection.append(r.randrange(n))
            data = examples.getitems(selection)
            classifiers.append(self.learner(data))
            
        return Classifier(classifiers = classifiers, name=self.name, domain=examples.domain)

class Classifier:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __call__(self, example, resultType = orange.GetValue):
        freq = [0.] * len(self.domain.classVar.values)
        for c in self.classifiers:
            freq[int(c(example))] += 1
        index = freq.index(max(freq))
        value = orange.Value(self.domain.classVar, index)
        for i in range(len(freq)):
            freq[i] = freq[i]/len(self.classifiers)
        if resultType == orange.GetValue: return value
        elif resultType == orange.GetProbabilities: return freq
        else: return (value, freq)
        
