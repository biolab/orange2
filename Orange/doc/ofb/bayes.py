# Description: Class that implements the naive Bayesian learner and classifier (warning: just for educational purposes, for real, use naive Bayes as implemented in core Orange)
# Category:    modelling
# Referenced:  c_nb.htm

import orange

class Learner(object):
    def __new__(cls, examples=None, **kwds):
        learner = object.__new__(cls, **kwds)
        if examples:
            learner.__init__(**kwds)
            return learner(examples)
        else:
            return learner

    def __init__(self, m=0.0, name='std naive bayes', **kwds):
        self.__dict__.update(kwds)
        self.m = m
        self.name = name

    def __call__(self, examples, weight=None, **kwds):
        for k in kwds.keys():
            self.__dict__[k] = kwds[k]
        domain = examples.domain

        # first, compute class probabilities
        n_class = [0.] * len(domain.classVar.values)
        for e in examples:
            n_class[int(e.getclass())] += 1

        p_class = [0.] * len(domain.classVar.values)
        for i in range(len(domain.classVar.values)):
            p_class[i] = n_class[i] / len(examples)

        # count examples with specific attribute and
        # class value, pc[attribute][value][class]
        
        # initialization of pc
        pc = []
        for i in domain.attributes:
            p = [[0.]*len(domain.classVar.values) for i in range(len(i.values))]
            pc.append(p)

        # count instances, store them in pc
        for e in examples:
            c = int(e.getclass())
            for i in range(len(domain.attributes)):
                if not e[i].isSpecial():
                    pc[i][int(e[i])][c] += 1.0

        # compute conditional probabilities
        for i in range(len(domain.attributes)):
            for j in range(len(domain.attributes[i].values)):
                for k in range(len(domain.classVar.values)):
                    pc[i][j][k] = (pc[i][j][k] + self.m * p_class[k])/ \
                                  (n_class[k] + self.m)

        return Classifier(m = self.m, domain=domain, p_class=p_class, \
                                       p_cond=pc, name=self.name)    

class Classifier:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __call__(self, example, result_type=orange.GetValue):
        # compute the class probabilities
        p = map(None, self.p_class)
        for c in range(len(self.domain.classVar.values)):
            for a in range(len(self.domain.attributes)):
                if not example[a].isSpecial():
                    p[c] *= self.p_cond[a][int(example[a])][c]
                    
        # normalize probabilities to sum to 1
        sum =0.
        for pp in p: sum += pp
        if sum>0:
            for i in range(len(p)): p[i] = p[i]/sum
            
        # find the class with highest probability
        v_index = p.index(max(p))
        v = orange.Value(self.domain.classVar, v_index)

        # return the value based on requested return type
        if result_type == orange.GetValue:
            return v
        if result_type == orange.GetProbabilities:
            return p
        return (v,p)
        
    def show(self):
        print 'm=', self.m
        print 'class prob=', self.p_class
        print 'cond prob=', self.p_cond
