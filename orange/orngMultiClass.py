# ORANGE multiclass support
#    by Alex Jakulin (jakulin@acm.org)
#
#       based on:
#           Zadrozny, B.:
#           Reducing multiclass to binary by coupling probability estimates
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Version 1.7 (26/8/2002)
#
# - Added some intelligence to Zadrozny iterator. It now reduces tolerance
# if the convergence is not happening.
#
# Version 1.6 (1/11/2001)
#
# Multiclass module enables the support for multiclass problems with simple
# biclass margin classifiers.
# 
#
# To Do:
#   - ECOC as MCM
#

import orange


class MultiClassClassifier(orange.Classifier):
    def __init__(self, cmatrix, template, domain):
        self.cmatrix = cmatrix
        self.template = template
        self.domain = domain

    def normalize(self, np):
        sum = 0.0
        for i in np:
            sum += i
        return [w/sum for w in np]

    def getPD(self, example, ncv):
        # initial approximate for probability distribution
        r = 1.0/ncv
        p = [r]*ncv
        return p

    def __call__(self, example, format = orange.GetValue):
        ncv = len(self.domain.classVar.values)
        p = self.getPD(example, ncv)
        
        # find the max probability
        maxp = -1e200
        for i in range(len(p)):
            if p[i] > maxp:
                maxp = p[i]
                bestc = i
        v = self.domain.classVar(bestc)
        
        # return
        if format == orange.GetValue:
            return v
        if format == orange.GetBoth:
            return (v,p)
        if format == orange.GetProbabilities:
            return p
    

class MCPEZadrozny(MultiClassClassifier):
    def __init__(self, *args):
        apply(MultiClassClassifier.__init__, (self,) + args )
        self.iterations = 100
        self.tolerance = 1e-4

        # need the set definitions
        self.I = [[] for i in self.template[0]]  # which templates cover a class with +1
        self.J = [[] for i in self.template[0]]  # which templates cover a class with -1
        self.iI = [[] for i in self.template]    # which templates are covered by a class with +1
        self.iJ = [[] for i in self.template]    # which templates are covered by a class with -1
        for i in range(len(self.template)):
            for j in range(len(self.template[0])):
                if self.template[i][j] == -1:
                    self.J[j].append(i)
                    self.iJ[i].append(j)
                if self.template[i][j] == +1:
                    self.I[j].append(i)
                    self.iI[i].append(j)

    def getRB(self, phat):
        # computes the r on the basis of current probability estimate
        rhat = []
        for i in range(len(self.template)):
            pt = 0.0 # top
            for x in self.iI[i]:
                pt += phat[x]
            pb = 0.0 # bottom
            for x in self.iJ[i]:
                pb += phat[x]
            if(pt == 0.0):
                rhat.append(0.0)
            else:
                rhat.append(pt/(pb+pt))
        return rhat
              
    def getPD(self, example, k):
        # do all the experiments
        r = []
        n = []
        for i in self.cmatrix:
            r.append(i[0](example,orange.GetProbabilities)[1]) # probability the class is 1
            n.append(i[1]+0.0)

        # initial approximation
        phat = MultiClassClassifier.getPD(self,example,k)

        rt = []
        for c in range(k):
            tt = 1e-4  # prevent divide by 0
            for b in self.I[c]:
                tt += n[b]*r[b]
            for b in self.J[c]:
                tt += n[b]*(1.0-r[b])
            rt.append(tt)

        # do the iteration
        iterations = self.iterations
        tolerance = self.tolerance
        while iterations > 0:
            iterations -= 1
            rhat = self.getRB(phat)

            diff = 0.0 # testing for convergence

            # for all classes, compute new approximation
            for c in range(k):
                tt = 0.0
                for b in self.I[c]:
                    tt += n[b]*rhat[b]
                for b in self.J[c]:
                    tt += n[b]*(1.0-rhat[b])
                t = phat[c]*rt[c]/tt
                diff += (t-phat[c])*(t-phat[c])
                phat[c] = t

            phat = self.normalize(phat)
            if diff < tolerance:
                break
            if iterations == 0 and tolerance < 1.0:
                tolerance *= 10.0
                iterations = self.iterations
        if iterations <= 0:
            print "Zadrozny didn't converge. p=",phat
        return phat

#
# Friedman's method simply counts the # of wins, and
# weights the PD on the basis of this. Class frequencies
# not considered.
#
class MCPEFriedman(MultiClassClassifier):
    def getPD(self, example, ncv):
        # do all the experiments
        results = []
        for i in self.cmatrix:
            results.append(i[0](example))

        # count the wins
        wins = [0]*ncv
        assert(len(self.template)==len(results))
        assert(int(example.domain.classVar(1))==1)
        for i in range(len(self.template)):
            if int(results[i]) == 1:
                # winners are the +1's
                match = +1
            else:
                # winners are the -1's
                match = -1
            for j in range(ncv):
                if self.template[i][j] == match:
                    # the in class
                    wins[j] += 1

        # find the sum of winnings
        return self.normalize(wins)


#
# MCM's are matrix generators for the multiclass problems.
# Rows of the matrix indicate binary classifiers
# Columns of the matrix indicate attributes.
# Values mean: 0 - ignore this class, 1 - positive class, -1 - negative class
#
class MCMOneOne:
    def __call__(self, nc):
        m = []
        for i in range(nc):
            for j in range(nc):
                if i < j:
                    r = [0]*nc
                    r[i] = 1
                    r[j] = -1
                    m.append(r)
        return m
    
class MCMOneAll:
    def __call__(self, nc):
        m = []
        for i in range(nc):
            r = [-1]*nc
            r[i] = 1
            m.append(r)
        return m

class MCMOrdinal:    
    def __call__(self, nc):
        m = []
        for i in range(1,nc):
            r = [-1]*i+[1]*(nc-i)
            m.append(r)
        return m

class MultiClassLearner(orange.Learner):
    def __init__(self, learner, matrix = MCMOneOne, pestimator = MCPEFriedman, name='MultiClass'):
        self.learner = learner
        self.pestimator = pestimator
        self.matrix = matrix()
        self.name = name

    def __call__(self, examples, weight = 0):
        if examples.domain.classVar.varType != 1:
            raise "MultiClassLearner only works with discrete class"
        
        # simple handling for simple 2-class problems 
        if len(examples.domain.classVar.values) <= 2:
            if weight != 0:
                return self.learner(examples, weight)
            else:
                return self.learner(examples)

        # count the classes and generate the classifier matrix
        nc = len(examples.domain.classVar.values)
        nv = len(examples.domain.attributes)
        template = self.matrix(nc)

        # prepare the domain, and the new binary class
        bin = orange.EnumVariable(name="binary",values=['0','1'])
        b0 = bin(0)
        b1 = bin(1)
        nd = orange.Domain(examples.domain.attributes+[bin])
        
        # generate all classifiers
        cm = []
        for i in template:
            exs = orange.ExampleTable(nd)
            if weight != 0:
                exs.addMetaAttribute(1)
            for j in examples:
                if i[int(j.getclass())] == 1:
                    r = [j[x] for x in range(nv)]
                    r.append(b1)
                    x = orange.Example(nd,r)
                    if weight != 0:
                        x.setmeta(j.getMetaAttribute(weight),1)
                    exs.append(x)
                else:
                    if i[int(j.getclass())] == -1:
                        r = [j[x] for x in range(nv)]
                        r.append(b0)
                        x = orange.Example(nd,r)
                        if weight != 0:
                            x.setmeta(j.getMetaAttribute(weight),1)
                        exs.append(x)
            # prepare the classifier
            if len(exs) <= 0:
                raise "MultiClass: More than one of the declared class values do not appear in the data. Filter them out."
            if weight!=0:
                c = self.learner(exs,weight = 1)
            else:
                c = self.learner(exs)
            cm.append((c,len(exs)))
        return self.pestimator(cm, template, examples.domain)


