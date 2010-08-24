# Author: Blaz Zupan <blaz.zupan at fri.uni-lj.si>

"""
Feature interaction analysis. Implements 3-way attribute interactions as proposed
by Aleks Jakulin in his Ph. D. Thesis. Replaces his own orange module (orngInteract)
for the reasons of speed and simpler interface. Introduces sampling-based p-value
estimation.
"""

import operator
import random
from math import log

import orange
import orngTest
import orngStat
import orngCI
import itertools  
import warnings
import copy
import bisect

#def entropy_frequency_matrix(m):
#    n = sum(sum(v) for v in m)
#    return -sum(sum(plogp(x/n) for x in v) for v in m)

def flatten(matrix):
   """return a list of matrix elements"""
   return reduce(operator.add, [list(vector) for vector in matrix])

def p2f(vector):
    """convert list of frequencies to list of probabilities"""
    s = float(sum(vector))
    return [x/s for x in vector]

def printsym(sym, headers=False):
    """print out orange's sym matrix"""
    print
    if headers: print "    " + " ".join(["%-7d" % e for e in range(len(sym[0]))])
    for (i, line) in enumerate(sym):
        if headers: print "%2d" % i,
        print " ".join(["%7.4f" % e for e in line])
        
def attribute_pairs(data, n=None):
    """iterator through attribute pairs for given data set"""
    return itertools.combinations(data.domain.attributes, 2)

# information-based scoring

def plogp(p):
    return 0. if p==0. else p * log(p, 2.)

def _entropy(ps):
    """entropy computed from the vector of probabilities"""
    return -sum(plogp(p) for p in ps)

def entropy(x, data):
    """entropy of an attribute x from dataset data"""
    if type(x)==orange.EnumVariable:
        return _entropy(p2f(orange.Distribution(x, data)))
    if type(x)==list:
        if len(x)==2: # joint entropy of a pair of attributes
            c = orange.ContingencyAttrAttr(x, y, data)
            return _entropy(p2f(flatten(c)))
        else: # joint entropy of for a set of attributes
            pass

def joint_entropy(x, y, data):
    """return H(x,y), a joint entropy of attributes x and y"""
    c = orange.ContingencyAttrAttr(x, y, data)
    return _entropy(p2f(flatten(c)))

def conditional_entropy(x, y, data):
    """return H(x|y), a conditional entropy of attributes x and y"""
    return joint_entropy(x, y, data) - entropy(x, data) # H(x|y) = H(x,y) - H(x) 

class Mutual_information():
    def __init__(self, data):
        self.hs = dict([(x, entropy(x, data)) for x in data.domain])
        self.data = data
    def __call__(self, x, y):
        hxy = joint_entropy(x, y, self.data)
        return self.hs[x] + self.hs[y] - hxy
    
def mutual_information_matrix(data):
    """return a matrix with mutual information for attribute pairs"""
    atts = data.domain.attributes
    mis = orange.SymMatrix(len(atts))
    for a in range(len(atts)-1):
        for b in range(a+1, len(atts)):
            mis[a,b] = mutual_information(atts[a], atts[b], data)
    return mis

def conditional_entropy(x, y, data):
    """returns conditional entropy H(X|Y) given attributes X and Y
    where H(X|Y) = H(X,Y) - H(Y)"""
    hxy = joint_entropy(x, y, data)
    hy = entropy(y, data)
    return hxy - hy

class Interaction:
    """
    Two-way attribute interactions (feature synergies).
    """
    def __init__(self, data, p_values=False, samples=10000, permutations=100):
        self.data = data
        self.measure = orange.MeasureAttribute_info
        self.gain = dict([(a, self.measure(a, data)) for a in data.domain.attributes])
        self.class_entropy =  entropy(data.domain.classVar, data)
        self.samples = samples
        self.permutations = permutations
        self.p_values = p_values
        if p_values:
            self.score_dist = self.compute_score_dist()
    
    def compute_score_dist(self):
        """Distribution (list) of interaction scores obtained by permutation analysis"""
        
        def permute_class():
            random.shuffle(classvalues)
            for v, d in itertools.izip(classvalues, self.data):
                d.setclass(v)
                
        orig_classvalues = [d.getclass() for d in self.data]
        classvalues = copy.copy(orig_classvalues)
        attributes = self.data.domain.attributes
        samples_in_permutations = self.samples / self.permutations
        self.permuted_scores = []
        for _ in range(self.permutations):
            permute_class()
            scores = [self.get_score(random.choice(attributes), random.choice(attributes)) for _ in range(samples_in_permutations)]
            self.permuted_scores.extend(scores)
        self.permuted_scores_len = float(len(self.permuted_scores))
        self.permuted_scores.sort()

        # restore class values to original values
        for v, d in itertools.izip(orig_classvalues, self.data):
            d.setclass(v)
            
    def get_score(self, a1, a2):
        return orngCI.FeatureByCartesianProduct(self.data, (a1, a2), measure=self.measure)[1] - self.gain[a1] - self.gain[a2]
        
    def __call__(self, a1, a2):
        """Return two-attribute interaction and proportion of explained class entropy"""
        score = self.get_score(a1, a2)
        if self.p_values:
            return score, score/self.class_entropy, 1.0 - bisect.bisect(self.permuted_scores, score)/self.permuted_scores_len
        else:
            return score, score/self.class_entropy

#a1, a2 = data.domain.attributes[0], data.domain.attributes[1]
#ab, quality = orngCI.FeatureByCartesianProduct(data, [a1, a2], measure=orange.MeasureAttribute_info)
#r = mutual_information(a1, a2, data)

# meas = orange.MeasureAttribute_info()

def test():
    x = data.domain.attributes[1]
    y = data.domain.attributes[2]
    c = data.domain.classVar
    print "H(%s) = %5.5f" % (x.name, _entropy(p2f(orange.Distribution(x, data))))
    print "H(%s) = %5.5f" % (y.name, _entropy(p2f(orange.Distribution(y, data))))
    print "H(%s,%s)= %5.5f" % (x.name, y.name, joint_entropy(x, y, data))
    print "I(%s;%s)= %5.5f" % (x.name, y.name, mutual_information(x, y, data))
    print "H(%s|%s)= %5.5f" % (x.name, c.name, mutual_information(x, c, data))
    print "InfoGain = %5.5f" % orange.MeasureAttribute_info(x, data)
