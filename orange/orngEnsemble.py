import orange, math, orngTest, orngStat, random, orngMisc

# This function is built-in in Python 2.3,
# but we define it to be compatible with 2.2 as well
from operator import add
def sum(x):
    return reduce(add, x)

########################################################################
# Boosting

inf = 100000

def BoostedLearner(learner, examples=None, t=10, name='AdaBoost.M1'):
    learner = BoostedLearnerClass(learner, t, name)
    if examples:
        return learner(examples)
    else:
        return learner

class BoostedLearnerClass(orange.Learner):
    def __init__(self, learner, t, name):
        self.t = t
        self.name = name
        self.learner = learner

    def __call__(self, instances, origWeight = 0):
        weight = orange.newmetaid()
        if origWeight:
            for i in instances:
                i.setweight(weight, i.getweight(origWeight))
        else:
            instances.addMetaAttribute(weight, 1.0)
            
        n = len(instances)
        classifiers = []
        for i in range(self.t):
            epsilon = 0.0
            classifier = self.learner(instances, weight)
            corr = []
            for ex in instances:
                if classifier(ex) != ex.getclass():
                    epsilon += ex.getweight(weight)
                    corr.append(0)
                else:
                    corr.append(1)
            epsilon = epsilon / float(reduce(lambda x,y:x+y.getweight(weight), instances, 0))
            classifiers.append((classifier, epsilon and math.log((1-epsilon)/epsilon) or inf))
            if epsilon==0 or epsilon >= 0.499:
                if epsilon >= 0.499 and len(classifiers)>1:
                    del classifiers[-1]
                instances.removeMetaAttribute(weight)
                return BoostedClassifier(classifiers = classifiers, name=self.name, classvar=instances.domain.classVar)
            beta = epsilon/(1-epsilon)
            for e in range(n):
                if corr[e]:
                    instances[e].setweight(weight, instances[e].getweight(weight)*beta)
            f = 1/float(reduce(add, [e.getweight(weight) for e in instances]))
            for e in range(n):
                instances[e].setweight(weight, instances[e].getweight(weight)*f)

        instances.removeMetaAttribute(weight)
        return BoostedClassifier(classifiers = classifiers, name=self.name, classvar=instances.domain.classVar)

class BoostedClassifier(orange.Classifier):
    def __init__(self, **kwds):
        self.__dict__ = kwds

    def __call__(self, example, resultType = orange.GetValue):
        votes = [0.] * len(self.classvar.values)
        for c, e in self.classifiers:
            votes[int(c(example))] += e
        index = orngMisc.selectBestIndex(votes)
        value = orange.Value(self.classvar, index)
        if resultType == orange.GetValue:
            return value
        sv = sum(votes)
        for i in range(len(votes)):
            votes[i] = votes[i]/sv
        if resultType == orange.GetProbabilities:
            return votes
        else:
            return (value, votes)
        
########################################################################
# Bagging

def BaggedLearner(learner=None, t=10, name='Bagging', examples=None):
    learner = BaggedLearnerClass(learner, t, name)
    if examples:
        return learner(examples)
    else:
        return learner

class BaggedLearnerClass(orange.Learner):
    def __init__(self, learner, t, name):
        self.t = t
        self.name = name
        self.learner = learner

    def __call__(self, examples, weight=0):
        r = random.Random()
        r.seed(0)
        
        n = len(examples)
        classifiers = []
        for i in range(self.t):
            selection = []
            for i in range(n):
                selection.append(r.randrange(n))
            examples = orange.ExampleTable(examples)
            data = examples.getitems(selection)
            classifiers.append(self.learner(data, weight))
        return BaggedClassifier(classifiers = classifiers, name=self.name, classvar=examples.domain.classVar)

class BaggedClassifier(orange.Classifier):
    def __init__(self, **kwds):
        self.__dict__ = kwds

    def __call__(self, example, resultType = orange.GetValue):
        freq = [0.] * len(self.classvar.values)
        for c in self.classifiers:
            freq[int(c(example))] += 1
        index = freq.index(max(freq))
        value = orange.Value(self.classvar, index)
        if resultType == orange.GetValue:
            return value
        for i in range(len(freq)):
            freq[i] = freq[i]/len(self.classifiers)
        if resultType == orange.GetProbabilities:
            return freq
        else:
            return (value, freq)

########################################################################
# Random Forests

from math import sqrt, floor
import orngTree

class SplitConstructor_AttributeSubset(orange.TreeSplitConstructor):
    def __init__(self, scons, attributes, rand = None):
        self.scons = scons           # split constructor of original tree
        self.attributes = attributes # number of attributes to consider
        if rand:
            self.rand = rand             # a random generator
        else:
            self.rand = random.Random()
            self.rand.seed(0)

    def __call__(self, gen, weightID, contingencies, apriori, candidates, clsfr):
        cand = [1]*self.attributes + [0]*(len(candidates) - self.attributes)
        self.rand.shuffle(cand)
        # instead with all attributes, we will invoke split constructor only for the
        # subset of a attributes
        t = self.scons(gen, weightID, contingencies, apriori, cand, clsfr)
        return t

class RandomForestLearner(orange.Learner):
    def __new__(cls, examples=None, weight = 0, **kwds):
        self = orange.Learner.__new__(cls, **kwds)
        if examples:
            self.__init__(**kwds)
            return self.__call__(examples, weight)
        else:
            return self

    def __init__(self, learner=None, trees=100, attributes=None, name='Random Forest', rand=None, callback=None):
        """random forest learner"""
        self.trees = trees
        self.name = name
        self.learner = learner
        self.attributes = attributes
        self.callback = callback
        if rand:
            self.rand = rand
        else:
            self.rand = random.Random()
            self.rand.seed(0)

        if not learner:
            # tree learner assembled as suggested by Brieman (2001)
            smallTreeLearner = orngTree.TreeLearner(storeNodeClassifier = 0, storeContingencies=0, storeDistributions=1, minExamples=5).instance()
            smallTreeLearner.split.discreteSplitConstructor.measure = smallTreeLearner.split.continuousSplitConstructor.measure = orange.MeasureAttribute_gini()
            smallTreeLearner.split = SplitConstructor_AttributeSubset(smallTreeLearner.split, attributes, self.rand)
            self.learner = smallTreeLearner

    def __call__(self, examples, weight=0):
        # if number of attributes for subset is not set, use square root
        if hasattr(self.learner.split, 'attributes') and not self.learner.split.attributes:
            self.learner.split.attributes = int(sqrt(len(examples.domain.attributes)))

        n = len(examples)
        # build the forest
        classifiers = []
        for i in range(self.trees):
            # draw bootstrap sample
            selection = []
            for j in range(n):
                selection.append(self.rand.randrange(n))
            data = examples.getitems(selection)
            # build the model from the bootstrap sample
            classifiers.append(self.learner(data))
            if self.callback:
                self.callback()
            # if self.callback: self.callback((i+1.)/self.trees)

        return RandomForestClassifier(classifiers = classifiers, name=self.name, domain=examples.domain, classVar=examples.domain.classVar)
        
class RandomForestClassifier(orange.Classifier):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __call__(self, example, resultType = orange.GetValue):
        from operator import add

        # voting for class probabilities
        if resultType == orange.GetProbabilities or resultType == orange.GetBoth:
            cprob = [0.] * len(self.domain.classVar.values)
            for c in self.classifiers:
                a = [x for x in c(example, orange.GetProbabilities)]
                cprob = map(add, cprob, a)
            norm = sum(cprob)
            for i in range(len(cprob)):
                cprob[i] = cprob[i]/norm

        # voting for crisp class membership, notice that
        # this may not be the same class as one obtaining the
        # highest probability through probability voting
        if resultType == orange.GetValue or resultType == orange.GetBoth:
            cfreq = [0] * len(self.domain.classVar.values)
            for c in self.classifiers:
                cfreq[int(c(example))] += 1
            index = cfreq.index(max(cfreq))
            cvalue = orange.Value(self.domain.classVar, index)

        if resultType == orange.GetValue: return cvalue
        elif resultType == orange.GetProbabilities: return cprob
        else: return (cvalue, cprob)


##########################################################
### MeasureAttribute_randomForests

class MeasureAttribute_randomForests(orange.MeasureAttribute):

  def __init__(self, learner=None, trees = 100, attributes=None, rand=None):
    self.trees = trees
    self.learner = learner
    self.bufexamples = None
    self.attributes = attributes
    
    if self.learner == None:
      temp = RandomForestLearner(attributes=self.attributes)
      self.learner = temp.learner
    
    if hasattr(self.learner.split, 'attributes'):
      self.origattr = self.learner.split.attributes
      
    if rand:
      self.rand = rand             # a random generator
    else:
      self.rand = random.Random()
      self.rand.seed(0)

  def __call__(self, a1, a2, a3=None):
    """
    Returns importance of a given attribute. Can be given by index, 
    name or as a orange.Variable.
    """
    attrNo = None
    examples = None

    if type(a1) == int: #by attr. index
      attrNo, examples, apriorClass = a1, a2, a3
    elif type(a1) == type("a"): #by attr. name
      attrName, examples, apriorClass = a1, a2, a3
      attrNo = examples.domain.index(attrName)
    elif isinstance(a1, orange.Variable):
      a1, examples, apriorClass = a1, a2, a3
      atrs = [a for a in examples.domain.attributes]
      attrNo = atrs.index(a1)
    else:
      contingency, classDistribution, apriorClass = a1, a2, a3
      raise Exception("MeasureAttribute_rf can not be called with (contingency, classDistribution, apriorClass) as fuction arguments.")

    self.buffer(examples)

    return self.avimp[attrNo]*100/self.trees

  def importances(self, examples):
    """
    Returns importances of all attributes in dataset in a list. Buffered.
    """
    self.buffer(examples)
    
    return [a*100/self.trees for a in self.avimp]

  def buffer(self, examples):
    """
    recalcule importances if needed (new examples)
    """
    recalculate = False
    
    if examples != self.bufexamples:
      recalculate = True
    elif examples.version != self.bufexamples.version:
      recalculate = True
         
    if (recalculate):
      self.bufexamples = examples
      self.avimp = [0.0]*len(self.bufexamples.domain.attributes)
      self.acu = 0
      
      if hasattr(self.learner.split, 'attributes'):
          self.learner.split.attributes = self.origattr
      
      # if number of attributes for subset is not set, use square root
      if hasattr(self.learner.split, 'attributes') and not self.learner.split.attributes:
          self.learner.split.attributes = int(sqrt(len(examples.domain.attributes)))
      
      self.importanceAcu(self.bufexamples, self.trees, self.avimp)
      
  def getOOB(self, examples, selection, nexamples):
        ooblist = filter(lambda x: x not in selection, range(nexamples))
        return examples.getitems(ooblist)

  def numRight(self, oob, classifier):
        """
        returns a number of examples which are classified correcty
        """
        right = 0
        for el in oob:
            if (el.getclass() == classifier(el)):
                right = right + 1
        return right
    
  def numRightMix(self, oob, classifier, attr):
        """
        returns a number of examples  which are classified
        correctly even if an attribute is shuffled
        """
        n = len(oob)

        perm = range(n)
        self.rand.shuffle(perm)

        right = 0

        for i in range(n):
            ex = orange.Example(oob[i])
            ex[attr] = oob[perm[i]][attr]
            
            if (ex.getclass() == classifier(ex)):
                right = right + 1
                
        return right

  def importanceAcu(self, examples, trees, avimp):
        """
        accumulate avimp by importances for a given number of trees
        """
  

        n = len(examples)

        attrs = len(examples.domain.attributes)

        attrnum = {}
        for attr in range(len(examples.domain.attributes)):
           attrnum[examples.domain.attributes[attr].name] = attr            
   
        # build the forest
        classifiers = []  
        for i in range(trees):
            
            # draw bootstrap sample
            selection = []
            for j in range(n):
                selection.append(self.rand.randrange(n))
            data = examples.getitems(selection)
            
            # build the model from the bootstrap sample
            cla = self.learner(data)

            #prepare OOB data
            oob = self.getOOB(examples, selection, n)
            
            #right on unmixed
            right = self.numRight(oob, cla)
            
            presl = list(self.presentInTree(cla.tree, attrnum))
                      
            #randomize each attribute in data and test
            #only those on which there was a split
            for attr in presl:
                #calculate number of right classifications
                #if the values of this attribute are permutated randomly
                rightimp = self.numRightMix(oob, cla, attr)                
                avimp[attr] += (float(right-rightimp))/len(oob)

        self.acu += trees  

  def presentInTree(self, node, attrnum):
        """
        returns attributes present in tree (attributes that split)
        """

        if not node:
          return set([])

        if  node.branchSelector:
            j = attrnum[node.branchSelector.classVar.name]
            
            cs = set([])
            for i in range(len(node.branches)):
                s = self.presentInTree(node.branches[i], attrnum)
                cs = s | cs
            
            cs = cs | set([j])
            
            return cs
            
        else:
          return set([])


