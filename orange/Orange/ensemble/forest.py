from math import sqrt, floor
import Orange.core as orange
import Orange
import Orange.feature.scoring
import orngTree
import random


class RandomForestLearner(orange.Learner):
    """
    Just like bagging, classifiers in random forests are trained from bootstrap
    samples of training data. Here, classifiers are trees, but to increase
    randomness build in the way that at each node the best attribute is chosen
    from a subset of attributes in the training set. We closely follows the
    original algorithm (Brieman, 2001) both in implementation and parameter
    defaults.

    .. note::
        Random forest classifier uses decision trees induced from bootstrapped
        training set to vote on class of presented example. Most frequent vote
        is returned. However, in our implementation, if class probability is
        requested from a classifier, this will return the averaged probabilities
        from each of the trees.
    """
    def __new__(cls, examples=None, weight = 0, **kwds):
        self = orange.Learner.__new__(cls, **kwds)
        if examples:
            self.__init__(**kwds)
            return self.__call__(examples, weight)
        else:
            return self

    def __init__(self, learner=None, trees=100, attributes=None,\
                    name='Random Forest', rand=None, callback=None):
        """:param examples: If these are passed, the call returns 
                RandomForestClassifier, that is, creates the required set of
                decision trees, which, when presented with an examples, vote
                for the predicted class.
        :type examples: :class:`Orange.data.Table`
        :param trees: Number of trees in the forest.
        :type trees: int
        :param learner: Although not required, one can use this argument
                to pass one's own tree induction algorithm. If None is passed
                , RandomForestLearner will use Orange's tree induction 
                algorithm such that in induction nodes with less then 5 
                examples will not be considered for (further) splitting.
        :type learner: :class:`Orange.core.Learner`
        :param attributes: Number of attributes used in a randomly drawn
                subset when searching for best attribute to split the node
                in tree growing (default: None, and if kept this way, this
                is turned into square root of attributes in the training set,
                when this is presented to learner).
        :param rand: Random generator used in bootstrap sampling. 
                If none is passed, then Python's Random from random library is 
                used, with seed initialized to 0.
        :type rand: function
        :param callback:  A function to be called after every iteration of
                induction of classifier. This is called with parameter 
                (from 0.0 to 1.0) that gives estimates on learning progress.
        :param name: The name of the learner.
        :type name: string
        :rtype: :class:`Orange.ensemble.forest.RandomForestClassifier` or 
                :class:`Orange.ensemble.forest.RandomForestLearner`"""
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
            
        self.randstate = self.rand.getstate() #original state

        if not learner:
            # tree learner assembled as suggested by Brieman (2001)
            smallTreeLearner = orngTree.TreeLearner(
            storeNodeClassifier = 0, storeContingencies=0, 
            storeDistributions=1, minExamples=5).instance()
            smallTreeLearner.split.discreteSplitConstructor.measure = \
                    smallTreeLearner.split.continuousSplitConstructor.measure =\
                        Orange.feature.scoring.Gini()
            smallTreeLearner.split = SplitConstructor_AttributeSubset(\
                    smallTreeLearner.split, attributes, self.rand)
            self.learner = smallTreeLearner

    def __call__(self, examples, weight=0):
        """Learn from the given table of data instances.
        
        :param instances: Data instances to learn from.
        :type instances: Orange.data.Table
        :param origWeight: Weight.
        :type origWeight: int
        :rtype: :class:`Orange.ensemble.forest.RandomForestClassifier`"""
        # if number of attributes for subset is not set, use square root
        if hasattr(self.learner.split, 'attributes') and\
                    not self.learner.split.attributes:
            self.learner.split.attributes = int(sqrt(\
                    len(examples.domain.attributes)))

        self.rand.setstate(self.randstate) #when learning again, set the same state

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

        return RandomForestClassifier(classifiers = classifiers, name=self.name,\
                    domain=examples.domain, classVar=examples.domain.classVar)
        
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
            cvalue = Orange.data.Value(self.domain.classVar, index)

        if resultType == orange.GetValue: return cvalue
        elif resultType == orange.GetProbabilities: return cprob
        else: return (cvalue, cprob)

### MeasureAttribute_randomForests

class MeasureAttribute_randomForests(orange.MeasureAttribute):
    def __init__(self, learner=None, trees = 100, attributes=None, rand=None):
        """:param trees: Number of trees in the forest.
        :type trees: int
        :param learner: Although not required, one can use this argument to pass
            one's own tree induction algorithm. If None is 
            passed, :class:`Orange.ensemble.forest.MeasureAttribute` will 
            use Orange's tree induction algorithm such that in 
            induction nodes with less then 5 examples will not be 
            considered for (further) splitting.
        :type learner: None or :class:`Orange.core.Learner`
        :param attributes: Number of attributes used in a randomly drawn
            subset when searching for best attribute to split the node in tree
            growing (default: None, and if kept this way, this is turned into
            square root of attributes in example set).
        :type attributes: int
        :param rand: Random generator used in bootstrap sampling. If None is 
            passed, then Python's Random from random library is used, with seed
            initialized to 0."""
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
          self.rand = rand  # a random generator
        else:
          self.rand = random.Random()
          self.rand.seed(0)

    def __call__(self, a1, a2, a3=None):
        """Return importance of a given attribute. Can be given by index, 
        name or as a Orange.data.feature.Feature."""
        attrNo = None
        examples = None

        if type(a1) == int: #by attr. index
          attrNo, examples, apriorClass = a1, a2, a3
        elif type(a1) == type("a"): #by attr. name
          attrName, examples, apriorClass = a1, a2, a3
          attrNo = examples.domain.index(attrName)
        elif isinstance(a1, Orange.data.feature.Feature):
          a1, examples, apriorClass = a1, a2, a3
          atrs = [a for a in examples.domain.attributes]
          attrNo = atrs.index(a1)
        else:
          contingency, classDistribution, apriorClass = a1, a2, a3
          raise Exception("MeasureAttribute_rf can not be called with (\
                contingency,classDistribution, apriorClass) as fuction arguments.")

        self.buffer(examples)

        return self.avimp[attrNo]*100/self.trees

    def importances(self, examples):
        """Return importances of all attributes in dataset in a list.
        Buffered."""
        self.buffer(examples)
    
        return [a*100/self.trees for a in self.avimp]

    def buffer(self, examples):
        """Recalcule importances if needed (new examples)."""
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
          if hasattr(self.learner.split, 'attributes') and not\
                    self.learner.split.attributes:
              self.learner.split.attributes = int(sqrt(\
                            len(examples.domain.attributes)))
      
          self.importanceAcu(self.bufexamples, self.trees, self.avimp)
      
    def getOOB(self, examples, selection, nexamples):
        ooblist = filter(lambda x: x not in selection, range(nexamples))
        return examples.getitems(ooblist)

    def numRight(self, oob, classifier):
        """Return a number of examples which are classified correcty."""
        right = 0
        for el in oob:
            if (el.getclass() == classifier(el)):
                right = right + 1
        return right
    
    def numRightMix(self, oob, classifier, attr):
        """Return a number of examples  which are classified 
        correctly even if an attribute is shuffled."""
        n = len(oob)

        perm = range(n)
        self.rand.shuffle(perm)

        right = 0

        for i in range(n):
            ex = Orange.data.Instance(oob[i])
            ex[attr] = oob[perm[i]][attr]
            
            if (ex.getclass() == classifier(ex)):
                right = right + 1
                
        return right

    def importanceAcu(self, examples, trees, avimp):
        """Accumulate avimp by importances for a given number of trees."""
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
        """Return attributes present in tree (attributes that split)."""
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

class SplitConstructor_AttributeSubset(orange.TreeSplitConstructor):
    def __init__(self, scons, attributes, rand = None):
        import random
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
        # instead with all attributes, we will invoke split constructor 
        # only for the subset of a attributes
        t = self.scons(gen, weightID, contingencies, apriori, cand, clsfr)
        return t