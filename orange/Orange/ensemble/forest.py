from math import sqrt, floor
import Orange.core as orange
import Orange
import Orange.feature.scoring
import random

class RandomForestLearner(orange.Learner):
    """
    Just like bagging, classifiers in random forests are trained from bootstrap
    samples of training data. Here, classifiers are trees. However, to increase
    randomness, classifiers are built so that at each node the best feature is
    chosen from a subset of features in the training set. We closely follow the
    original algorithm (Brieman, 2001) both in implementation and parameter
    defaults.
        
    :param learner: although not required, one can use this argument
            to pass one's own tree induction algorithm. If None is passed,
            RandomForestLearner will use Orange's tree induction 
            algorithm such that induction nodes with less than 5 
            data instances will not be considered for (further) splitting.
    :type learner: :class:`Orange.core.Learner`
    :param trees: number of trees in the forest.
    :type trees: int
    :param attributes: number of features used in a randomly drawn
            subset when searching for best feature to split the node
            in tree growing (default: None, and if kept this way, this
            is turned into square root of the number of features in the
            training set, when this is presented to learner).
    :type attributes: int
    :param rand: random generator used in bootstrap sampling. 
            If None is passed, then Python's Random from random library is 
            used, with seed initialized to 0.
    :type rand: function
    :param callback: a function to be called after every iteration of
            induction of classifier. This is called with parameter 
            (from 0.0 to 1.0) that gives estimates on learning progress.
    :param name: name of the learner.
    :type name: string
    :rtype: :class:`Orange.ensemble.forest.RandomForestClassifier` or 
            :class:`Orange.ensemble.forest.RandomForestLearner`
    """

    def __new__(cls, instances=None, weight = 0, **kwds):
        self = orange.Learner.__new__(cls, **kwds)
        if instances:
            self.__init__(**kwds)
            return self.__call__(instances, weight)
        else:
            return self

    def __init__(self, learner=None, trees=100, attributes=None,\
                    name='Random Forest', rand=None, callback=None):
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

    def __call__(self, instances, weight=0):
        """
        Learn from the given table of data instances.
        
        :param instances: data instances to learn from.
        :type instances: class:`Orange.data.Table`
        :param origWeight: weight.
        :type origWeight: int
        :rtype: :class:`Orange.ensemble.forest.RandomForestClassifier`
        """
        
        
        # If there is no learner we create our own
        
        if not self.learner:
            
            # tree learner assembled as suggested by Brieman (2001)
            smallTreeLearner = Orange.classification.tree.TreeLearner(
            storeNodeClassifier = 0, storeContingencies=0, 
            storeDistributions=1, minExamples=5).instance()
            
            # Use MSE on continuous class and Gini on discreete
            if instances.domain.class_var.var_type == Orange.data.variable.Continuous.Continuous:
                smallTreeLearner.split.discreteSplitConstructor.measure = \
                    smallTreeLearner.split.continuousSplitConstructor.measure =\
                        Orange.feature.scoring.MSE()
            else:
                smallTreeLearner.split.discreteSplitConstructor.measure = \
                    smallTreeLearner.split.continuousSplitConstructor.measure =\
                        Orange.feature.scoring.Gini()
            
            smallTreeLearner.split = SplitConstructor_AttributeSubset(\
                    smallTreeLearner.split, self.attributes, self.rand)
            self.learner = smallTreeLearner
        
        # if number of features for subset is not set, use square root
        if hasattr(self.learner.split, 'attributes') and\
                    not self.learner.split.attributes:
            self.learner.split.attributes = int(sqrt(\
                    len(instances.domain.attributes)))

        self.rand.setstate(self.randstate) #when learning again, set the same state

        n = len(instances)
        # build the forest
        classifiers = []
        for i in range(self.trees):
            # draw bootstrap sample
            selection = []
            for j in range(n):
                selection.append(self.rand.randrange(n))
            data = instances.getitems(selection)
            # build the model from the bootstrap sample
            classifiers.append(self.learner(data))
            if self.callback:
                self.callback()
            # if self.callback: self.callback((i+1.)/self.trees)

        return RandomForestClassifier(classifiers = classifiers, name=self.name,\
                    domain=instances.domain, classVar=instances.domain.classVar)
        
class RandomForestClassifier(orange.Classifier):
    """
    Random forest classifier uses decision trees induced from bootstrapped
    training set to vote on class of presented instance. Most frequent vote
    is returned. However, in our implementation, if class probability is
    requested from a classifier, this will return the averaged probabilities
    from each of the trees.

    When constructing the classifier manually, the following parameters can
    be passed:

    :param classifiers: a list of classifiers to be used.
    :type classifiers: list
    
    :param name: name of the resulting classifier.
    :type name: str
    
    :param domain: the domain of the learning set.
    :type domain: :class:`Orange.data.Domain`
    
    :param classVar: the class feature.
    :type classVar: :class:`Orange.data.variable.Variable`

    """
    def __init__(self, classifiers, name, domain, classVar, **kwds):
        self.classifiers = classifiers
        self.name = name
        self.domain = domain
        self.classVar = classVar
        self.__dict__.update(kwds)

    def __call__(self, instance, resultType = orange.GetValue):
        """
        :param instance: instance to be classified.
        :type instance: :class:`Orange.data.Instance`
        
        :param result_type: :class:`Orange.classification.Classifier.GetValue` or \
              :class:`Orange.classification.Classifier.GetProbabilities` or
              :class:`Orange.classification.Classifier.GetBoth`
        
        :rtype: :class:`Orange.data.Value`, 
              :class:`Orange.statistics.Distribution` or a tuple with both
        """
        from operator import add
        
        # handle discreete class
        
        if self.class_var.var_type == Orange.data.variable.Discrete.Discrete:
        
            # voting for class probabilities
            if resultType == orange.GetProbabilities or resultType == orange.GetBoth:
                prob = [0.] * len(self.domain.classVar.values)
                for c in self.classifiers:
                    a = [x for x in c(instance, orange.GetProbabilities)]
                    prob = map(add, prob, a)
                norm = sum(prob)
                cprob = Orange.statistics.distributions.Discrete(self.classVar)
                for i in range(len(prob)):
                    cprob[i] = prob[i]/norm
                
                
    
            # voting for crisp class membership, notice that
            # this may not be the same class as one obtaining the
            # highest probability through probability voting
            if resultType == orange.GetValue or resultType == orange.GetBoth:
                cfreq = [0] * len(self.domain.classVar.values)
                for c in self.classifiers:
                    cfreq[int(c(instance))] += 1
                index = cfreq.index(max(cfreq))
                cvalue = Orange.data.Value(self.domain.classVar, index)
    
            if resultType == orange.GetValue: return cvalue
            elif resultType == orange.GetProbabilities: return cprob
            else: return (cvalue, cprob)
        
        else:
            # Handle continuous class
            
            # voting for class probabilities
            if resultType == orange.GetProbabilities or resultType == orange.GetBoth:
                probs = [c(instance, orange.GetProbabilities) for c in self.classifiers]
                cprob = dict()
                for prob in probs:
                    a = dict(prob.items())
                    cprob = dict( (n, a.get(n, 0)+cprob.get(n, 0)) for n in set(a)|set(cprob) )
                cprob = Orange.statistics.distributions.Continuous(cprob)
                cprob.normalize()
                
            # gather average class value
            if resultType == orange.GetValue or resultType == orange.GetBoth:
                values = [c(instance).value for c in self.classifiers]
                cvalue = Orange.data.Value(self.domain.classVar, sum(values) / len(self.classifiers))
            
            if resultType == orange.GetValue: return cvalue
            elif resultType == orange.GetProbabilities: return cprob
            else: return (cvalue, cprob)

### MeasureAttribute_randomForests

class ScoreFeature(orange.MeasureAttribute):
    """
    :param learner: although not required, one can use this argument to pass
        one's own tree induction algorithm. If None is 
        passed, :class:`Orange.ensemble.forest.MeasureAttribute` will 
        use Orange's tree induction algorithm such that 
        induction nodes with less than 5 data instances will not be 
        considered for (further) splitting.
    :type learner: None or :class:`Orange.core.Learner`
    :param trees: number of trees in the forest.
    :type trees: int
    :param attributes: number of features used in a randomly drawn
            subset when searching for best feature to split the node
            in tree growing (default: None, and if kept this way, this
            is turned into square root of the number of features in the
            training set, when this is presented to learner).
    :type attributes: int
    :param rand: random generator used in bootstrap sampling. If None is 
        passed, then Python's Random from random library is used, with seed
        initialized to 0.
    """
    def __init__(self, learner=None, trees = 100, attributes=None, rand=None):

        self.trees = trees
        self.learner = learner
        self.bufinstances = None
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

    def __call__(self, feature, instances, apriorClass=None):
        """
        Return importance of a given feature.
        
        :param feature: feature to evaluate (by index, name or
            :class:`Orange.data.variable.Variable` object).
        :type feature: int, str or :class:`Orange.data.variable.Variable`.
        
        :param instances: data instances to use for importance evaluation.
        :type instances: :class:`Orange.data.Table`
        
        :param apriorClass: not used!
        
        """
        attrNo = None

        if type(feature) == int: #by attr. index
          attrNo  = feature
        elif type(feature) == type("a"): #by attr. name
          attrName = feature
          attrNo = instances.domain.index(attrName)
        elif isinstance(feature, Orange.data.variable.Variable):
          atrs = [a for a in instances.domain.attributes]
          attrNo = atrs.index(feature)
        else:
          raise Exception("MeasureAttribute_rf can not be called with (\
                contingency,classDistribution, apriorClass) as fuction arguments.")

        self.buffer(instances)

        return self.avimp[attrNo]*100/self.trees

    def importances(self, table):
        """
        Return importance of all features in the dataset as a list. The result
        is buffered, so repeated calls on the same (unchanged) dataset are
        computationally cheap.
        
        :param table: dataset of which the features' importance needs to be
            measured.
        :type table: :class:`Orange.data.Table` 

        """
        self.buffer(table)
    
        return [a*100/self.trees for a in self.avimp]

    def buffer(self, instances):
        """
        Recalculate importance of features if needed (ie. if it has been
        buffered for the given dataset yet).

        :param table: dataset of which the features' importance needs to be
            measured.
        :type table: :class:`Orange.data.Table` 

        """
        recalculate = False
    
        if instances != self.bufinstances:
          recalculate = True
        elif instances.version != self.bufinstances.version:
          recalculate = True
         
        if (recalculate):
          self.bufinstances = instances
          self.avimp = [0.0]*len(self.bufinstances.domain.attributes)
          self.acu = 0
      
          if hasattr(self.learner.split, 'attributes'):
              self.learner.split.attributes = self.origattr
      
          # if number of attributes for subset is not set, use square root
          if hasattr(self.learner.split, 'attributes') and not\
                    self.learner.split.attributes:
              self.learner.split.attributes = int(sqrt(\
                            len(instances.domain.attributes)))
      
          self.importanceAcu(self.bufinstances, self.trees, self.avimp)
      
    def getOOB(self, instances, selection, nexamples):
        ooblist = filter(lambda x: x not in selection, range(nexamples))
        return instances.getitems(ooblist)

    def numRight(self, oob, classifier):
        """
        Return a number of instances which are classified correctly.
        """
        right = 0
        for el in oob:
            if (el.getclass() == classifier(el)):
                right = right + 1
        return right
    
    def numRightMix(self, oob, classifier, attr):
        """
        Return a number of instances which are classified 
        correctly even if a feature is shuffled.
        """
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

    def importanceAcu(self, instances, trees, avimp):
        """Accumulate avimp by importances for a given number of trees."""
        n = len(instances)

        attrs = len(instances.domain.attributes)

        attrnum = {}
        for attr in range(len(instances.domain.attributes)):
           attrnum[instances.domain.attributes[attr].name] = attr            
   
        # build the forest
        classifiers = []  
        for i in range(trees):
            # draw bootstrap sample
            selection = []
            for j in range(n):
                selection.append(self.rand.randrange(n))
            data = instances.getitems(selection)
            
            # build the model from the bootstrap sample
            cla = self.learner(data)

            #prepare OOB data
            oob = self.getOOB(instances, selection, n)
            
            #right on unmixed
            right = self.numRight(oob, cla)
            
            presl = list(self.presentInTree(cla.tree, attrnum))
                      
            #randomize each feature in data and test
            #only those on which there was a split
            for attr in presl:
                #calculate number of right classifications
                #if the values of this features are permutated randomly
                rightimp = self.numRightMix(oob, cla, attr)                
                avimp[attr] += (float(right-rightimp))/len(oob)
        self.acu += trees  

    def presentInTree(self, node, attrnum):
        """Return features present in tree (features that split)."""
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
        self.attributes = attributes # number of features to consider
        if rand:
            self.rand = rand             # a random generator
        else:
            self.rand = random.Random()
            self.rand.seed(0)

    def __call__(self, gen, weightID, contingencies, apriori, candidates, clsfr):
        cand = [1]*self.attributes + [0]*(len(candidates) - self.attributes)
        self.rand.shuffle(cand)
        # instead with all features, we will invoke split constructor 
        # only for the subset of a features
        t = self.scons(gen, weightID, contingencies, apriori, cand, clsfr)
        return t