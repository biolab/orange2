import orange, math, orngTest, orngStat, random, orngMisc
from operator import add
inf = 100000

def BoostedLearner(learner=None, t=10, name='boosted classifier', examples=None):
	learner = BoostedLearnerClass(learner, t, name)
	if examples:
		return learner(examples)
	else:
		return learner

class BoostedLearnerClass:
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

class BoostedClassifier:
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
		for i in range(len(votes)):
			votes[i] = votes[i]/len(self.classifiers)
		if resultType == orange.GetProbabilities:
			return votes
		else:
			return (value, votes)
		


#
# BAGGING
#

def BaggedLearner(learner=None, t=10, name='bagged classifier', examples=None):
	learner = BaggedLearnerClass(learner, t, name)
	if examples:
		return learner(examples)
	else:
		return learner

class BaggedLearnerClass:
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

class BaggedClassifier:
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

