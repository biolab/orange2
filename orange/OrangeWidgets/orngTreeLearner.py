import orange

def TreeLearner(examples = None, weightID = 0, **argkw):
	tl = apply(TreeLearnerClass, (), argkw)
	if examples:
		tl = tl(examples, weightID)
	return tl

class TreeLearnerClass:
	def __init__(self, **kw):
		self.learner = None
		self.__dict__.update(kw)
		
	def __setattr__(self, name, value):
		if name in ["split", "binarization", "measure", "worstAcceptable", "minSubset",
					"stop", "maxMajority", "minExample"]:
			self.learner = None
		self.__dict__[name] = value

	def __call__(self, examples, weight=0):
		if not self.learner:
			self.learner = self.instance()
		tree = self.learner(examples, weight)
		if getattr(self, "sameMajorityPruning", 0):
			tree = orange.TreePruner_SameMajority(tree)
		if getattr(self, "mForPruning", 0):
			tree = orange.TreePruner_m(tree, m = self.mForPruning)
		return tree


	def instance(self):
		learner = orange.TreeLearner()

		if hasattr(self, "split"):
			learner.split = self.split

		else:
			learner.split = orange.TreeSplitConstructor_Combined()
			learner.split.continuousSplitConstructor = orange.TreeSplitConstructor_Threshold()
			if getattr(self, "binarization", 0):
				learner.split.discreteSplitConstructor = orange.TreeSplitConstructor_ExhaustiveBinary()
			else:
				learner.split.discreteSplitConstructor = orange.TreeSplitConstructor_Attribute()

			measures = {"infoGain": orange.MeasureAttribute_info,
					"gainRatio": orange.MeasureAttribute_gainRatio,
					"gini": orange.MeasureAttribute_gini,
					"relief": orange.MeasureAttribute_relief,
					"retis": orange.MeasureAttribute_retis
					}

			measure = getattr(self, "measure", None)
			if not measure:
				measure = orange.MeasureAttribute_gainRatio()
			elif type(measure) == str:
				measure = measures[measure]()

			learner.split.continuousSplitConstructor.measure = measure
			learner.split.discreteSplitConstructor.measure = measure

			wa = getattr(self, "worstAcceptable", 0)
			if wa:
				learner.split.continuousSplitConstructor.worstAcceptable = wa
				learner.split.discreteSplitConstructor.worstAcceptable = wa

			ms = getattr(self, "minSubset", 0)
			if ms:
				learner.split.continuousSplitConstructor.minSubset = ms
				learner.split.discreteSplitConstructor.minSubset = ms

		if hasattr(self, "stop"):
			learner.stop = self.stop
		else:
			learner.stop = orange.TreeStopCriteria_common()
			mm = getattr(self, "maxMajority", 0)
			if mm:
				learner.stop.maxMajority = self.maxMajority
			me = getattr(self, "minExamples", 0)
			if me:
				learner.stop.minExamples = self.minExamples

		for a in ["storeDistributions", "storeContingencies", "storeExamples", "storeNodeClassifier"]:
			if hasattr(self, a):
				setattr(learner, a, getattr(self, a))

		return learner
