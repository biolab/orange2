# Description: Shows how to derive a Python class from orange.TreeSplitConstructor
# Category:    classification, decision trees, callbacks to Python
# Classes:     TreeSplitConstructor, Classifier, SubsetsGenerator_constSize, orngMisc.BestOnTheFly
# Uses:        lenses
# Referenced:  callbacks.htm

import orange, orngTree, orngMisc
tab = orange.ExampleTable(r"lenses.tab")


class CartesianClassifier(orange.Classifier):
    def __init__(self, var1, var2):
        self.var1 = var1
        self.var2 = var2
        self.noValues2 = len(var2.values)
        self.classVar = orange.EnumVariable("%sx%s" % (var1.name, var2.name))
        self.classVar.values = ["%s-%s" % (v1, v2) for v1 in var1.values for v2 in var2.values]

    def __call__(self, ex, what = orange.Classifier.GetValue):
        val = ex[self.var1] * self.noValues2 + ex[self.var2]
        if what == orange.Classifier.GetValue:
            return orange.Value(self.classVar, val)
        probs = orange.DiscDistribution(self.classVar)
        probs[val] = 1.0
        if what == orange.Classifier.GetProbabilities:
            return probs
        else:
            return (orange.Value(self.classVar, val), probs)


class SplitConstructor_CartesianMeasure(orange.TreeSplitConstructor):
    def __init__(self, measure):
        self.measure = measure
        
    def __call__(self, gen, weightID, contingencies, apriori, candidates, nodeClassifier):
#        print "SplitConstructor_CartesianMeasure called"
        attributes = tab.domain.attributes
        selectBest = orngMisc.BestOnTheFly(orngMisc.compare2_firstBigger)
        for var1, var2 in orange.SubsetsGenerator_constSize(attributes, B=2):
            if candidates[attributes.index(var1)] and candidates[attributes.index(var2)]:
                cc = CartesianClassifier(var1, var2)
                cc.classVar.getValueFrom = cc
                meas = self.measure(cc.classVar, gen)
                selectBest.candidate((meas, cc))

        if not selectBest.best:
            return None

        bestMeas, bestSelector = selectBest.winner()
        bestSelector.classVar.getValueFrom = None
        return (bestSelector, bestSelector.classVar.values, None, bestMeas)


treeLearner = orange.TreeLearner()
treeLearner.split = SplitConstructor_CartesianMeasure(orange.MeasureAttribute_gainRatio)
tree = treeLearner(tab)
orngTree.printTxt(tree)