# Description: Shows how to derive a Python classifier from orange.Classifier
# Category:    classification, callbacks to Python
# Classes:     Classifier
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

tt =CartesianClassifier(tab.domain[0], tab.domain[1])

for ex in tab:
  print "%s ---> %s" % (ex, tt(ex))