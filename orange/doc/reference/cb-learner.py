# Description: Shows how to derive a Python class form orange.Learner
# Category:    classification, learning, callbacks to Python
# Classes:     Learner, ContingencyAttrClass, orngMisc.BestOnTheFly
# Uses:        lenses
# Referenced:  callbacks.htm

import orange, orngTree, orngMisc
tab = orange.ExampleTable(r"lenses.tab")

class OneAttributeLearner(orange.Learner):
    def __init__(self, measure):
        self.measure = measure

    def __call__(self, gen, weightID=0):
        selectBest = orngMisc.BestOnTheFly()
        for attr in gen.domain.attributes:
            selectBest.candidate(self.measure(attr, gen, None, weightID))
        bestAttr = gen.domain.attributes[selectBest.winnerIndex()]
        classifier = orange.ClassifierByLookupTable(gen.domain.classVar, bestAttr)

        contingency = orange.ContingencyAttrClass(bestAttr, gen, weightID)
        for i in range(len(contingency)):
            classifier.lookupTable[i] = contingency[i].modus()
            classifier.distributions[i] = contingency[i]
        classifier.lookupTable[-1] = contingency.innerDistribution.modus()
        classifier.distributions[-1] = contingency.innerDistribution
        for d in classifier.distributions:
            d.normalize()
        
        return classifier


oal = OneAttributeLearner(orange.MeasureAttribute_gainRatio())
c = oal(tab)

print c.variable
print c.variable.values
print c.lookupTable
print c.distributions

for ex in tab:
    print "%s ---> %s" % (ex, c(ex))