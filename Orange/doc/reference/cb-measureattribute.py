# Description: Shows how to derive a Python class from orange.MeasureAttribute
# Category:    attribute quality, callbacks to Python
# Classes:     MeasureAttribute, TreeSplitConstructor
# Uses:        lenses
# Referenced:  callbacks.htm

import orange, orngTree, orngMisc
tab = orange.ExampleTable(r"lenses.tab")

class MeasureAttribute_Cardinality(orange.MeasureAttribute):
    def __init__(self, moreIsBetter = 1):
        self.moreIsBetter = moreIsBetter
        
    def __call__(self, a1, a2, a3):
        if type(a1) == int:
            attrNo, domainContingency, apriorClass = a1, a2, a3
            q = len(domainContingency[attrNo])
            name = domainContingency[attrNo].outerVariable.name
        else:
            contingency, classDistribution, apriorClass = a1, a2, a3
            q = len(contingency)
            name = contingency.outerVariable.name
#        print "Asked about %s; will answer %i" % (name, q)

        if self.moreIsBetter:
            return q
        else:
            return -q
    
def measure_cardinality(a1, a2, a3):
    if type(a1) == int:
        attrNo, domainContingency, apriorClass = a1, a2, a3
        q = len(domainContingency[attrNo])
        name = domainContingency[attrNo].outerVariable.name
    else:
        contingency, classDistribution, apriorClass = a1, a2, a3
        q = len(contingency)
        name = contingency.outerVariable.name
#    print "Asked about %s; will answer %i" % (name, q)
    return q


def measure_cardinality(a1, a2, a3):
  if type(a1) == int:
    return len(a2[a1])
  else:
    return len(a1)


print "LESS IS BETTER"
treeLearner = orange.TreeLearner()
treeLearner.split = orange.TreeSplitConstructor_Attribute(measure = MeasureAttribute_Cardinality(0))
tree = treeLearner(tab)
orngTree.printTxt(tree)


print "\n\nMORE IS BETTER"
treeLearner = orange.TreeLearner()
treeLearner.split = orange.TreeSplitConstructor_Attribute(measure = MeasureAttribute_Cardinality(1))
tree = treeLearner(tab)
orngTree.printTxt(tree)

print "\n\nFUNCTION (more is better)"
treeLearner = orange.TreeLearner()
treeLearner.split = orange.TreeSplitConstructor_Attribute(measure = measure_cardinality)
tree = treeLearner(tab)
orngTree.printTxt(tree)

print "\n\nRandom test based on the number of examples"
class MeasureAttribute_Generator(orange.MeasureAttribute):
     def __init__(self):
         self.needs = orange.MeasureAttribute.Generator
         
     def __call__(self, attr, generator, priorDist, weigthID):
         import random
         r = random.Random()
         r.seed(len(generator))
         return r.random()
     
treeLearner = orange.TreeLearner()
treeLearner.split = orange.TreeSplitConstructor_Attribute(measure = MeasureAttribute_Generator())
tree = treeLearner(tab)
orngTree.printTxt(tree)
 