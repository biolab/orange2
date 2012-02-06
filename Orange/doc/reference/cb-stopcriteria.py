# Description: Shows how to derive a Python class from orange.TreeStopCriteria
# Category:    classification, callbacks to Python
# Classes:     TreeStopCriteria
# Uses:        lenses
# Referenced:  callbacks.htm

import orange, orngMisc, orngTree

data = orange.ExampleTable("lenses")

from random import randint, seed
seed(0)

defStop = orange.TreeStopCriteria()
treeLearner = orange.TreeLearner()
treeLearner.stop = lambda exs, wID, cont: defStop(exs, wID, cont) or randint(1, 5)==1

print "\n\nTree build with stop criteria as a function"
tree = treeLearner(data)
orngTree.printTxt(tree)

class StoppingCriterion_random(orange.TreeStopCriteria):
  def __call__(self, gen, weightID, contingency):
    return orange.TreeStopCriteria.__call__(self, gen, weightID, contingency) \
           or randint(1, 5)==1

print "\n\nTree build with stop criteria as a class"
treeLearner.stop = StoppingCriterion_random()
tree = treeLearner(data)
orngTree.printTxt(tree)
