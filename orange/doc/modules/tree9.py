# Author:      B Zupan
# Version:     1.0
# Description: Takes three different domains, builds a classification
#              tree for each single attribute in the domain, and reports
#              on the size of the tree (quite useless)
# Category:    modelling
# Uses:        promoters,voting,adult_sample

import orange, orngTree

def t(att="handicapped-infants", learner=None):
  classVar = data.domain.classVar
  newDomain = orange.Domain([att] + [classVar], data.domain)
  d = data.select(newDomain)
  if not learner:
    learner = orngTree.TreeLearner()
  tree = learner(d)
  print '[%i] Domain: %s' % (tree.treesize(), newDomain)
  #orngTree.printTxt(tree)

datasets = ["../datasets/promoters", "voting", "../datasets/adult_sample"]

for f in datasets:
  data = orange.ExampleTable(f)
  print f, "\n---------------------"
  for a in data.domain.attributes:
    t(a)
