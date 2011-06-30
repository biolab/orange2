# Description: A simple implementation of wrapper feature subset selection
# Category:    modelling
# Uses:        imports-85
# Classes:     Domain, orngTest.crossValidation
# Referenced:  domain.htm

import orange, orngTest, orngStat, orngTree

def WrapperFSS(data, learner, verbose=0, folds=10):
  classVar = data.domain.classVar
  currentAtt = []
  freeAttributes = list(data.domain.attributes)

  newDomain = orange.Domain(currentAtt + [classVar])
  d = data.select(newDomain)
  results = orngTest.crossValidation([learner], d, folds=folds)
  maxStat = orngStat.CA(results)[0]
  if verbose>=2:
    print "start (%5.3f)" % maxStat

  while 1:
    stat = []
    for a in freeAttributes:
      newDomain = orange.Domain([a] + currentAtt + [classVar])
      d = data.select(newDomain)
      results = orngTest.crossValidation([learner], d, folds=folds)
      stat.append(orngStat.CA(results)[0])
      if verbose>=2:
        print "  %s gained %5.3f" % (a.name, orngStat.CA(results)[0])

    if (max(stat) > maxStat):
      oldMaxStat = maxStat
      maxStat = max(stat)
      bestVarIndx = stat.index(max(stat))
      if verbose:
        print "gain: %5.3f, attribute: %s" % (maxStat-oldMaxStat, freeAttributes[bestVarIndx].name)
      currentAtt = currentAtt + [freeAttributes[bestVarIndx]]
      del freeAttributes[bestVarIndx]
    else:
      if verbose:
        print "stopped (%5.3f)" % (max(stat) - maxStat)
      return orange.Domain(currentAtt + [classVar])
      break

data = orange.ExampleTable("voting")
learner = orngTree.TreeLearner(mForPruning=0.5)
#learner = orange.BayesLearner()

bestDomain = WrapperFSS(data, learner, verbose=1)
print bestDomain
