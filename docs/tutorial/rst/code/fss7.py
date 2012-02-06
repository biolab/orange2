# Author:      B Zupan
# Version:     1.0
# Description: Shows the use of feature subset selection and compares
#              plain naive Bayes (with discretization) and the same classifier but with
#              feature subset selection. On crx data set, both classifiers achieve similarly
#              accuracy but naive Bayes with feature subset selection uses substantially
#              less features. Wrappers FilteredLearner and DiscretizedLearner are used,
#              and example illustrates how to analyze classifiers used in ten-fold cross
#              validation (how many and which attributes were used?).
# Category:    preprocessing
# Uses:        crx.tab
# Referenced:  o_fss.htm

import orange, orngDisc, orngTest, orngStat, orngFSS

data = orange.ExampleTable("crx.tab")

bayes = orange.BayesLearner()
dBayes = orngDisc.DiscretizedLearner(bayes, name='disc bayes')
fss = orngFSS.FilterAttsAboveThresh(threshold=0.05)
fBayes = orngFSS.FilteredLearner(dBayes, filter=fss, name='bayes & fss')

learners = [dBayes, fBayes]
results = orngTest.crossValidation(learners, data, folds=10, storeClassifiers=1)

# how many attributes did each classifier use?

natt = [0.] * len(learners)
for fold in range(results.numberOfIterations):
  for lrn in range(len(learners)):
    natt[lrn] += len(results.classifiers[fold][lrn].domain.attributes)
for lrn in range(len(learners)):
  natt[lrn] = natt[lrn] / 10.

print "\nLearner         Accuracy  #Atts"
for i in range(len(learners)):
  print "%-15s %5.3f     %5.2f" % (learners[i].name, orngStat.CA(results)[i], natt[i])

# which attributes were used in filtered case?

print '\nAttribute usage (in how many folds attribute was used?):'
used = {}
for fold in range(results.numberOfIterations):
  for att in results.classifiers[fold][1].domain.attributes:
    a = att.name
    if a in used.keys(): used[a] += 1
    else: used[a] = 1
for a in used.keys():
  print '%2d x %s' % (used[a], a)
