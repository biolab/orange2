.. index::
   single: feature subset selection

Feature Subset Selection
========================

While the core Orange provides mechanisms to estimate relevance of
attributes that describe classified instances, a module called
:py:mod:`Orange.feature.selection` provides functions and wrappers that simplify feature
subset selection. For instance, the following code loads the data,
sets-up a filter that will use Relief measure to estimate the
relevance of attributes and remove attribute with relevance lower than
0.01, and in this way construct a new data set (:download:`fss6.py <code/fss6.py>`, uses
:download:`adult_sample.tab <code/adult_sample.tab>`)::

   import orange, orngFSS
   data = orange.ExampleTable("adult_sample")
   
   def report_relevance(data):
     m = orngFSS.attMeasure(data)
     for i in m:
       print "%5.3f %s" % (i[1], i[0])
   
   print "Before feature subset selection (%d attributes):" % \
     len(data.domain.attributes)
   report_relevance(data)
   data = orange.ExampleTable("adult_sample")
   
   marg = 0.01
   filter = orngFSS.FilterRelief(margin=marg)
   ndata = filter(data)
   print "\nAfter feature subset selection with margin %5.3f (%d attributes):" % \
     (marg, len(ndata.domain.attributes))
   
   report_relevance(ndata)

Notice that we have also defined a function ``report_relevance`` that
takes the data, computes the relevance of attributes (by calling
``orngFSS.attMeasure``) and then reports the computed
relevance. Notice that (by chance!) both ``orngFSS.attMeasure`` and
``orngFSS.FilterRelief`` use the same measure to estimate attributes,
so this code would actually get better if one would first set up an
object that would measure the attributes, and give it to both
``orngFSS.FilterRelief`` and ``report_relevance`` (we leave this for
you as an exercise). The output of the above script is::

   Before feature subset selection (14 attributes):
   0.183 relationship
   0.154 marital-status
   0.063 occupation
   0.031 education
   0.026 workclass
   0.020 age
   0.017 sex
   0.012 hours-per-week
   0.010 capital-loss
   0.009 education-num
   0.007 capital-gain
   0.006 race
   -0.002 fnlwgt
   -0.008 native-country
   
After feature subset selection with margin 0.010 (9 attributes)::

   0.108 marital-status
   0.101 relationship
   0.066 education
   0.021 education-num
   0.020 sex
   0.017 workclass
   0.017 occupation
   0.015 age
   0.010 hours-per-week

.. index::
   single: feature subset selection; wrapper

Out of 14 attributes, 5 were considered to be most relevant. We can
not check if this would help some classifier to achieve a better
performance. We will use 10-fold cross validation for comparison. To
do thinks rightfully, we need to do feature subset selection every
time we see new learning data, so we need to construct a learner that
has feature subset selection up-front, i.e., before it actually
learns. For a learner, we will use Naive Bayes with categorization (a
particular wrapper from orngDisc). The code is quite short since we
will also use a wrapper called FilteredLearner from orngFSS module
(part of :download:`fss7.py <code/fss7.py>`)::

   import orange, orngDisc, orngTest, orngStat, orngFSS
   
   data = orange.ExampleTable("crx")
   
   bayes = orange.BayesLearner()
   dBayes = orngDisc.DiscretizedLearner(bayes, name='disc bayes')
   fss = orngFSS.FilterAttsAboveThresh(threshold=0.05)
   fBayes = orngFSS.FilteredLearner(dBayes, filter=fss, name='bayes & fss')
   
   learners = [dBayes, fBayes]
   results = orngTest.crossValidation(learners, data, folds=10, storeClassifiers=1)

Below is the result. In terms of classification accuracy, feature
subset selection did not help. But, the rightmost column shows the
number of features used in each classifier (results are averaged
across ten trials of cross validation), and it is quite surprising
that on average only the use of about two features was sufficient::

   Learner         Accuracy  #Atts
   disc bayes      0.857     14.00
   bayes & fss     0.846      2.60

The code that computes this statistics, as well as determines which
are those features that were used, is shown below (from :download:`fss7.py <code/fss7.py>`)::

   # how many attributes did each classifier use?
   natt = [0.] * len(learners)
   for fold in range(results.numberOfIterations):
     for lrn in range(len(learners)):
       natt[lrn] += len(results.classifiers[fold][lrn].domain.attributes)
   for lrn in range(len(learners)):
     natt[lrn] = natt[lrn]/10.
   
   print "\nLearner         Accuracy  #Atts"
   for i in range(len(learners)):
     print "%-15s %5.3f     %5.2f" % (learners[i].name, orngEval.CA(results)[i], natt[i])
   
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

Following is the part of the output that shows the attribute
usage. Quite interesting, four attributes were used in constructed
classifiers, but only one (A9) in all ten classifiers constructed by
cross validation::

   Attribute usage (in how many folds attribute was used?):
   10 x A9
    2 x A10
    3 x A7
    6 x A6

