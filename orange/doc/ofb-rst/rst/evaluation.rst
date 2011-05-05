.. _accuracy.py: code/accuracy.py
.. _accuracy2.py: code/accuracy2.py
.. _accuracy3.py: code/accuracy3.py
.. _accuracy4.py: code/accuracy4.py
.. _accuracy5.py: code/accuracy5.py
.. _accuracy6.py: code/accuracy6.py
.. _accuracy7.py: code/accuracy7.py
.. _accuracy8.py: code/accuracy8.py
.. _orngStat.htm: ../modules/orngStat.htm
.. _orngTest.htm: ../modules/orngTest.htm
.. _roc.py: code/roc.py
.. _voting.tab: code/voting.tab


Testing and evaluating your classifiers
=======================================

.. index::
   single: classifiers; accuracy of

In this lesson you will learn how to estimate the accuracy of
classifiers. The simplest way to do this is to use Orange's
`orngTest.htm`_ and `orngStat.htm`_ modules. This is probably how you
will perform evaluation in your scripts, and thus we start with
examples that uses these two modules. You may as well perform testing
and scoring on your own, so we further provide several example scripts
to compute classification accuracy, measure it on a list of
classifiers, do cross-validation, leave-one-out and random
sampling. While all of this functionality is available in
`orngTest.htm`_ and `orngStat.htm`_ modules, these example scripts may
still be useful for those that want to learn more about Orange's
learner/classifier objects and the way to use them in combination with
data sampling.

.. index:: cross validation

Orange's classes for performance evaluation
-------------------------------------------

Below is a script that takes a list of learners (naive Bayesian
classifer and classification tree) and scores their predictive
performance on a single data set using ten-fold cross validation. The
script reports on four different scores: classification accuracy,
information score, Brier score and area under ROC curve
(`accuracy7.py`_, uses `voting.tab`_)::

   import orange, orngTest, orngStat, orngTree
   
   # set up the learners
   bayes = orange.BayesLearner()
   tree = orngTree.TreeLearner(mForPruning=2)
   bayes.name = "bayes"
   tree.name = "tree"
   learners = [bayes, tree]
   
   # compute accuracies on data
   data = orange.ExampleTable("voting")
   results = orngTest.crossValidation(learners, data, folds=10)
   
   # output the results
   print "Learner  CA     IS     Brier    AUC"
   for i in range(len(learners)):
       print "%-8s %5.3f  %5.3f  %5.3f  %5.3f" % (learners[i].name, \
           orngStat.CA(results)[i], orngStat.IS(results)[i],
           orngStat.BrierScore(results)[i], orngStat.AUC(results)[i])
   
The output of this script is::

   Learner  CA     IS     Brier    AUC
   bayes    0.901  0.758  0.176  0.976
   tree     0.961  0.845  0.075  0.956

The call to ``orngTest.CrossValidation`` does the hard work.  Function
``crossValidation`` returns the object stored in ``results``, which
essentially stores the probabilities and class values of the instances
that were used as test cases. Based on ``results``, the classification
accuracy, information score, Brier score and area under ROC curve
(AUC) for each of the learners are computed (function ``CA``, ``IS``
and ``AUC``).

Apart from statistics that we have mentioned above, `orngStat.htm`_
has build-in functions that can compute other performance metrics, and
`orngTest.htm`_ includes other testing schemas. If you need to test
your learners with standard statistics, these are probably all you
need. Compared to the script above, we below show the use of some
other statistics, with perhaps more modular code as above (part of
`accuracy8.py`_)::

   data = orange.ExampleTable("voting")
   res = orngTest.crossValidation(learners, data, folds=10)
   cm = orngStat.computeConfusionMatrices(res,
           classIndex=data.domain.classVar.values.index('democrat'))
   
   stat = (('CA', 'CA(res)'),
           ('Sens', 'sens(cm)'),
           ('Spec', 'spec(cm)'),
           ('AUC', 'AUC(res)'),
           ('IS', 'IS(res)'),
           ('Brier', 'BrierScore(res)'),
           ('F1', 'F1(cm)'),
           ('F2', 'Falpha(cm, alpha=2.0)'))
   
   scores = [eval("orngStat."+s[1]) for s in stat]
   print "Learner  " + "".join(["%-7s" % s[0] for s in stat])
   for (i, l) in enumerate(learners):
       print "%-8s " % l.name + "".join(["%5.3f  " % s[i] for s in scores])
   
For a number of scoring measures we needed to compute the confusion
matrix, for which we also needed to specify the target class
(democrats, in our case). This script has a similar output to the
previous one::

   Learner  CA     Sens   Spec   AUC    IS     Brier  F1     F2
   bayes    0.901  0.891  0.917  0.976  0.758  0.176  0.917  0.908
   tree     0.961  0.974  0.940  0.956  0.845  0.075  0.968  0.970

Do it on your own: a warm-up
----------------------------

Let us continue with a line of exploration of voting data set, and
build a naive Bayesian classifier from it, and compute the
classification accuracy on the same data set (`accuracy.py`_, uses
`voting.tab`_)::

   import orange
   data = orange.ExampleTable("voting")
   classifier = orange.BayesLearner(data)
   
   # compute classification accuracy
   correct = 0.0
   for ex in data:
       if classifier(ex) == ex.getclass():
           correct += 1
   print "Classification accuracy:", correct/len(data)

To compute classification accuracy, the script examines every
data item and checks how many times this has been classified
correctly. Running this script on shows that this is just above
90%.

.. warning::
   Training and testing on the same data set is not something we
   should do, as good performance scores may be simply due to
   overfitting. We use this type of testing here for code
   demonstration purposes only.

Let us extend the code with a function that is given a data set and a
set of classifiers (e.g., ``accuracy(test_data, classifiers)``) and
computes the classification accuracies for each of the classifier. By
this means, let us compare naive Bayes and classification trees
(`accuracy2.py`_, uses `voting.tab`_)::

   import orange, orngTree
   
   def accuracy(test_data, classifiers):
       correct = [0.0]*len(classifiers)
       for ex in test_data:
           for i in range(len(classifiers)):
               if classifiers[i](ex) == ex.getclass():
                   correct[i] += 1
       for i in range(len(correct)):
           correct[i] = correct[i] / len(test_data)
       return correct
   
   # set up the classifiers
   data = orange.ExampleTable("voting")
   bayes = orange.BayesLearner(data)
   bayes.name = "bayes"
   tree = orngTree.TreeLearner(data);
   tree.name = "tree"
   classifiers = [bayes, tree]
   
   # compute accuracies
   acc = accuracy(data, classifiers)
   print "Classification accuracies:"
   for i in range(len(classifiers)):
       print classifiers[i].name, acc[i]

This is the first time in out tutorial that we define a function.  You
may see that this is quite simple in Python; functions are introduced
with a keyword ``def``, followed by function's name and list of
arguments. Do not forget semicolon at the end of the definition
string. Other than that, there is nothing new in this code. A mild
exception to that is an expression ``classifiers[i](ex)``, but
intuition tells us that here the i-th classifier is called with a
function with example to classify as an argument. So, finally, which
method does better? Here is the output::

   Classification accuracies:
   bayes 0.903448275862
   tree 0.997701149425

It looks like a classification tree are much more accurate here.
But beware the overfitting (especially unpruned classification
trees are prone to that) and read on!

Training and test set
---------------------

In machine learning, one should not learn and test classifiers on the
same data set. For this reason, let us split our data in half, and use
first half of the data for training and the rest for testing. The
script is similar to the one above, with a part which is different
shown below (part of `accuracy3.py`_, uses `voting.tab`_)::

   # set up the classifiers
   data = orange.ExampleTable("voting")
   selection = orange.MakeRandomIndices2(data, 0.5)
   train_data = data.select(selection, 0)
   test_data = data.select(selection, 1)
   
   bayes = orange.BayesLearner(train_data)
   tree = orngTree.TreeLearner(train_data)

Orange's function ``RandomIndicesS2Gen`` takes the data and generates
a vector of length equal to the number of the data instances. Elements
of vectors are either 0 or 1, and the probability of the element being
0 is 0.5 (are whatever we specify in the argument of the
function). Then, for i-th instance of data, this may go either to the
training set (if selection[i]==0) or to test set (if
selection[i]==1). Notice that ``MakeRandomIndices2`` makes sure that
this split is stratified, e.g., the class distribution in training and
test set is approximately equal (you may use the attribute
``stratified=0`` if you do not like stratification).

The output of this testing is::

   Classification accuracies:
   bayes 0.93119266055
   tree 0.802752293578

Here, the accuracy naive Bayes is much higher. But warning: the result
is inconclusive, since it depends on only one random split of the
data.

70-30 random sampling
---------------------

Above, we have used the function ``accuracy(data, classifiers)`` that
took a data set and a set of classifiers and measured the
classification accuracy of classifiers on the data. Remember,
classifiers were models that have been already constructed (they have
*seen* the learning data already), so in fact the data in accuracy
served as a test data set. Now, let us write another function, that
will be given a set of learners and a data set, will repeatedly split
the data set to, say 70% and 30%, use the first part of the data (70%)
to learn the model and obtain a classifier, which, using accuracy
function developed above, will be tested on the remaining data (30%).

A learner in Orange is an object that encodes a specific machine
learning algorithm, and is ready to accept the data to construct and
return the predictive model. We have met quite a number of learners so
far (but we did not call them this way): ``orange.BayesLearner()``,
``orange.knnLearner()``, and others. If we use python to simply call a
learner, say with::

   ``learner = orange.BayesLearner()``

then ``learner`` becomes an instance of ``orange.BayesLearner`` and
is ready to get some data to return a classifier. For instance, in our
lessons so far we have used::

   ``classifier = orange.BayesLearner(data)``

and we could equally use::

   ``learner = orange.BayesLearner()``
   ``classifier = learner(data)``
   
So why complicating with learners? Well, in the task we are just
foreseeing, we will repeatedly do learning and testing. If we want to
build a reusable function that has in the input a set of machine
learning algorithm and on the output reports on their performance, we
can do this only through the use of learners (remember, classifiers
have already seen the data and cannot be re-learned).

Our script, without accuracy function, which is exactly like the
one we have defined in `accuracy2.py`_, is (part of `accuracy4.py`_)::

   def test_rnd_sampling(data, learners, p=0.7, n=10):
       acc = [0.0]*len(learners)
       for i in range(n):
           selection = orange.MakeRandomIndices2(data, p)
           train_data = data.select(selection, 0)
           test_data = data.select(selection, 1)
           classifiers = []
           for l in learners:
               classifiers.append(l(train_data))
           acc1 = accuracy(test_data, classifiers)
           print "%d: %s" % (i+1, acc1)
           for j in range(len(learners)):
               acc[j] += acc1[j]
       for j in range(len(learners)):
           acc[j] = acc[j]/n
       return acc
       
   # set up the learners
   bayes = orange.BayesLearner()
   tree = orngTree.TreeLearner()
   bayes.name = "bayes"
   tree.name = "tree"
   learners = [bayes, tree]
   
   # compute accuracies on data
   data = orange.ExampleTable("voting")
   acc = test_rnd_sampling(data, learners)
   print "Classification accuracies:"
   for i in range(len(learners)):
       print learners[i].name, acc[i]

Essential to the above script is a function test_rnd_sampling, which
takes the data and list of classifiers, and returns their accuracy
estimated through repetitive sampling. Additional (and optional)
parameter p tells what percentage of the data is used for
learning. There is another parameter n that specifies how many times
to repeat the learn-and-test procedure. Note that in the code, when
test_rnd_sampling was called, these two parameters were not specified
so that their default values were used (70% and 10, respectively). You
may try to change the code, and instead use test_rnd_sampling(data,
learners, n=100, p=0.5), or experiment in other ways. There is also a
print statement in test_rnd_sampling&nbsp; that reports on the
accuracies of the individual runs (just to see that the code really
works), which should probably be removed if you would not like to have
a long printout when testing with large n. Depending on the random
seed setup on your machine, the output of this script should be
something like::

   1: [0.9007633587786259, 0.79389312977099236]
   2: [0.9007633587786259, 0.79389312977099236]
   3: [0.95419847328244278, 0.92366412213740456]
   4: [0.87786259541984735, 0.86259541984732824]
   5: [0.86259541984732824, 0.80152671755725191]
   6: [0.87022900763358779, 0.80916030534351147]
   7: [0.87786259541984735, 0.82442748091603058]
   8: [0.92366412213740456, 0.93893129770992367]
   9: [0.89312977099236646, 0.82442748091603058]
   10: [0.92366412213740456, 0.86259541984732824]
   Classification accuracies:
   bayes 0.898473282443
   tree 0.843511450382

Ok, so we were rather lucky before with the tree results, and it looks
like naive Bayes does not do bad at all in comparison. But a warning
is in order: these are with trees with no punning. Try to use
something like ``tree = orngTree.TreeLearner(train_data,
mForPruning=2)`` in your script instead, and see if the result gets
any different (when we have tryed this, we get some improvement with
pruning)!

10-fold cross-validation
------------------------

The evaluation through k-fold cross validation method is probably the
most common in machine learning community. The data set is here split
into k equally sized subsets, and then in i-th iteration (i=1..k) i-th
subset is used for testing the classifier that has been build on all
other remaining subsets. Notice that in this method each instance has
been classified (for testing) exactly once. The number of subsets k is
usually set to 10. Orange has build-in procedure that splits develops
an array of length equal to the number of data instances, with each
element of the array being a number from 0 to k-1. This numbers are
assigned such that each resulting data subset has class distribution
that is similar to original subset (stratified k-fold
cross-validation).

The script for k-fold cross-validation is similar to the script for
repetitive random sampling above. We define a function called
``cross_validation`` and use it to compute the accuracies (part of
`accuracy5.py`_)::

   def cross_validation(data, learners, k=10):
       acc = [0.0]*len(learners)
       selection = orange.MakeRandomIndicesCV(data, folds=k)
       for test_fold in range(k):
           train_data = data.select(selection, test_fold, negate=1)
           test_data = data.select(selection, test_fold)
           classifiers = []
           for l in learners:
               classifiers.append(l(train_data))
           acc1 = accuracy(test_data, classifiers)
           print "%d: %s" % (test_fold+1, acc1)
           for j in range(len(learners)):
               acc[j] += acc1[j]
       for j in range(len(learners)):
           acc[j] = acc[j]/k
       return acc
   
   # ... some code skipped ...
   
   bayes = orange.BayesLearner()
   tree = orngTree.TreeLearner(mForPruning=2)
   
   # ... some code skipped ...
   
   # compute accuracies on data
   data = orange.ExampleTable("voting")
   acc = cross_validation(data, learners, k=10)
   print "Classification accuracies:"
   for i in range(len(learners)):
       print learners[i].name, acc[i]

Notice that to select the instances, we have again used
``data.select``. To obtain train data, we have instructed Orange to
use all instances that have a value different from ``test_fold``, an
integer that stores the current index of the fold to be used for
testing. Also notice that this time we have included pruning for
trees.

Running the 10-fold cross validation on our data set results in
similar numbers as produced by random sampling (when pruning was
used). For those of you curious if this is really so, run the script
yourself.

Leave-one-out
-------------

This evaluation procedure is often performed when data sets are small
(no really the case for the data we are using in our example). If each
cycle, a single instance is used for testing, while the classifier is
build on all other instances. One can define leave-one-out test
through a single Python function (part of `accuracy6.py`_)::

   def leave_one_out(data, learners):
       print 'leave-one-out: %d of %d' % (i, len(data))
       acc = [0.0]*len(learners)
       selection = [1] * len(data)
       last = 0
       for i in range(len(data)):
           selection[last] = 1
           selection[i] = 0
           train_data = data.select(selection, 1)
           for j in range(len(learners)):
               classifier = learners[j](train_data)
               if classifier(data[i]) == data[i].getclass():
                   acc[j] += 1
           last = i
   
       for j in range(len(learners)):
           acc[j] = acc[j]/len(data)
       return acc

What is not shown in the code above but contained in the script, is
that we have introduced some pre-pruning with trees and used ``tree =
orngTree.TreeLearner(minExamples=10, mForPruning=2)``. This was just
to decrease the time one needs to wait for results of the testing (on
our moderately fast machines, it takes about half-second for each
iteration).

Again, Python's list variable selection is used to filter out the data
for learning: this time all its elements but i-th are equal
to 1. There is no need to separately create test set, since it
contains only one (i-th) item, which is referred to directly as
``data[i]``. Everything else (except for the call to leave_one_out, which
this time requires no extra parameters) is the same as in the scripts
defined for random sampling and cross-validation.  Interestingly, the
accuracies obtained on voting data set are similar as well::

   Classification accuracies:
   bayes 0.901149425287
   tree 0.96091954023

Area under roc
--------------

Going back to the data set we use in this lesson (`voting.tab`_), let
us say that at the end of 1984 we met on a corridor two members of
congress. Somebody tells us that they are for a different party. We
now use the classifier we have just developed on our data to compute
the probability that each of them is republican. What is the chance
that the one we have assigned a higher probability is the one that is
republican indeed?

This type of statistics is much used in medicine and is called area
under ROC curve (see, for instance, JR Beck &amp; EK Schultz: The use
of ROC curves in test performance evaluation. Archives of Pathology
and Laboratory Medicine 110:13-20, 1986 and Hanley &amp; McNeil: The
meaning and use of the area under receiver operating characteristic
curve. Radiology, 143:29--36, 1982). It is a discrimination measure
that ranges from 0.5 (random guessing) to 1.0 (a clear margin exists
in probability that divides the two classes). Just to give another
example for yet another statistics that can be assessed in Orange, we
here present a simple (but not optimized and rather inefficient)
implementation of this measure.

We will use a script similar to `accuracy5.py`_ (k-fold cross
validation) and will replace the accuracy() function with a function
that computes area under ROC for a given data set and set of
classifiers. The algorithm will investigate all pairs of data
items. Those pairs where the outcome was originally different (e.g.,
one item represented a republican, the other one democrat) will be
termed valid pairs and will be checked. Given a valid pair, if the
higher probability for republican was indeed assigned to the item that
was republican also originally, this pair will be termed a correct
pair. Area under ROC is then the proportion of correct pairs in the
set of valid pairs of instances. In case of ties (both instances were
assigned the same probability of representing a republican), this
would be counted as 0.5 instead of 1. The code for function that
computes the area under ROC using this method is coded in Python as
(part of `roc.py`_)::

   def aroc(data, classifiers):
       ar = []
       for c in classifiers:
           p = []
           for d in data:
               p.append(c(d, orange.GetProbabilities)[0])
           correct = 0.0; valid = 0.0
           for i in range(len(data)-1):
               for j in range(i+1,len(data)):
                   if data[i].getclass() <> data[j].getclass():
                       valid += 1
                       if p[i] == p[j]:
                           correct += 0.5
                       elif data[i].getclass() == 0:
                           if p[i] > p[j]:
                               correct += 1.0
                       else:
                           if p[j] > p[i]:
                               correct += 1.0
           ar.append(correct / valid)
       return ar
   
Notice that the array p of length equal to the data set contains the
probabilities of the item being classified as republican. We have to
admit that although on the voting data set and under 10-fold
cross-validation computing area under ROC is rather fast (below 3s),
there exist a better algorithm with complexity O(n log n) instead of
O(n^2). Anyway, running `roc.py`_ shows that naive Bayes is better in
terms of discrimination using area under ROC::

   Area under ROC:
   bayes 0.970308048433
   tree 0.954274027987
   majority 0.5

.. note::
   Just for a check a majority classifier was also included in the
   test case this time. As expected, its area under ROC is minimal and
   equal to 0.5.
