Classification
==============

.. index:: classification
.. index:: supervised data mining

A substantial part of Orange is devoted to machine learning methods
for classification, or supervised data mining. These methods start
from the data that incorporates class-labeled instances, like
:download:`voting.tab <code/voting.tab>`::

   >>> data = orange.ExampleTable("voting.tab")
   >>> data[0]
   ['n', 'y', 'n', 'y', 'y', 'y', 'n', 'n', 'n', 'y', '?', 'y', 'y', 'y', 'n', 'y', 'republican']
   >>> data[0].getclass()
   <orange.Value 'party'='republican'>

Supervised data mining attempts to develop predictive models from such
data that, given the set of feature values, predict a corresponding
class.

.. index:: classifiers
.. index::
   single: classifiers; naive Bayesian

There are two types of objects important for classification: learners
and classifiers. Orange has a number of build-in learners. For
instance, ``orange.BayesLearner`` is a naive Bayesian learner. When
data is passed to a learner (e.g., ``orange.BayesLearner(data))``, it
returns a classifier. When data instance is presented to a classifier,
it returns a class, vector of class probabilities, or both.

A Simple Classifier
-------------------

Let us see how this works in practice. We will
construct a naive Bayesian classifier from voting data set, and
will use it to classify the first five instances from this data set
(:download:`classifier.py <code/classifier.py>`, uses :download:`voting.tab <code/voting.tab>`)::

   import orange
   data = orange.ExampleTable("voting")
   classifier = orange.BayesLearner(data)
   for i in range(5):
       c = classifier(data[i])
       print "original", data[i].getclass(), "classified as", c

The script loads the data, uses it to constructs a classifier using
naive Bayesian method, and then classifies first five instances of the
data set. Note that both original class and the class assigned by a
classifier is printed out.

The data set that we use includes votes for each of the U.S.  House of
Representatives Congressmen on the 16 key votes; a class is a
representative's party. There are 435 data instances - 267 democrats
and 168 republicans - in the data set (see UCI ML Repository and
voting-records data set for further description).  This is how our
classifier performs on the first five instances:

   1: republican (originally republican)
   2: republican (originally republican)
   3: republican (originally democrat)
   4: democrat (originally democrat)
   5: democrat (originally democrat)

Naive Bayes made a mistake at a third instance, but otherwise predicted
correctly.

Obtaining Class Probabilities
-----------------------------

To find out what is the probability that the classifier assigns
to, say, democrat class, we need to call the classifier with
additional parameter ``orange.GetProbabilities``. Also, note that the
democrats have a class index 1. We find this out with print
``data.domain.classVar.values`` (:download:`classifier2.py <code/classifier2.py>`, uses :download:`voting.tab <code/voting.tab>`)::

   import orange
   data = orange.ExampleTable("voting")
   classifier = orange.BayesLearner(data)
   print "Possible classes:", data.domain.classVar.values
   print "Probabilities for democrats:"
   for i in range(5):
       p = classifier(data[i], orange.GetProbabilities)
       print "%d: %5.3f (originally %s)" % (i+1, p[1], data[i].getclass())

The output of this script is::

   Possible classes: <republican, democrat>
   Probabilities for democrats:
   1: 0.000 (originally republican)
   2: 0.000 (originally republican)
   3: 0.005 (originally democrat)
   4: 0.998 (originally democrat)
   5: 0.957 (originally democrat)

The printout, for example, shows that with the third instance
naive Bayes has not only misclassified, but the classifier missed
quite substantially; it has assigned only a 0.005 probability to
the correct class.

.. note::
   Python list indexes start with 0.

.. note::
   The ordering of class values depend on occurence of classes in the
   input data set.

Classification tree
-------------------

.. index:: classifiers
.. index::
   single: classifiers; classification trees

Classification tree learner (yes, this is the same *decision tree*)
is a native Orange learner, but because it is a rather
complex object that is for its versatility composed of a number of
other objects (for attribute estimation, stopping criterion, etc.),
a wrapper (module) called ``orngTree`` was build around it to simplify
the use of classification trees and to assemble the learner with
some usual (default) components. Here is a script with it (:download:`tree.py <code/tree.py>`,
uses :download:`voting.tab <code/voting.tab>`)::

   import orange, orngTree
   data = orange.ExampleTable("voting")
   
   tree = orngTree.TreeLearner(data, sameMajorityPruning=1, mForPruning=2)
   print "Possible classes:", data.domain.classVar.values
   print "Probabilities for democrats:"
   for i in range(5):
       p = tree(data[i], orange.GetProbabilities)
       print "%d: %5.3f (originally %s)" % (i+1, p[1], data[i].getclass())
   
   orngTree.printTxt(tree)

.. note:: 
   The script for classification tree is almost the same as the one
   for naive Bayes (:download:`classifier2.py <code/classifier2.py>`), except that we have imported
   another module (``orngTree``) and used learner
   ``orngTree.TreeLearner`` to build a classifier called ``tree``.

.. note::
   For those of you that are at home with machine learning: the
   default parameters for tree learner assume that a single example is
   enough to have a leaf for it, gain ratio is used for measuring the
   quality of attributes that are considered for internal nodes of the
   tree, and after the tree is constructed the subtrees no pruning
   takes place.

The resulting tree with default parameters would be rather big, so we
have additionally requested that leaves that share common predecessor
(node) are pruned if they classify to the same class, and requested
that tree is post-pruned using m-error estimate pruning method with
parameter m set to 2.0. The output of our script is::

   Possible classes: <republican, democrat>
   Probabilities for democrats:
   1: 0.051 (originally republican)
   2: 0.027 (originally republican)
   3: 0.989 (originally democrat)
   4: 0.985 (originally democrat)
   5: 0.985 (originally democrat)

Notice that all of the instances are classified correctly. The last
line of the script prints out the tree that was used for
classification::

   physician-fee-freeze=n: democrat (98.52%)
   physician-fee-freeze=y
   |    synfuels-corporation-cutback=n: republican (97.25%)
   |    synfuels-corporation-cutback=y
   |    |    mx-missile=n
   |    |    |    el-salvador-aid=y
   |    |    |    |    adoption-of-the-budget-resolution=n: republican (85.33%)
   |    |    |    |    adoption-of-the-budget-resolution=y
   |    |    |    |    |    anti-satellite-test-ban=n: democrat (99.54%)
   |    |    |    |    |    anti-satellite-test-ban=y: republican (100.00%)
   |    |    |    el-salvador-aid=n
   |    |    |    |    handicapped-infants=n: republican (100.00%)
   |    |    |    |    handicapped-infants=y: democrat (99.77%)
   |    |    mx-missile=y
   |    |    |    religious-groups-in-schools=y: democrat (99.54%)
   |    |    |    religious-groups-in-schools=n
   |    |    |    |    immigration=y: republican (98.63%)
   |    |    |    |    immigration=n
   |    |    |    |    |    handicapped-infants=n: republican (98.63%)
   |    |    |    |    |    handicapped-infants=y: democrat (99.77%)

The printout includes the feature on which the tree branches in the
internal nodes. For leaves, it shows the the class label to which a
tree would make a classification. The probability of that class, as
estimated from the training data set, is also displayed.

If you are more of a *visual* type, you may like the graphical 
presentation of the tree better. This was achieved by printing out a
tree in so-called dot file (the line of the script required for this
is ``orngTree.printDot(tree, fileName='tree.dot',
internalNodeShape="ellipse", leafShape="box")``), which was then
compiled to PNG using program called `dot`_.

.. image:: files/tree.png
   :alt: A graphical presentation of a classification tree

.. _dot: http://graphviz.org/

Nearest neighbors and majority classifiers
------------------------------------------

.. index:: classifiers
.. index:: 
   single: classifiers; k nearest neighbours
.. index:: 
   single: classifiers; majority classifier

Let us here check on two other classifiers. Majority classifier always
classifies to the majority class of the training set, and predicts 
class probabilities that are equal to class distributions from the training
set. While being useless as such, it may often be good to compare this
simplest classifier to any other classifier you test &ndash; if your
other classifier is not significantly better than majority classifier,
than this may a reason to sit back and think.

The second classifier we are introducing here is based on k-nearest
neighbors algorithm, an instance-based method that finds k examples
from training set that are most similar to the instance that has to be
classified. From the set it obtains in this way, it estimates class
probabilities and uses the most frequent class for prediction.

The following script takes naive Bayes, classification tree (what we
have already learned), majority and k-nearest neighbors classifier
(new ones) and prints prediction for first 10 instances of voting data
set (:download:`handful.py <code/handful.py>`, uses :download:`voting.tab <code/voting.tab>`)::

   import orange, orngTree
   data = orange.ExampleTable("voting")
   
   # setting up the classifiers
   majority = orange.MajorityLearner(data)
   bayes = orange.BayesLearner(data)
   tree = orngTree.TreeLearner(data, sameMajorityPruning=1, mForPruning=2)
   knn = orange.kNNLearner(data, k=21)
   
   majority.name="Majority"; bayes.name="Naive Bayes";
   tree.name="Tree"; knn.name="kNN"
   
   classifiers = [majority, bayes, tree, knn]
   
   # print the head
   print "Possible classes:", data.domain.classVar.values
   print "Probability for republican:"
   print "Original Class",
   for l in classifiers:
       print "%-13s" % (l.name),
   print
   
   # classify first 10 instances and print probabilities
   for example in data[:10]:
       print "(%-10s)  " % (example.getclass()),
       for c in classifiers:
           p = apply(c, [example, orange.GetProbabilities])
           print "%5.3f        " % (p[0]),
       print

The code is somehow long, due to our effort to print the results
nicely. The first part of the code sets-up our four classifiers, and
gives them names. Classifiers are then put into the list denoted with
variable ``classifiers`` (this is nice since, if we would need to add
another classifier, we would just define it and put it in the list,
and for the rest of the code we would not worry about it any
more). The script then prints the header with the names of the
classifiers, and finally uses the classifiers to compute the
probabilities of classes. Note for a special function ``apply`` that
we have not met yet: it simply calls a function that is given as its
first argument, and passes it the arguments that are given in the
list. In our case, ``apply`` invokes our classifiers with a data
instance and request to compute probabilities. The output of our
script is::

   Possible classes: <republican, democrat>
   Probability for republican:
   Original Class Majority      Naive Bayes   Tree          kNN
   (republican)   0.386         1.000         0.949         1.000
   (republican)   0.386         1.000         0.973         1.000
   (democrat  )   0.386         0.995         0.011         0.138
   (democrat  )   0.386         0.002         0.015         0.468
   (democrat  )   0.386         0.043         0.015         0.035
   (democrat  )   0.386         0.228         0.015         0.442
   (democrat  )   0.386         1.000         0.973         0.977
   (republican)   0.386         1.000         0.973         1.000
   (republican)   0.386         1.000         0.973         1.000
   (democrat  )   0.386         0.000         0.015         0.000

.. note::
   The prediction of majority class classifier does not depend on the
   instance it classifies (of course!).

.. note:: 
   At this stage, it would be inappropriate to say anything conclusive
   on the predictive quality of the classifiers - for this, we will
   need to resort to statistical methods on comparison of
   classification models.
