.. index:: regression

Regression
==========

At the time of writing of this part of tutorial, there were
essentially two different learning methods for regression modelling:
regression trees and instance-based learner (k-nearest neighbors). In
this lesson, we will see that using regression is just like using
classifiers, and evaluation techniques are not much different either.

.. index::
   single: regression; regression trees

Few simple regressors
---------------------

Let us start with regression trees. Below is an example script that builds
the tree from :download:`housing.tab <code/housing.tab>` data set and prints
out the tree in textual form (:download:`regression1.py <code/regression1.py>`)::

   import orange, orngTree
   
   data = orange.ExampleTable("housing.tab")
   rt = orngTree.TreeLearner(data, measure="retis", mForPruning=2, minExamples=20)
   orngTree.printTxt(rt, leafStr="%V %I")
   
Notice special setting for attribute evaluation measure! Following is
the output of this script::
   
   RM<6.941: 19.9 [19.333-20.534]
   RM>=6.941
   |    RM<7.437
   |    |    CRIM>=7.393: 14.4 [10.172-18.628]
   |    |    CRIM<7.393
   |    |    |    DIS<1.886: 45.7 [37.124-54.176]
   |    |    |    DIS>=1.886: 32.7 [31.656-33.841]
   |    RM>=7.437
   |    |    TAX<534.500: 45.9 [44.295-47.498]
   |    |    TAX>=534.500: 21.9 [21.900-21.900]

.. index::
   single: regression; k nearest neighbours

Predicting continues classes is just like predicting crisp ones. In
this respect, the following script will be nothing new. It uses both
regression trees and k-nearest neighbors, and also uses a majority
learner which for regression simply returns an average value from
learning data set (:download:`regression2.py <code/regression2.py>`)::

   import orange, orngTree, orngTest, orngStat
   
   data = orange.ExampleTable("housing.tab")
   selection = orange.MakeRandomIndices2(data, 0.5)
   train_data = data.select(selection, 0)
   test_data = data.select(selection, 1)
   
   maj = orange.MajorityLearner(train_data)
   maj.name = "default"
   
   rt = orngTree.TreeLearner(train_data, measure="retis", mForPruning=2, minExamples=20)
   rt.name = "reg. tree"
   
   k = 5
   knn = orange.kNNLearner(train_data, k=k)
   knn.name = "k-NN (k=%i)" % k
   
   regressors = [maj, rt, knn]
   
   print "\n%10s " % "original",
   for r in regressors:
     print "%10s " % r.name,
   print
   
   for i in range(10):
     print "%10.1f " % test_data[i].getclass(),
     for r in regressors:
       print "%10.1f " % r(test_data[i]),
     print

The otput of this script is::

     original     default   reg. tree  k-NN (k=5)
         24.0        50.0        25.0        24.6
         21.6        50.0        25.0        22.0
         34.7        50.0        35.4        26.6
         28.7        50.0        25.0        36.2
         27.1        50.0        21.7        18.9
         15.0        50.0        21.7        18.9
         18.9        50.0        21.7        18.9
         18.2        50.0        21.7        21.0
         17.5        50.0        21.7        16.6
         20.2        50.0        21.7        23.1

.. index: mean squared error

Evaluation and scoring
----------------------

For our third and last example for regression, let us see how we can
use cross-validation testing and for a score function use
(:download:`regression3.py <code/regression3.py>`, uses `housing.tab <code/housing.tab>`)::

   import orange, orngTree, orngTest, orngStat
   
   data = orange.ExampleTable("housing.tab")
   
   maj = orange.MajorityLearner()
   maj.name = "default"
   rt = orngTree.TreeLearner(measure="retis", mForPruning=2, minExamples=20)
   rt.name = "regression tree"
   k = 5
   knn = orange.kNNLearner(k=k)
   knn.name = "k-NN (k=%i)" % k
   learners = [maj, rt, knn]
   
   data = orange.ExampleTable("housing.tab")
   results = orngTest.crossValidation(learners, data, folds=10)
   mse = orngStat.MSE(results)
   
   print "Learner        MSE"
   for i in range(len(learners)):
     print "%-15s %5.3f" % (learners[i].name, mse[i])

Again, compared to classification tasks, this is nothing new. The only
news in the above script is a mean squared error evaluation function
(``orngStat.MSE``). The scripts prints out the following report::

   Learner        MSE
   default         84.777
   regression tree 40.096
   k-NN (k=5)      17.532

Other scoring techniques are available to evaluate the success of
regression. Script below uses a range of them, plus features a nice
implementation where a list of scoring techniques is defined
independetly from the code that reports on the results (part of
:download:`regression4.py <code/regression4.py>`)::

   lr = orngRegression.LinearRegressionLearner(name="lr")
   rt = orngTree.TreeLearner(measure="retis", mForPruning=2,
                             minExamples=20, name="rt")
   maj = orange.MajorityLearner(name="maj")
   knn = orange.kNNLearner(k=10, name="knn")
   learners = [maj, lr, rt, knn]
   
   # evaluation and reporting of scores
   results = orngTest.learnAndTestOnTestData(learners, train, test)
   scores = [("MSE", orngStat.MSE),
             ("RMSE", orngStat.RMSE),
             ("MAE", orngStat.MAE),
             ("RSE", orngStat.RSE),
             ("RRSE", orngStat.RRSE),
             ("RAE", orngStat.RAE),
             ("R2", orngStat.R2)]
   
   print "Learner  " + "".join(["%-7s" % s[0] for s in scores])
   for i in range(len(learners)):
       print "%-8s " % learners[i].name + "".join(["%6.3f " % s[1](results)[i] for s in scores])

Here, we used a number of different scores, including:

* MSE - mean squared errror,
* RMSE - root mean squared error,
* MAE - mean absolute error,
* RSE - relative squared error,
* RRSE - root relative squared error,
* RAE - relative absolute error, and
* R2 - coefficient of determinatin, also referred to as R-squared.

For precise definition of these measures, see :py:mod:`Orange.statistics`. Running
the script above yields::

   Learner  MSE    RMSE   MAE    RSE    RRSE   RAE    R2
   maj      84.777  9.207  6.659  1.004  1.002  1.002 -0.004
   lr       23.729  4.871  3.413  0.281  0.530  0.513  0.719
   rt       40.096  6.332  4.569  0.475  0.689  0.687  0.525
   knn      17.244  4.153  2.670  0.204  0.452  0.402  0.796

