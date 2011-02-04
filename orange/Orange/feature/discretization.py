"""

This module implements some functions and classes that can be used for
categorization of continuous features. Besides several general classes that
can help in this task, we also provide a function that may help in
entropy-based discretization (Fayyad & Irani), and a wrapper around classes for
categorization that can be used for learning.

.. method:: entropyDiscretization(table)

    Take the classified data set (table) and categorize all continuous
    features using the entropy based discretization 
    :obj:`EntropyDiscretization`. After categorization, 
    features that were categorized to a single interval (to a constant value) 
    are removed from table. Returns the data set that includes all categorical
    and discretized continuous features from the original data table.

.. class:: EntropyDiscretization

    This is simple wrapper class around the function
    :obj:`entropyDiscretization`. Once invoked it would either create an
    object that can be passed a data set for discretization, or if invoked
    with the data set, would return a discretized data set::

        discretizer = Orange.feature.dicretization.EntropyDiscretization()
        disc_data = discretizer(table)
        another_disc_data = Orange.feature.dicretization.EntropyDiscretization(table)

.. class:: DiscretizedLearner([baseLearner[, table[, discretizer[, name]]]])

    This class allows to set an learner object, such that before learning a
    data passed to a learner is discretized. In this way we can prepare an 
    object that lears without giving it the data, and, for instance, use it in
    some standard testing procedure that repeats learning/testing on several
    data samples. 

    :param baseLearner: learner to which give discretized data
    :type baseLearner: Orange.classification.Learner
    
    :param table: data whose continuous features need to be discretized
    :type table: Orange.data.Table
    
    :param discretizer: a discretizer that converts. Defaults to 
      :obj:`Orange.feature.discretization.EntropyDiscretization.
    :type discretizer: Orange.feature.discretization.Discretization
    
    :param name: name to assign to learner 
    :type name: string

    An example on how such learner is set and used in ten-fold cross validation
    is given below::

        bayes = Orange.classification.bayes.NaiveBayesLearner()
        disc = orange.Preprocessor_discretize(method=Orange.feature.discretization.EquiNDiscretization(numberOfIntervals=10))
        dBayes = Orange.feature.discretization.DiscretizedLearner(bayes, name='disc bayes')
        dbayes2 = Orange.feature.discretization.DiscretizedLearner(bayes, name="EquiNBayes", discretizer=disc)
        results = Orange.evaluation.testing.CrossValidation([dBayes], table)
        classifier = Orange.feature.discretization.DiscretizedLearner(bayes, examples=table)

========
Examples
========

A chapter on `feature subset selection <../ofb/o_fss.htm>`_ in Orange
for Beginners tutorial shows the use of DiscretizedLearner. Other
discretization classes from core Orange are listed in chapter on
`categorization <../ofb/o_categorization.htm>`_ of the same tutorial.


.. note::
    add from reference http://orange.biolab.si/doc/reference/discretization.htm

==========
References
==========

* UM Fayyad and KB Irani. Multi-interval discretization of continuous valued
  attributes for classification learning. In Proceedings of the 13th
  International Joint Conference on Artificial Intelligence, pages
  1022--1029, Chambery, France, 1993.

"""

import Orange.core as orange

from Orange.core import \
    Discrete2Continuous, \
    Discretizer, \
        BiModalDiscretizer, \
        EquiDistDiscretizer, \
        IntervalDiscretizer, \
        ThresholdDiscretizer

######
# from orngDics.py
def entropyDiscretization(data):
  """
  Discretizes continuous attributes using the entropy based discretization.
  It removes the attributes discretized to a single interval and prints their names.
  Arguments: data
  Returns:   table of examples with discretized atributes. Attributes that are
             categorized to a single value (constant) are removed.

  """
  orange.setrandseed(0)
  tablen=orange.Preprocessor_discretize(data, method=orange.EntropyDiscretization())

  attrlist=[]
  nrem=0
  for i in tablen.domain.attributes:
    if (len(i.values)>1):
      attrlist.append(i)
    else:
      nrem=nrem+1

  attrlist.append(tablen.domain.classVar)
  return tablen.select(attrlist)


class EntropyDiscretization:
  def __call__(self, data):
    return entropyDiscretization(data)


def DiscretizedLearner(baseLearner, examples=None, weight=0, **kwds):
  learner = apply(DiscretizedLearner_Class, [baseLearner], kwds)
  if examples: return learner(examples, weight)
  else: return learner

class DiscretizedLearner_Class:
  def __init__(self, baseLearner, discretizer=EntropyDiscretization(), **kwds):
    self.baseLearner = baseLearner
    if hasattr(baseLearner, "name"):
      self.name = baseLearner.name
    self.discretizer = discretizer
    self.__dict__.update(kwds)
  def __call__(self, data, weight=None):
    # filter the data and then learn
    ddata = self.discretizer(data)
    if weight<>None:
      model = self.baseLearner(ddata, weight)
    else:
      model = self.baseLearner(ddata)
    dcl = DiscretizedClassifier(classifier = model)
    if hasattr(model, "domain"):
      dcl.domain = model.domain
    if hasattr(model, "name"):
      dcl.name = model.name
    return dcl

class DiscretizedClassifier:
  def __init__(self, **kwds):
    self.__dict__.update(kwds)
  def __call__(self, example, resultType = orange.GetValue):
    return self.classifier(example, resultType)
