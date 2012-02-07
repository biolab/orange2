import Orange
import Orange.core as orange

from Orange.core import \
    Discrete2Continuous, \
    Discretizer, \
    BiModalDiscretizer, \
    EquiDistDiscretizer as EqualWidthDiscretizer, \
    IntervalDiscretizer, \
    ThresholdDiscretizer,\
    EntropyDiscretization as Entropy, \
    EquiDistDiscretization as EqualWidth, \
    EquiNDiscretization as EqualFreq, \
    BiModalDiscretization as BiModal, \
    Discretization, \
    Preprocessor_discretize

def entropyDiscretization_wrapper(data):
    """Discretize all continuous features in class-labeled data set with the entropy-based discretization
    :obj:`Entropy`.
    
    :param data: data to discretize.
    :type data: Orange.data.Table
    :rtype: :obj:`Orange.data.Table` includes all categorical and discretized\
    continuous features from the original data table.
    
    After categorization, features that were categorized to a single interval
    (to a constant value) are removed from table and prints their names.
    Returns a table that 

    """
    orange.setrandseed(0)
    data_new = orange.Preprocessor_discretize(data, method=Entropy())
    
    attrlist = []
    nrem = 0
    for i in data_new.domain.attributes:
        if (len(i.values)>1):
            attrlist.append(i)
        else:
            nrem=nrem+1
    attrlist.append(data_new.domain.classVar)
    return data_new.select(attrlist)


class EntropyDiscretization_wrapper:
    """This is simple wrapper class around the function 
    :obj:`entropyDiscretization`. 
    
    :param data: data to discretize.
    :type data: Orange.data.Table
    
    Once invoked it would either create an object that can be passed a data
    set for discretization, or if invoked with the data set, would return a
    discretized data set::

        discretizer = Orange.feature.dicretization.Entropy()
        disc_data = discretizer(table)
        another_disc_data = Orange.feature.dicretization.Entropy(table)

    """
    def __call__(self, data):
        return entropyDiscretization(data)

def DiscretizedLearner(baseLearner, examples=None, weight=0, **kwds):
  learner = apply(DiscretizedLearner_Class, [baseLearner], kwds)
  if examples: return learner(examples, weight)
  else: return learner

class DiscretizedLearner_Class:
    """This class allows to set an learner object, such that before learning a
    data passed to a learner is discretized. In this way we can prepare an 
    object that lears without giving it the data, and, for instance, use it in
    some standard testing procedure that repeats learning/testing on several
    data samples. 

    :param baseLearner: learner to which give discretized data
    :type baseLearner: Orange.classification.Learner
    
    :param table: data whose continuous features need to be discretized
    :type table: Orange.data.Table
    
    :param discretizer: a discretizer that converts continuous values into
      discrete. Defaults to
      :obj:`Orange.feature.discretization.Entropy`.
    :type discretizer: Orange.feature.discretization.Discretization
    
    :param name: name to assign to learner 
    :type name: string

    An example on how such learner is set and used in ten-fold cross validation
    is given below::

        from Orange.feature import discretization
        bayes = Orange.classification.bayes.NaiveBayesLearner()
        disc = orange.Preprocessor_discretize(method=discretization.EquiNDiscretization(numberOfIntervals=10))
        dBayes = discretization.DiscretizedLearner(bayes, name='disc bayes')
        dbayes2 = discretization.DiscretizedLearner(bayes, name="EquiNBayes", discretizer=disc)
        results = Orange.evaluation.testing.CrossValidation([dBayes], table)
        classifier = discretization.DiscretizedLearner(bayes, examples=table)

    """
    def __init__(self, baseLearner, discretizer=Entropy(), **kwds):
        self.baseLearner = baseLearner
        if hasattr(baseLearner, "name"):
            self.name = baseLearner.name
        self.discretizer = discretizer
        self.__dict__.update(kwds)
    def __call__(self, data, weight=None):
        # filter the data and then learn
        from Orange.preprocess import Preprocessor_discretize
        ddata = Preprocessor_discretize(data, method=self.discretizer)
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
