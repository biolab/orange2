""" 
.. index:: optimization

Wrappers for Tuning Parameters and Thresholds

Classes for two very useful purposes: tuning learning algorithm's parameters
using internal validation and tuning the threshold for classification into
positive class.

*****************
Tuning parameters
*****************

Two classes support tuning parameters.
:obj:`Orange.optimization.Tune1Parameter` for fitting a single parameter and
:obj:`Orange.optimization.TuneMParameters` fitting multiple parameters at once,
trying all possible combinations. When called with examples and, optionally, id
of meta attribute with weights, they find the optimal setting of arguments
using the cross validation. The classes can also be used as ordinary learning
algorithms - they are in fact derived from
:obj:`Orange.classification.Learner`.

Both classes have a common parent, :obj:`Orange.optimization.TuneParameters`,
and a few common attributes.

.. autoclass:: Orange.optimization.TuneParameters
   :members:

.. autoclass:: Orange.optimization.Tune1Parameter
   :members:
  
.. autoclass:: Orange.optimization.TuneMParameters
   :members: 
   
**************************
Setting Optimal Thresholds
**************************

Some models may perform well in terms of AUC which measures the ability to
distinguish between examples of two classes, but have low classifications
accuracies. The reason may be in the threshold: in binary problems, classifiers
usually classify into the more probable class, while sometimes, when class
distributions are highly skewed, a modified threshold would give better
accuracies. Here are two classes that can help.
  
.. autoclass:: Orange.optimization.ThresholdLearner
   :members: 
     
.. autoclass:: Orange.optimization.ThresholdClassifier
   :members: 
   
Examples
========

This is how you use the learner.

part of `optimization-thresholding1.py`_

.. literalinclude:: code/optimization-thresholding1.py

The output::

    W/out threshold adjustement: 0.633
    With adjusted thredhold: 0.659
    With threshold at 0.80: 0.449

shows that fitting threshold is good (well, although 2.5 percent increase in
the accuracy absolutely guarantees you a publication at ICML, the difference is
still unimportant), while setting it at 80% is a bad idea. Or is it?

part of `optimization-thresholding2.py`_

.. literalinclude:: code/optimization-thresholding2.py

The script first divides the data into training and testing examples. It trains
a naive Bayesian classifier and than wraps it into
:obj:`Orange.optimization.ThresholdClassifiers` with thresholds of .2, .5 and
.8. The three models are tested on the left-out examples, and we compute the
confusion matrices from the results. The printout::

    0.20: TP 60.000, TN 1.000
    0.50: TP 42.000, TN 24.000
    0.80: TP 2.000, TN 43.000

shows how the varying threshold changes the balance between the number of true
positives and negatives.

.. autoclass:: Orange.optimization.PreprocessedLearner
   :members: 
   
.. _optimization-thresholding1.py: code/optimization-thresholding1.py
.. _optimization-thresholding2.py: code/optimization-thresholding2.py

"""

import Orange.core
import Orange.classification
import Orange.evaluation.scoring
import Orange.evaluation.testing
import Orange.misc

class TuneParameters(Orange.classification.Learner):
    
    """.. attribute:: examples
    
        Data table with either discrete or continuous features
    
    .. attribute:: weightID
    
        The ID of the weight meta attribute
    
    .. attribute:: object
    
        The learning algorithm whose parameters are to be tuned. This can be,
        for instance, :obj:`Orange.classification.tree.TreeLearner`. You will
        usually use the wrapped learners from modules, not the built-in
        classifiers, such as :obj:`Orange.classification.tree.TreeLearner`
        directly, since the arguments to be fitted are easier to address in the
        wrapped versions. But in principle it doesn't matter.
    
    .. attribute:: evaluate
    
        The statistics to evaluate. The default is
        :obj:`Orange.evaluation.scoring.CA`, so the learner will be fit for the
        optimal classification accuracy. You can replace it with, for instance,
        :obj:`Orange.evaluation.scoring.AUC` to optimize the AUC. Statistics
        can return either a single value (classification accuracy), a list with
        a single value (this is what :obj:`Orange.evaluation.scoring.CA`
        actually does), or arbitrary objects which the compare function below
        must be able to compare.
    
    .. attribute:: folds
    
        The number of folds used in internal cross-validation. Default is 5.
    
    .. attribute:: compare
    
        The function used to compare the results. The function should accept
        two arguments (e.g. two classification accuracies, AUCs or whatever the
        result of evaluate is) and return a positive value if the first
        argument is better, 0 if they are equal and a negative value if the
        first is worse than the second. The default compare function is cmp.
        You don't need to change this if evaluate is such that higher values
        mean a better classifier.
    
    .. attribute:: returnWhat
    
        Decides what should be result of tuning. Possible values are:
    
        * TuneParameters.returnNone (or 0): tuning will return nothing,
        * TuneParameters.returnParameters (or 1): return the optimal value(s) of parameter(s),
        * TuneParameters.returnLearner (or 2): return the learner set to optimal parameters,
        * TuneParameters.returnClassifier (or 3): return a classifier trained with the optimal parameters on the entire data set. This is the default setting.
        
        Regardless of this, the learner (given as object) is left set to the
        optimal parameters.
    
    .. attribute:: verbose
    
        If 0 (default), the class doesn't print anything. If set to 1, it will
        print out the optimal value found, if set to 2, it will print out all
        tried values and the related
    
    If tuner returns the classifier, it behaves as a learning algorithm. As the
    examples below will demonstrate, it can be called, given the examples and
    the result is a "trained" classifier. It can, for instance, be used in
    cross-validation.

    Out of these attributes, the only necessary argument is object. The real
    tuning classes add two additional - the attributes that tell what
    parameter(s) to optimize and which values to use.
    
    """
    
    returnNone=0
    returnParameters=1
    returnLearner=2
    returnClassifier=3
    
    def __new__(cls, examples = None, weightID = 0, **argkw):
        self = Orange.classification.Learner.__new__(cls, **argkw)
        self.__dict__.update(argkw)
        if examples:
            return self.__call__(examples, weightID)
        else:
            return self

    def findobj(self, name):
        import string
        names=string.split(name, ".")
        lastobj=self.object
        for i in names[:-1]:
            lastobj=getattr(lastobj, i)
        return lastobj, names[-1]
        
class Tune1Parameter(TuneParameters):
    
    """Class :obj:`Orange.optimization.Tune1Parameter` tunes a single parameter.
    
    .. attribute:: parameter
    
        The name of the parameter (or a list of names, if the same parameter is
        stored at multiple places - see the examples) to be tuned.
    
    .. attribute:: values
    
        A list of parameter's values to be tried.
    
    To show how it works, we shall fit the minimal number of examples in a leaf
    for a tree classifier.
    
    part of `optimization-tuning1.py`_

    .. literalinclude:: code/optimization-tuning1.py
        :lines: 3-11

    Set up like this, when the tuner is called, set learner.minSubset to 1, 2,
    3, 4, 5, 10, 15 and 20, and measure the AUC in 5-fold cross validation. It
    will then reset the learner.minSubset to the optimal value found and, since
    we left returnWhat at the default (returnClassifier), construct and return
    the classifier from the entire data set. So, what we get is a classifier,
    but if we'd also like to know what the optimal value was, we can get it
    from learner.minSubset.

    Tuning is of course not limited to setting numeric parameters. You can, for
    instance, try to find the optimal criteria for assessing the quality of
    attributes by tuning parameter="measure", trying settings like
    values=[orange.MeasureAttribute_gainRatio(),
    orange.MeasureAttribute_gini()].
    
    Since the tuner returns a classifier and thus behaves like a learner, it
    can be used in a cross-validation. Let us see whether a tuning tree indeed
    enhances the AUC or not. We shall reuse the tuner from above, add another
    tree learner, and test them both.
    
    part of `optimization-tuning1.py`_

    .. literalinclude:: code/optimization-tuning1.py
        :lines: 13-18
    
    This will take some time: for each of 8 values for minSubset it will
    perform 5-fold cross validation inside a 10-fold cross validation -
    altogether 400 trees. Plus, it will learn the optimal tree afterwards for
    each fold. Add a tree without tuning, and you get 420 trees build.
    
    Well, not that long, and the results are good::
    
        Untuned tree: 0.930
        Tuned tree: 0.986
    
    .. _optimization-tuning1.py: code/optimization-tuning1.py
    
    """
    
    def __call__(self, table, weight=None, verbose=0):
        verbose = verbose or getattr(self, "verbose", 0)
        evaluate = getattr(self, "evaluate", Orange.evaluation.scoring.CA)
        folds = getattr(self, "folds", 5)
        compare = getattr(self, "compare", cmp)
        returnWhat = getattr(self, "returnWhat", 
                             Tune1Parameter.returnClassifier)

        if (type(self.parameter)==list) or (type(self.parameter)==tuple):
            to_set = [self.findobj(ld) for ld in self.parameter]
        else:
            to_set = [self.findobj(self.parameter)]

        cvind = Orange.core.MakeRandomIndicesCV(table, folds)
        findBest = Orange.misc.selection.BestOnTheFly(seed = table.checksum(), 
                                         callCompareOn1st = True)
        tableAndWeight = weight and (table, weight) or table
        for par in self.values:
            for i in to_set:
                setattr(i[0], i[1], par)
            res = evaluate(Orange.evaluation.testing.testWithIndices(
                                        [self.object], tableAndWeight, cvind))
            findBest.candidate((res, par))
            if verbose==2:
                print '*** optimization  %s: %s:' % (par, res)

        bestpar = findBest.winner()[1]
        for i in to_set:
            setattr(i[0], i[1], bestpar)

        if verbose:
            print "*** Optimal parameter: %s = %s" % (self.parameter, bestpar)

        if returnWhat==Tune1Parameter.returnNone:
            return None
        elif returnWhat==Tune1Parameter.returnParameters:
            return bestpar
        elif returnWhat==Tune1Parameter.returnLearner:
            return self.object
        else:
            classifier = self.object(table)
            classifier.setattr("fittedParameter", bestpar)
            return classifier

class TuneMParameters(TuneParameters):
    
    """The use of :obj:`Orange.optimization.TuneMParameters` differs from 
    :obj:`Orange.optimization.Tune1Parameter` only in specification of tuning
    parameters.
    
    .. attribute:: parameters
    
        A list of two-element tuples, each containing the name of a parameter
        and its possible values.
    
    For exercise we can try to tune both settings mentioned above, the minimal
    number of examples in leaves and the splitting criteria by setting the
    tuner as follows:
    
    `optimization-tuningm.py`_

    .. literalinclude:: code/optimization-tuningm.py
        
    Everything else stays like above, in examples for
    :obj:`Orange.optimization.Tune1Parameter`.
    
    .. _optimization-tuningm.py: code/optimization-tuningm.py
        
    """
    
    def __call__(self, table, weight=None, verbose=0):
        evaluate = getattr(self, "evaluate", Orange.evaluation.scoring.CA)
        folds = getattr(self, "folds", 5)
        compare = getattr(self, "compare", cmp)
        verbose = verbose or getattr(self, "verbose", 0)
        returnWhat=getattr(self, "returnWhat", Tune1Parameter.returnClassifier)
        progressCallback = getattr(self, "progressCallback", lambda i: None)
        
        to_set = []
        parnames = []
        for par in self.parameters:
            if (type(par[0])==list) or (type(par[0])==tuple):
                to_set.append([self.findobj(ld) for ld in par[0]])
                parnames.append(par[0])
            else:
                to_set.append([self.findobj(par[0])])
                parnames.append([par[0]])


        cvind = Orange.core.MakeRandomIndicesCV(table, folds)
        findBest = Orange.misc.selection.BestOnTheFly(seed = table.checksum(), 
                                         callCompareOn1st = True)
        tableAndWeight = weight and (table, weight) or table
        numOfTests = sum([len(x[1]) for x in self.parameters])
        milestones = set(range(0, numOfTests, max(numOfTests / 100, 1)))
        for itercount, valueindices in enumerate(Orange.misc.counters.LimitedCounter( \
                                        [len(x[1]) for x in self.parameters])):
            values = [self.parameters[i][1][x] for i,x \
                      in enumerate(valueindices)]
            for pi, value in enumerate(values):
                for i, par in enumerate(to_set[pi]):
                    setattr(par[0], par[1], value)
                    if verbose==2:
                        print "%s: %s" % (parnames[pi][i], value)
                        
            res = evaluate(Orange.evaluation.testing.testWithIndices(
                                        [self.object], tableAndWeight, cvind))
            if itercount in milestones:
                progressCallback(100.0 * itercount / numOfTests)
            
            findBest.candidate((res, values))
            if verbose==2:
                print "===> Result: %s\n" % res

        bestpar = findBest.winner()[1]
        if verbose:
            print "*** Optimal set of parameters: ",
        for pi, value in enumerate(bestpar):
            for i, par in enumerate(to_set[pi]):
                setattr(par[0], par[1], value)
                if verbose:
                    print "%s: %s" % (parnames[pi][i], value),
        if verbose:
            print

        if returnWhat==Tune1Parameter.returnNone:
            return None
        elif returnWhat==Tune1Parameter.returnParameters:
            return bestpar
        elif returnWhat==Tune1Parameter.returnLearner:
            return self.object
        else:
            classifier = self.object(table)
            classifier.fittedParameters = bestpar
            return classifier

class ThresholdLearner(Orange.classification.Learner):
    
    """:obj:`Orange.optimization.ThresholdLearner` is a class that wraps around 
    another learner. When given the data, it calls the wrapped learner to build
    a classifier, than it uses the classifier to predict the class
    probabilities on the training examples. Storing the probabilities, it
    computes the threshold that would give the optimal classification accuracy.
    Then it wraps the classifier and the threshold into an instance of
    :obj:`Orange.optimization.ThresholdClassifier`.

    Note that the learner doesn't perform internal cross-validation. Also, the
    learner doesn't work for multivalued classes. If you don't understand why,
    think harder. If you still don't, try to program it yourself, this should
    help. :)

    :obj:`Orange.optimization.ThresholdLearner` has the same interface as any
    learner: if the constructor is given examples, it returns a classifier,
    else it returns a learner. It has two attributes.
    
    .. attribute:: learner
    
        The wrapped learner, for example an instance of
        :obj:`Orange.classification.bayes.NaiveLearner`.
    
    .. attribute:: storeCurve
    
        If set, the resulting classifier will contain an attribute curve, with
        a list of tuples containing thresholds and classification accuracies at
        that threshold.
    
    """
    
    def __new__(cls, examples = None, weightID = 0, **kwds):
        self = Orange.classification.Learner.__new__(cls, **kwds)
        self.__dict__.update(kwds)
        if examples:
            return self.__call__(examples, weightID)
        else:
            return self

    def __call__(self, examples, weightID = 0):
        if not hasattr(self, "learner"):
            raise AttributeError("learner not set")
        
        classifier = self.learner(examples, weightID)
        threshold, optCA, curve = Orange.wrappers.ThresholdCA(classifier, 
                                                          examples, 
                                                          weightID)
        if getattr(self, "storeCurve", 0):
            return ThresholdClassifier(classifier, threshold, curve = curve)
        else:
            return ThresholdClassifier(classifier, threshold)

class ThresholdClassifier(Orange.classification.Classifier):
    
    """:obj:`Orange.optimization.ThresholdClassifier`, used by both 
    :obj:`Orange.optimization.ThredholdLearner` and
    :obj:`Orange.optimization.ThresholdLearner_fixed` is therefore another
    wrapper class, containing a classifier and a threshold. When it needs to
    classify an example, it calls the wrapped classifier to predict
    probabilities. The example will be classified into the second class only if
    the probability of that class is above the threshold.

    .. attribute:: classifier
    
    The wrapped classifier, normally the one related to the ThresholdLearner's
    learner, e.g. an instance of
    :obj:`Orange.classification.bayes.NaiveLearner`.
    
    .. attribute:: threshold
    
    The threshold for classification into the second class.
    
    The two attributes can be specified set as attributes or given to the
    constructor as ordinary arguments.
    
    """
    
    def __init__(self, classifier, threshold, **kwds):
        self.classifier = classifier
        self.threshold = threshold
        self.__dict__.update(kwds)

    def __call__(self, example, what = Orange.classification.Classifier.GetValue):
        probs = self.classifier(example, self.GetProbabilities)
        if what == self.GetProbabilities:
            return probs
        value = Orange.data.Value(self.classifier.classVar, probs[1] > \
                                  self.threshold)
        if what == Orange.classification.Classifier.GetValue:
            return value
        else:
            return (value, probs)

def ThresholdLearner_fixed(learner, threshold, 
                           examples=None, weightId=0, **kwds):
    
    """There's also a dumb variant of 
    :obj:`Orange.optimization.ThresholdLearner`, a class called
    :obj:`Orange.optimization.ThreshholdLearner_fixed`. Instead of finding the
    optimal threshold it uses a prescribed one. So, it has the following two
    attributes.
    
    .. attriute:: learner
    
    The wrapped learner, for example an instance of
    :obj:`Orange.classification.bayes.NaiveLearner`.
    
    .. attriute:: threshold
    
    Threshold to use in classification.
    
    What this guy does is therefore simple: to learn, it calls the learner and
    puts the resulting classifier together with the threshold into an instance
    of ThresholdClassifier.
    
    """
    
    lr = apply(ThresholdLearner_fixed_Class, (learner, threshold), kwds)
    if examples:
        return lr(examples, weightId)
    else:
        return lr
    
class ThresholdLearner_fixed(Orange.classification.Learner):
    def __new__(cls, examples = None, weightID = 0, **kwds):
        self = Orange.classification.Learner.__new__(cls, **kwds)
        self.__dict__.update(kwds)
        if examples:
            return self.__call__(examples, weightID)
        else:
            return self

    def __call__(self, examples, weightID = 0):
        if not hasattr(self, "learner"):
            raise AttributeError("learner not set")
        if not hasattr(self, "threshold"):
            raise AttributeError("threshold not set")
        if len(examples.domain.classVar.values)!=2:
            raise ValueError("ThresholdLearner handles binary classes only")
        
        return ThresholdClassifier(self.learner(examples, weightID), 
                                   self.threshold)

class PreprocessedLearner(object):
    def __new__(cls, preprocessor = None, learner = None):
        self = object.__new__(cls)
        if learner is not None:
            self.__init__(preprocessor)
            return self.wrapLearner(learner)
        else:
            return self
        
    def __init__(self, preprocessor = None, learner = None):
        if isinstance(preprocessor, list):
            self.preprocessors = preprocessor
        elif preprocessor is not None:
            self.preprocessors = [preprocessor]
        else:
            self.preprocessors = []
        #self.preprocessors = [Orange.core.Preprocessor_addClassNoise(proportion=0.8)]
        if learner:
            self.wrapLearner(learner)
        
    def processData(self, data, weightId = None):
        hadWeight = hasWeight = weightId is not None
        for preprocessor in self.preprocessors:
            if hasWeight:
                t = preprocessor(data, weightId)  
            else:
                t = preprocessor(data)
                
            if isinstance(t, tuple):
                data, weightId = t
                hasWeight = True
            else:
                data = t
        if hadWeight:
            return data, weightId
        else:
            return data

    def wrapLearner(self, learner):
        class WrappedLearner(learner.__class__):
            preprocessor = self
            wrappedLearner = learner
            name = getattr(learner, "name", "")
            def __call__(self, data, weightId=0, getData = False):
                t = self.preprocessor.processData(data, weightId or 0)
                processed, procW = t if isinstance(t, tuple) else (t, 0)
                classifier = self.wrappedLearner(processed, procW)
                if getData:
                    return classifier, processed
                else:
                    return classifier # super(WrappedLearner, self).__call__(processed, procW)
                
            def __reduce__(self):
                return PreprocessedLearner, (self.preprocessor.preprocessors, \
                                             self.wrappedLearner)
            
            def __getattr__(self, name):
                return getattr(learner, name)
            
        return WrappedLearner()
