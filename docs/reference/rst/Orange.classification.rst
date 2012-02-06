.. automodule:: Orange.classification

###################################
Classification (``classification``)
###################################

All classifiers in Orange consist of two parts, a Learner and a Classifier. A
learner is constructed with all parameters that will be used for learning.
When a data table is passed to its __call__ method, a model is fitted to the
data and return in a form of a Classifier, which is then used for predicting
the dependent variable(s) of new instances.

.. class:: Learner()

    Base class for all orange learners.

    .. method:: __call__(instances)

        Fit a model and return it as an instance of :class:`Classifier`.

        This method is abstract and needs to be implemented on each learner.

.. class:: Classifier()

    Base class for all orange classifiers.

    .. attribute:: GetValue

        Return value of the target class when performing prediction.

    .. attribute:: GetProbabilities

        Return probability of each target class when performing prediction.

    .. attribute:: GetBoth

        Return a tuple of target class value and probabilities for each class.


    .. method:: __call__(instances, return_type)

        Classify a new instance using this model.

        This method is abstract and needs to be implemented on each classifier.

        :param instance: data instance to be classified.
        :type instance: :class:`~Orange.data.Instance`

        :param return_type: what needs to be predicted
        :type return_type: :obj:`GetBoth`,
                           :obj:`GetValue`,
                           :obj:`GetProbabilities`

        :rtype: :class:`~Orange.data.Value`,
              :class:`~Orange.statistics.distribution.Distribution` or a
              tuple with both

You can often program learners and classifiers as classes or functions written
entirely in Python and independent from Orange. Such classes can participate,
for instance, in the common evaluation functions like those available in
modules :obj:`Orange.evaluation.testing` and :obj:`Orange.evaluation.scoring`.

On the other hand, these classes can't be used as components for pure C++
classes. For instance, :obj:`Orange.classification.tree.TreeLearner`'s
attribute nodeLearner should contain a (wrapped) C++ object derived from
:obj:`Learner`, such as :obj:`Orange.classification.majority.MajorityLearner`
or :obj:`Orange.classification.bayes.NaiveLearner`. They cannot accommodate
Python's classes or even functions.

There's a workaround, though. You can subtype Orange classes :obj:`Learner` or
:obj:`Classifier` as if the two classes were defined in Python, but later use your
derived Python classes as if they were written in Orange's core. That is, you
can define your class in a Python script like this::

    class MyLearner(Orange.classifier.Learner): 
        def __call__(self, examples, weightID = 0): 
            <do something smart here>

Such a learner can then be used as any regular learner written in
Orange. You can, for instance, construct a tree learner and use your
learner to learn node classifier::

    treeLearner = Orange.classification.tree.TreeLearner()
    treeLearner.nodeLearner = MyLearner()

-----

Orange contains implementations of various classifiers that are described in
detail on separate pages.

.. toctree::
   :maxdepth: 2

   Orange.classification.bayes
   Orange.classification.knn
   Orange.classification.logreg
   Orange.classification.lookup
   Orange.classification.majority
   Orange.classification.rules
   Orange.classification.svm
   Orange.classification.tree
   Orange.classification.random   
