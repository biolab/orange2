"""

.. index:: discretization

.. index:: 
   single: feature; discretization


Example-based automatic discretization is in essence similar to learning:
given a set of examples, discretization method proposes a list of suitable
intervals to cut the attribute's values into. For this reason, Orange
structures for discretization resemble its structures for learning. Objects
derived from ``orange.Discretization`` play a role of "learner" that, 
upon observing the examples, construct an ``orange.Discretizer`` whose role
is to convert continuous values into discrete according to the rule found by
``Discretization``.

Orange supports several methods of discretization; here's a
list of methods with belonging classes.

* Equi-distant discretization (:class:`EquiDistDiscretization`, 
  :class:`EquiDistDiscretizer`). The range of attribute's values is split
  into prescribed number equal-sized intervals.
* Quantile-based discretization (:class:`EquiNDiscretization`,
  :class:`IntervalDiscretizer`). The range is split into intervals
  containing equal number of examples.
* Entropy-based discretization (:class:`EntropyDiscretization`,
  :class:`IntervalDiscretizer`). Developed by Fayyad and Irani,
  this method balances between entropy in intervals and MDL of discretization.
* Bi-modal discretization (:class:`BiModalDiscretization`,
  :class:`BiModalDiscretizer`/:class:`IntervalDiscretizer`).
  Two cut-off points set to optimize the difference of the distribution in
  the middle interval and the distributions outside it.
* Fixed discretization (:class:`IntervalDiscretizer`). Discretization with 
  user-prescribed cut-off points.

.. _discretization.py: code/discretization.py

Instances of classes derived from :class:`Discretization`. It define a
single method: the call operator. The object can also be called through
constructor.

.. class:: Discretization

    .. method:: __call__(attribute, examples[, weightID])

        Given a continuous ``attribute`, ``examples`` and, optionally id of
        attribute with example weight, this function returns a discretized
        attribute. Argument ``attribute`` can be a descriptor, index or
        name of the attribute.

Here's an example. Part of `discretization.py`_:

.. literalinclude:: code/discretization.py
    :lines: 7-15

The discretized attribute ``sep_w`` is constructed with a call to
:class:`EntropyDiscretization` (instead of constructing it and calling
it afterwards, we passed the arguments for calling to the constructor, as
is often allowed in Orange). We then constructed a new 
:class:`Orange.data.Table` with attributes "sepal width" (the original 
continuous attribute), ``sep_w`` and the class attribute. Script output is::

    Entropy discretization, first 10 examples
    [3.5, '>3.30', 'Iris-setosa']
    [3.0, '(2.90, 3.30]', 'Iris-setosa']
    [3.2, '(2.90, 3.30]', 'Iris-setosa']
    [3.1, '(2.90, 3.30]', 'Iris-setosa']
    [3.6, '>3.30', 'Iris-setosa']
    [3.9, '>3.30', 'Iris-setosa']
    [3.4, '>3.30', 'Iris-setosa']
    [3.4, '>3.30', 'Iris-setosa']
    [2.9, '<=2.90', 'Iris-setosa']
    [3.1, '(2.90, 3.30]', 'Iris-setosa']

:class:`EntropyDiscretization` named the new attribute's values by the
interval range (it also named the attribute as "D_sepal width"). The new
attribute's values get computed automatically when they are needed.

As those that have read about :class:`Orange.data.variable.Variable` know,
the answer to 
"How this works?" is hidden in the field 
:obj:`~Orange.data.variable.Variable.get_value_from`.
This little dialog reveals the secret.

::

    >>> sep_w
    EnumVariable 'D_sepal width'
    >>> sep_w.get_value_from
    <ClassifierFromVar instance at 0x01BA7DC0>
    >>> sep_w.get_value_from.whichVar
    FloatVariable 'sepal width'
    >>> sep_w.get_value_from.transformer
    <IntervalDiscretizer instance at 0x01BA2100>
    >>> sep_w.get_value_from.transformer.points
    <2.90000009537, 3.29999995232>

So, the ``select`` statement in the above example converted all examples
from ``data`` to the new domain. Since the new domain includes the attribute
``sep_w`` that is not present in the original, ``sep_w``'s values are
computed on the fly. For each example in ``data``, ``sep_w.get_value_from`` 
is called to compute ``sep_w``'s value (if you ever need to call
``get_value_from``, you shouldn't call ``get_value_from`` directly but call
``compute_value`` instead). ``sep_w.get_value_from`` looks for value of
"sepal width" in the original example. The original, continuous sepal width
is passed to the ``transformer`` that determines the interval by its field
``points``. Transformer returns the discrete value which is in turn returned
by ``get_value_from`` and stored in the new example.

You don't need to understand this mechanism exactly. It's important to know
that there are two classes of objects for discretization. Those derived from
:obj:`Discretizer` (such as :obj:`IntervalDiscretizer` that we've seen above)
are used as transformers that translate continuous value into discrete.
Discretization algorithms are derived from :obj:`Discretization`. Their 
job is to construct a :obj:`Discretizer` and return a new variable
with the discretizer stored in ``get_value_from.transformer``.

Discretizers
============

Different discretizers support different methods for conversion of
continuous values into discrete. The most general is 
:class:`IntervalDiscretizer` that is also used by most discretization
methods. Two other discretizers, :class:`EquiDistDiscretizer` and 
:class:`ThresholdDiscretizer`> could easily be replaced by 
:class:`IntervalDiscretizer` but are used for speed and simplicity.
The fourth discretizer, :class:`BiModalDiscretizer` is specialized
for discretizations induced by :class:`BiModalDiscretization`.

.. class:: Discretizer

    All discretizers support a handy method for construction of a new
    attribute from an existing one.

    .. method:: construct_variable(attribute)

        Constructs a new attribute descriptor; the new attribute is discretized
        ``attribute``. The new attribute's name equal ``attribute.name`` 
        prefixed  by "D_", and its symbolic values are discretizer specific.
        The above example shows what comes out form :class:`IntervalDiscretizer`. 
        Discretization algorithms actually first construct a discretizer and
        then call its :class:`construct_variable` to construct an attribute
        descriptor.

.. class:: IntervalDiscretizer

    The most common discretizer. 

    .. attribute:: points

        Cut-off points. All values below or equal to the first point belong
        to the first interval, those between the first and the second
        (including those equal to the second) go to the second interval and
        so forth to the last interval which covers all values greater than
        the last element in ``points``. The number of intervals is thus 
        ``len(points)+1``.

Let us manually construct an interval discretizer with cut-off points at 3.0
and 5.0. We shall use the discretizer to construct a discretized sepal length 
(part of `discretization.py`_):

.. literalinclude:: code/discretization.py
    :lines: 22-26

That's all. First five examples of ``data2`` are now

::

    [5.1, '>5.00', 'Iris-setosa']
    [4.9, '(3.00, 5.00]', 'Iris-setosa']
    [4.7, '(3.00, 5.00]', 'Iris-setosa']
    [4.6, '(3.00, 5.00]', 'Iris-setosa']
    [5.0, '(3.00, 5.00]', 'Iris-setosa']

Can you use the same discretizer for more than one attribute? Yes, as long
as they have same cut-off points, of course. Simply call construct_var for each
continuous attribute (part of `discretization.py`_):

.. literalinclude:: code/discretization.py
    :lines: 30-34

Each attribute now has its own (FIXME) ClassifierFromVar in its 
``get_value_from``, but all use the same :class:`IntervalDiscretizer`, 
``idisc``. Changing an element of its ``points`` affect all attributes.

Do not change the length of :obj:`~IntervalDiscretizer.points` if the
discretizer is used by any attribute. The length of
:obj:`~IntervalDiscretizer.points` should always match the number of values
of the attribute, which is determined by the length of the attribute's field
``values``. Therefore, if ``attr`` is a discretized
attribute, than ``len(attr.values)`` must equal
``len(attr.get_value_from.transformer.points)+1``. It always
does, unless you deliberately change it. If the sizes don't match,
Orange will probably crash, and it will be entirely your fault.



.. class:: EquiDistDiscretizer

    More rigid than :obj:`IntervalDiscretizer`: 
    it uses intervals of fixed width.

    .. attribute:: first_cut
        
        The first cut-off point.
    
    .. attribute:: step

        Width of intervals.

    .. attribute:: number_of_intervals
        
        Number of intervals.

    .. attribute:: points (read-only)
        
        The cut-off points; this is not a real attribute although it behaves
        as one. Reading it constructs a list of cut-off points and returns it,
        but changing the list doesn't affect the discretizer - it's a separate
        list. This attribute is here only for to give the 
        :obj:`EquiDistDiscretizer` the same interface as that of 
        :obj:`IntervalDiscretizer`.

All values below :obj:`~EquiDistDiscretizer.first_cut` belong to the first
intervala (including possible values smaller than ``firstVal``. Otherwise,
value ``val``'s interval is ``floor((val-firstVal)/step)``. If this is turns
out to be greater or equal to :obj:`~EquiDistDiscretizer.number_of_intervals`, 
it is decreased to ``number_of_intervals-1``.

This discretizer is returned by :class:`EquiDistDiscretization`; you can
see an example in the corresponding section. You can also construct it 
manually and call its ``construct_variable``, just as shown for the
:obj:`IntervalDiscretizer`.


.. class:: ThresholdDiscretizer

    Threshold discretizer converts continuous values into binary by comparing
    them with a threshold. This discretizer is actually not used by any
    discretization method, but you can use it for manual discretization.
    Orange needs this discretizer for binarization of continuous attributes
    in decision trees.

    .. attribute:: threshold

        Threshold; values below or equal to the threshold belong to the first
        interval and those that are greater go to the second.

.. class:: BiModalDiscretizer

    This discretizer is the first discretizer that couldn't be replaced by
    :class:`IntervalDiscretizer`. It has two cut off points and values are
    discretized according to whether they belong to the middle region
    (which includes the lower but not the upper boundary) or not. The
    discretizer is returned by :class:`BiModalDiscretization` if its
    field :obj:`~BiModalDiscretization.split_in_two` is true (the default).

    .. attribute:: low
        
        Lower boudary of the interval (included in the interval).

    .. attribute:: high

        Upper boundary of the interval (not included in the interval).


Discretization Algorithms
=========================

.. class:: EquiDistDiscretization 

    Discretizes the attribute by cutting it into the prescribed number
    of intervals of equal width. The examples are needed to determine the 
    span of attribute values. The interval between the smallest and the
    largest is then cut into equal parts.

    .. attribute:: number_of_intervals

        Number of intervals into which the attribute is to be discretized. 
        Default value is 4.

For an example, we shall discretize all attributes of Iris dataset into 6
intervals. We shall construct an :class:`Orange.data.Table` with discretized
attributes and print description of the attributes (part
of `discretization.py`_):

.. literalinclude:: code/discretization.py
    :lines: 38-43

Script's answer is

::

    D_sepal length: <<4.90, [4.90, 5.50), [5.50, 6.10), [6.10, 6.70), [6.70, 7.30), >7.30>
    D_sepal width: <<2.40, [2.40, 2.80), [2.80, 3.20), [3.20, 3.60), [3.60, 4.00), >4.00>
    D_petal length: <<1.98, [1.98, 2.96), [2.96, 3.94), [3.94, 4.92), [4.92, 5.90), >5.90>
    D_petal width: <<0.50, [0.50, 0.90), [0.90, 1.30), [1.30, 1.70), [1.70, 2.10), >2.10>

Any more decent ways for a script to find the interval boundaries than 
by parsing the symbolic values? Sure, they are hidden in the discretizer,
which is, as usual, stored in ``attr.get_value_from.transformer``.

Compare the following with the values above.

::

    >>> for attr in newattrs:
    ...    print "%s: first interval at %5.3f, step %5.3f" % \
    ...    (attr.name, attr.get_value_from.transformer.first_cut, \
    ...    attr.get_value_from.transformer.step)
    D_sepal length: first interval at 4.900, step 0.600
    D_sepal width: first interval at 2.400, step 0.400
    D_petal length: first interval at 1.980, step 0.980
    D_petal width: first interval at 0.500, step 0.400

As all discretizers, :class:`EquiDistDiscretizer` also has the method 
``construct_variable`` (part of `discretization.py`_):

.. literalinclude:: code/discretization.py
    :lines: 69-73


.. class:: EquiNDiscretization

    Discretization with Intervals Containing (Approximately) Equal Number
    of Examples.

    Discretizes the attribute by cutting it into the prescribed number of
    intervals so that each of them contains equal number of examples. The
    examples are obviously needed for this discretization, too.

    .. attribute:: number_of_intervals

        Number of intervals into which the attribute is to be discretized.
        Default value is 4.

The use of this discretization is the same as the use of 
:class:`EquiDistDiscretization`. The resulting discretizer is 
:class:`IntervalDiscretizer`, hence it has ``points`` instead of ``first_cut``/
``step``/``number_of_intervals``.

.. class:: EntropyDiscretization

    Entropy-based Discretization (Fayyad-Irani).

    Fayyad-Irani's discretization method works without a predefined number of
    intervals. Instead, it recursively splits intervals at the cut-off point
    that minimizes the entropy, until the entropy decrease is smaller than the
    increase of MDL induced by the new point.

    An interesting thing about this discretization technique is that an
    attribute can be discretized into a single interval, if no suitable
    cut-off points are found. If this is the case, the attribute is rendered
    useless and can be removed. This discretization can therefore also serve
    for feature subset selection.

    .. attribute:: force_attribute

        Forces the algorithm to induce at least one cut-off point, even when
        its information gain is lower than MDL (default: false).

Part of `discretization.py`_:

.. literalinclude:: code/discretization.py
    :lines: 77-80

The output shows that all attributes are discretized onto three intervals::

    sepal length: <5.5, 6.09999990463>
    sepal width: <2.90000009537, 3.29999995232>
    petal length: <1.89999997616, 4.69999980927>
    petal width: <0.600000023842, 1.0000004768>

.. class:: BiModalDiscretization

    Bi-Modal Discretization

    Sets two cut-off points so that the class distribution of examples in
    between is as different from the overall distribution as possible. The
    difference is measure by chi-square statistics. All possible cut-off
    points are tried, thus the discretization runs in O(n^2).

    This discretization method is especially suitable for the attributes in
    which the middle region corresponds to normal and the outer regions to
    abnormal values of the attribute. Depending on the nature of the
    attribute, we can treat the lower and higher values separately, thus
    discretizing the attribute into three intervals, or together, in a
    binary attribute whose values correspond to normal and abnormal.

    .. attribute:: split_in_two
        
        Decides whether the resulting attribute should have three or two.
        If true (default), we have three intervals and the discretizer is
        of type :class:`BiModalDiscretizer`. If false the result is the 
        ordinary :class:`IntervalDiscretizer`.

Iris dataset has three-valued class attribute, classes are setosa, virginica
and versicolor. As the picture below shows, sepal lenghts of versicolors are
between lengths of setosas and virginicas (the picture itself is drawn using
LOESS probability estimation).

.. image:: files/bayes-iris.gif

If we merge classes setosa and virginica into one, we can observe whether
the bi-modal discretization would correctly recognize the interval in
which versicolors dominate.

.. literalinclude:: code/discretization.py
    :lines: 84-87

In this script, we have constructed a new class attribute which tells whether
an iris is versicolor or not. We have told how this attribute's value is
computed from the original class value with a simple lambda function.
Finally, we have constructed a new domain and converted the examples.
Now for discretization.

.. literalinclude:: code/discretization.py
    :lines: 97-100

The script prints out the middle intervals::

    sepal length: (5.400, 6.200]
    sepal width: (2.000, 2.900]
    petal length: (1.900, 4.700]
    petal width: (0.600, 1.600]

Judging by the graph, the cut-off points for "sepal length" make sense.

Additional functions
====================

Some functions and classes that can be used for
categorization of continuous features. Besides several general classes that
can help in this task, we also provide a function that may help in
entropy-based discretization (Fayyad & Irani), and a wrapper around classes for
categorization that can be used for learning.

.. automethod:: Orange.feature.discretization.entropyDiscretization_wrapper

.. autoclass:: Orange.feature.discretization.EntropyDiscretization_wrapper

.. autoclass:: Orange.feature.discretization.DiscretizedLearner_Class

.. rubric:: Example

FIXME. A chapter on `feature subset selection <../ofb/o_fss.htm>`_ in Orange
for Beginners tutorial shows the use of DiscretizedLearner. Other
discretization classes from core Orange are listed in chapter on
`categorization <../ofb/o_categorization.htm>`_ of the same tutorial.

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
        ThresholdDiscretizer, \
        EntropyDiscretization, \
        EquiDistDiscretization, \
        EquiNDiscretization, \
        BiModalDiscretization, \
        Discretization

######
# from orngDics.py
def entropyDiscretization_wrapper(table):
    """Take the classified table set (table) and categorize all continuous
    features using the entropy based discretization
    :obj:`EntropyDiscretization`.
    
    :param table: data to discretize.
    :type table: Orange.data.Table
    :rtype: :obj:`Orange.data.Table` includes all categorical and discretized\
    continuous features from the original data table.
    
    After categorization, features that were categorized to a single interval
    (to a constant value) are removed from table and prints their names.
    Returns a table that 

    """
    orange.setrandseed(0)
    tablen=orange.Preprocessor_discretize(table, method=EntropyDiscretization())
    
    attrlist=[]
    nrem=0
    for i in tablen.domain.attributes:
        if (len(i.values)>1):
            attrlist.append(i)
        else:
            nrem=nrem+1
    attrlist.append(tablen.domain.classVar)
    return tablen.select(attrlist)


class EntropyDiscretization_wrapper:
    """This is simple wrapper class around the function 
    :obj:`entropyDiscretization`. 
    
    :param data: data to discretize.
    :type data: Orange.data.Table
    
    Once invoked it would either create an object that can be passed a data
    set for discretization, or if invoked with the data set, would return a
    discretized data set::

        discretizer = Orange.feature.dicretization.EntropyDiscretization()
        disc_data = discretizer(table)
        another_disc_data = Orange.feature.dicretization.EntropyDiscretization(table)

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
      :obj:`Orange.feature.discretization.EntropyDiscretization`.
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
    def __init__(self, baseLearner, discretizer=EntropyDiscretization(), **kwds):
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
