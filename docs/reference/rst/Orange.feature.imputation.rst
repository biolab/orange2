.. py:currentmodule:: Orange.feature.imputation

.. index:: imputation

.. index::
   single: feature; value imputation

***************************
Imputation (``imputation``)
***************************

Imputation replaces missing feature values with appropriate values, in this
case with minimal values:

.. literalinclude:: code/imputation-values.py
   :lines: 7-

The output of this code is::

    Example with missing values
    ['A', 1853, 'RR', ?, 2, 'N', 'DECK', 'WOOD', '?', 'S', 'WOOD']
    Imputed values:
    ['A', 1853, 'RR', 804, 2, 'N', 'DECK', 'WOOD', 'SHORT', 'S', 'WOOD']
    ['A', 1853, 'RR', 804, 2, 'N', 'DECK', 'WOOD', 'SHORT', 'S', 'WOOD']

Imputers
-----------------

:obj:`ImputerConstructor` is the abstract root in a hierarchy of classes
that accept training data and construct an instance of a class derived from
:obj:`Imputer`. When an :obj:`Imputer` is called with an
:obj:`Orange.data.Instance` it returns a new instance with the
missing values imputed (leaving the original instance intact). If imputer is
called with an :obj:`Orange.data.Table` it returns a new data table with
imputed instances.

.. class:: ImputerConstructor

    .. attribute:: impute_class

    Indicates whether to impute the class value. Defaults to True.

Simple imputation
=================

Simple imputers always impute the same value for a particular feature,
disregarding the values of other features. They all use the same class
:obj:`Imputer_defaults`.

.. class:: Imputer_defaults

    .. attribute::  defaults

    An instance :obj:`~Orange.data.Instance` with the default values to be
    imputed instead of missing value. Examples to be imputed must be from the
    same :obj:`~Orange.data.Domain` as :obj:`defaults`.

Instances of this class can be constructed by
:obj:`~Orange.feature.imputation.ImputerConstructor_minimal`,
:obj:`~Orange.feature.imputation.ImputerConstructor_maximal`,
:obj:`~Orange.feature.imputation.ImputerConstructor_average`.

For continuous features, they will impute the smallest, largest or the average
values encountered in the training examples. For discrete,
they will impute the lowest (the one with index 0, e. g. attr.values[0]),
the highest (attr.values[-1]), and the most common value encountered in the
data, respectively. If values of discrete features are ordered according to
their impact on class (for example, possible values for symptoms of some
disease can be ordered according to their seriousness),
the minimal and maximal imputers  will then represent optimistic and
pessimistic imputations.

User-define defaults can be given when constructing a
:obj:`~Orange.feature.imputation.Imputer_defaults`. Values that are left
unspecified do not get imputed. In the following example "LENGTH" is the
only attribute to get imputed with value 1234:

.. literalinclude:: code/imputation-complex.py
    :lines: 56-69

If :obj:`~Orange.feature.imputation.Imputer_defaults`'s constructor is given
an argument of type :obj:`~Orange.data.Domain` it constructs an empty instance
for :obj:`defaults`. If an instance is given, the reference to the
instance will be kept. To avoid problems associated with `Imputer_defaults
(data[0])`, it is better to provide a copy of the instance:
`Imputer_defaults(Orange.data.Instance(data[0]))`.

Random imputation
=================

.. class:: Imputer_Random

    Imputes random values. The corresponding constructor is
    :obj:`ImputerConstructor_Random`.

    .. attribute:: impute_class

    Tells whether to impute the class values or not. Defaults to True.

    .. attribute:: deterministic

    If true (defaults to False), random generator is initialized for each
    instance using the instance's hash value as a seed. This results in same
    instances being always imputed with the same (random) values.

Model-based imputation
======================

.. class:: ImputerConstructor_model

    Model-based imputers learn to predict the features's value from values of
    other features. :obj:`ImputerConstructor_model` are given two learning
    algorithms and they construct a classifier for each attribute. The
    constructed imputer :obj:`Imputer_model` stores a list of classifiers that
    are used for imputation.

    .. attribute:: learner_discrete, learner_continuous

    Learner for discrete and for continuous attributes. If any of them is
    missing, the attributes of the corresponding type will not get imputed.

    .. attribute:: use_class

    Tells whether the imputer can use the class attribute. Defaults to
    False. It is useful in more complex designs in which one imputer is used
    on learning instances, where it uses the class value,
    and a second imputer on testing instances, where class is not available.

.. class:: Imputer_model

    .. attribute:: models

    A list of classifiers, each corresponding to one attribute to be imputed.
    The :obj:`class_var`'s of the models should equal the instances'
    attributes. If an element is :obj:`None`, the corresponding attribute's
    values are not imputed.

.. rubric:: Examples

Examples are taken from :download:`imputation-complex.py
<code/imputation-complex.py>`. The following imputer predicts the missing
attribute values using classification and regression trees with the minimum
of 20 examples in a leaf.

.. literalinclude:: code/imputation-complex.py
    :lines: 74-76

A common setup, where different learning algorithms are used for discrete
and continuous features, is to use
:class:`~Orange.classification.bayes.NaiveLearner` for discrete and
:class:`~Orange.regression.mean.MeanLearner` (which
just remembers the average) for continuous attributes:

.. literalinclude:: code/imputation-complex.py
    :lines: 91-94

To construct a user-defined :class:`Imputer_model`:

.. literalinclude:: code/imputation-complex.py
    :lines: 108-112

A list of empty models is first initialized :obj:`Imputer_model.models`.
Continuous feature "LANES" is imputed with value 2 using
:obj:`DefaultClassifier`. A float must be given, because integer values are
interpreted as indexes of discrete features. Discrete feature "T-OR-D" is
imputed using :class:`~Orange.classification.ConstantClassifier` which is
given the index of value "THROUGH" as an argument.

Feature "LENGTH" is computed with a regression tree induced from "MATERIAL",
"SPAN" and "ERECTED" (feature "LENGTH" is used as class attribute here).
Domain is initialized by giving a list of feature names and domain as an
additional argument where Orange will look for features.

.. literalinclude:: code/imputation-complex.py
    :lines: 114-119

This is how the inferred tree should look like::

    <XMP class=code>SPAN=SHORT: 1158
    SPAN=LONG: 1907
    SPAN=MEDIUM
    |    ERECTED<1908.500: 1325
    |    ERECTED>=1908.500: 1528
    </XMP>

Wooden bridges and walkways are short, while the others are mostly
medium. This could be encoded in feature "SPAN" using
:class:`Orange.classifier.ClassifierByLookupTable`, which is faster than the
Python function used here:

.. literalinclude:: code/imputation-complex.py
    :lines: 121-128

If :obj:`compute_span` is written as a class it must behave like a
classifier: it accepts an example and returns a value. The second
argument tells what the caller expects the classifier to return - a value,
a distribution or both. Currently, :obj:`Imputer_model`,
always expects values and the argument can be ignored.

Missing values as special values
================================

Missing values sometimes have a special meaning. Cautious is needed when
using such values in decision models. When the decision not to measure
something (for example, performing a laboratory test on a patient) is based
on the expert's knowledge of the class value, such missing values clearly
should not be used in models.

.. class:: ImputerConstructor_asValue

    Constructs a new domain in which each discrete feature is replaced
    with a new feature that has one more value: "NA". The new feature
    computes its values on the fly from the old one,
    copying the normal values and replacing the unknowns with "NA".

    For continuous attributes, it constructs a two-valued discrete attribute
    with values "def" and "undef", telling whether the value is defined or
    not.  The features's name will equal the original's with "_def" appended.
    The original continuous feature will remain in the domain and its
    unknowns will be replaced by averages.

    :class:`ImputerConstructor_asValue` has no specific attributes.

    It constructs :class:`Imputer_asValue` that converts the example into
    the new domain.

.. class:: Imputer_asValue

    .. attribute:: domain

        The domain with the new feature constructed by
        :class:`ImputerConstructor_asValue`.

    .. attribute:: defaults

        Default values for continuous features.

The following code shows what the imputer actually does to the domain:

.. literalinclude:: code/imputation-complex.py
    :lines: 137-151

The script's output looks like this::

    [RIVER, ERECTED, PURPOSE, LENGTH, LANES, CLEAR-G, T-OR-D, MATERIAL, SPAN, REL-L, TYPE]

    [RIVER, ERECTED_def, ERECTED, PURPOSE, LENGTH_def, LENGTH, LANES_def, LANES, CLEAR-G, T-OR-D, MATERIAL, SPAN, REL-L, TYPE]

    RIVER: M -> M
    ERECTED: 1874 -> 1874 (def)
    PURPOSE: RR -> RR
    LENGTH: ? -> 1567 (undef)
    LANES: 2 -> 2 (def)
    CLEAR-G: ? -> NA
    T-OR-D: THROUGH -> THROUGH
    MATERIAL: IRON -> IRON
    SPAN: ? -> NA
    REL-L: ? -> NA
    TYPE: SIMPLE-T -> SIMPLE-T

The two examples have the same attribute, :samp:`imputed` having a few
additional ones. Comparing :samp:`original.domain[0] == imputed.domain[0]`
will result in False. While the names are same, they represent different
features. Writting, :samp:`imputed[i]`  would fail since :samp:`imputed` has
no attribute :samp:`i`, but it has an attribute with the same name.
Using :samp:`i.name` to index the attributes of :samp:`imputed` will work,
yet it is not fast. If a frequently used, it is better to compute the index
with :samp:`imputed.domain.index(i.name)`.

For continuous features, there is an additional feature with name prefix
"_def", which is accessible by :samp:`i.name+"_def"`. The value of the first
continuous feature "ERECTED" remains 1874, and the additional attribute
"ERECTED_def" has value "def". The undefined value  in "LENGTH" is replaced
by the average (1567) and the new attribute has value "undef". The
undefined discrete attribute  "CLEAR-G" (and all other undefined discrete
attributes) is assigned the value "NA".

Using imputers
--------------

Imputation is also used by learning algorithms and other methods that are not
capable of handling unknown values.

Imputer as a component
======================

Learners that cannot handle missing values should provide a slot
for imputer constructor. An example of such class is
:obj:`~Orange.classification.logreg.LogRegLearner` with attribute
:obj:`~Orange.classification.logreg.LogRegLearner.imputer_constructor`,
which imputes to average value by default. When given learning instances,
:obj:`~Orange.classification.logreg.LogRegLearner` will pass them to
:obj:`~Orange.classification.logreg.LogRegLearner.imputer_constructor` to get
an imputer and use it to impute the missing values in the learning data.
Imputed data is then used by the actual learning algorithm. When a
classifier :obj:`~Orange.classification.logreg.LogRegClassifier` is
constructed, the imputer is stored in its attribute
:obj:`~Orange.classification.logreg.LogRegClassifier.imputer`. During
classification the same imputer is used for imputation of missing values
in (testing) examples.

Details may vary from algorithm to algorithm, but this is how the imputation
is generally used. When writing user-defined learners,
it is recommended to use imputation according to the described procedure.

The choice of the imputer depends on the problem domain. In this example the
minimal value of each feature is imputed:

.. literalinclude:: code/imputation-logreg.py
   :lines: 7-

The output of this code is::

    Without imputation: 0.945
    With imputation: 0.954

.. note::

   Just one instance of
   :obj:`~Orange.classification.logreg.LogRegLearner` is constructed and then
   used twice in each fold. Once it is given the original instances as they
   are. It returns an instance of
   :obj:`~Orange.classification.logreg.LogRegLearner`. The second time it is
   called by :obj:`imra` and the
   :obj:`~Orange.classification.logreg.LogRegLearner` gets wrapped
   into :obj:`~Orange.feature.imputation.Classifier`. There is only one
   learner, which produces two different classifiers in each round of
   testing.

Wrappers for learning
=====================

In a learning/classification process, imputation is needed on two occasions.
Before learning, the imputer needs to process the training instances.
Afterwards, the imputer is called for each instance to be classified. For
example, in cross validation, imputation should be done on training folds
only. Imputing the missing values on all data and subsequently performing
cross-validation will give overly optimistic results.

Most of Orange's learning algorithms do not use imputers because they can
appropriately handle the missing values. Bayesian classifier, for instance,
simply skips the corresponding attributes in the formula, while
classification/regression trees have components for handling the missing
values in various ways. A wrapper is provided for learning algorithms that
require imputed data.

.. class:: ImputeLearner

    Wraps a learner and performs data imputation before learning.

    This is basically a learner, so the constructor will return either an
    instance of :obj:`ImputerLearner` or, if called with examples, an instance
    of some classifier.

    .. attribute:: base_learner

    A wrapped learner.

    .. attribute:: imputer_constructor

    An instance of a class derived from :obj:`ImputerConstructor` or a class
    with the same call operator.

    .. attribute:: dont_impute_classifier

    If set and a table is given, the classifier is not be
    wrapped into an imputer. This can be done if classifier can handle
    missing values.

    The learner is best illustrated by its code - here's its complete
    :obj:`__call__` method::

        def __call__(self, data, weight=0):
            trained_imputer = self.imputer_constructor(data, weight)
            imputed_data = trained_imputer(data, weight)
            base_classifier = self.base_learner(imputed_data, weight)
            if self.dont_impute_classifier:
                return base_classifier
            else:
                return ImputeClassifier(base_classifier, trained_imputer)

    During learning, :obj:`ImputeLearner` will first construct
    the imputer. It will then impute the data and call the
    given :obj:`baseLearner` to construct a classifier. For instance,
    :obj:`baseLearner` could be a learner for logistic regression and the
    result would be a logistic regression model. If the classifier can handle
    unknown values (that is, if :obj:`dont_impute_classifier`,
    it is returned as is, otherwise it is wrapped into
    :obj:`ImputeClassifier`, which holds the base classifier and
    the imputer used to impute the missing values in (testing) data.

.. class:: ImputeClassifier

    Objects of this class are returned by :obj:`ImputeLearner` when given data.

    .. attribute:: baseClassifier

    A wrapped classifier.

    .. attribute:: imputer

    An imputer for imputation of unknown values.

    .. method:: __call__

    This class's constructor accepts and stores two arguments,
    the classifier and the imputer. The call operator for classification
    looks like this::

        def __call__(self, ex, what=orange.GetValue):
            return self.base_classifier(self.imputer(ex), what)

    It imputes the missing values by calling the :obj:`imputer` and passes the
    class to the base classifier.

.. note::
   In this setup the imputer is trained on the training data. Even during
   cross validation, the imputer will be trained on the right data. In the
   classification phase, the imputer will be used to impute testing data.

.. rubric:: Code of ImputeLearner and ImputeClassifier

The learner is called with
:obj:`Orange.feature.imputation.ImputeLearner(base_learner=<someLearner>, imputer=<someImputerConstructor>)`.
When given examples, it trains the imputer, imputes the data,
induces a :obj:`base_classifier` by the
:obj:`base_learner` and constructs :obj:`ImputeClassifier` that stores the
:obj:`base_classifier` and the :obj:`imputer`. For classification, the missing
values are imputed and the classifier's prediction is returned.

This is a slightly simplified code, where details on how to handle
non-essential technical issues that are unrelated to imputation::

    class ImputeLearner(orange.Learner):
        def __new__(cls, examples = None, weightID = 0, **keyw):
            self = orange.Learner.__new__(cls, **keyw)
            self.__dict__.update(keyw)
            if examples:
                return self.__call__(examples, weightID)
            else:
                return self

        def __call__(self, data, weight=0):
            trained_imputer = self.imputer_constructor(data, weight)
            imputed_data = trained_imputer(data, weight)
            base_classifier = self.base_learner(imputed_data, weight)
            return ImputeClassifier(base_classifier, trained_imputer)

    class ImputeClassifier(orange.Classifier):
        def __init__(self, base_classifier, imputer):
            self.base_classifier = base_classifier
            self.imputer = imputer

        def __call__(self, ex, what=orange.GetValue):
            return self.base_classifier(self.imputer(ex), what)

Write your own imputer
----------------------

Imputation classes provide the Python-callback functionality. The simples
way to write custom imputation constructors or imputers is to write a Python
function that behaves like the built-in Orange classes. For imputers it is
enough to write a function that gets an instance as argument. Inputation for
data tables will then use that function.

Special imputation procedures or separate procedures for various attributes,
as demonstrated in the description of
:obj:`~Orange.feature.imputation.ImputerConstructor_model`,
are achieved by encoding it in a constructor that accepts a data table and
id of the weight meta-attribute, and returns the imputer. The benefit of
implementing an imputer constructor is that you can use is as a component
for learners (for example, in logistic regression) or wrappers, and that way
properly use the classifier in testing procedures.



..
    This was commented out:
    Examples
    --------

    Missing values sometimes have a special meaning, so they need to be replaced
    by a designated value. Sometimes we know what to replace the missing value
    with; for instance, in a medical problem, some laboratory tests might not be
    done when it is known what their results would be. In that case, we impute
    certain fixed value instead of the missing. In the most complex case, we assign
    values that are computed based on some model; we can, for instance, impute the
    average or majority value or even a value which is computed from values of
    other, known feature, using a classifier.

    In general, imputer itself needs to be trained. This is, of course, not needed
    when the imputer imputes certain fixed value. However, when it imputes the
    average or majority value, it needs to compute the statistics on the training
    examples, and use it afterwards for imputation of training and testing
    examples.

    While reading this document, bear in mind that imputation is a part of the
    learning process. If we fit the imputation model, for instance, by learning
    how to predict the feature's value from other features, or even if we
    simply compute the average or the minimal value for the feature and use it
    in imputation, this should only be done on learning data. Orange
    provides simple means for doing that.

    This page will first explain how to construct various imputers. Then follow
    the examples for `proper use of imputers <#using-imputers>`_. Finally, quite
    often you will want to use imputation with special requests, such as certain
    features' missing values getting replaced by constants and other by values
    computed using models induced from specified other features. For instance,
    in one of the studies we worked on, the patient's pulse rate needed to be
    estimated using regression trees that included the scope of the patient's
    injuries, sex and age, some attributes' values were replaced by the most
    pessimistic ones and others were computed with regression trees based on
    values of all features. If you are using learners that need the imputer as a
    component, you will need to `write your own imputer constructor
    <#write-your-own-imputer-constructor>`_. This is trivial and is explained at
    the end of this page.
