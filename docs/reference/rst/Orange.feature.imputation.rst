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
=================

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

    An instance :obj:`Orange.data.Instance` with the default values to be
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

User-define defaults can be given when constructing a :obj:`~Orange.feature
.imputation.Imputer_defaults`. Values that are left unspecified do not get
imputed. In the following example "LENGTH" is the
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
imputed using :class:`Orange.classification.ConstantClassifier` which is
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
==============

Imputation must run on training data only. Imputing the missing values
and subsequently using the data in cross-validation will give overly
optimistic results.

Learners with imputer as a component
------------------------------------

Learners that cannot handle missing values provide a slot for the imputer
component. An example of such a class is
:obj:`~Orange.classification.logreg.LogRegLearner` with an attribute called
:obj:`~Orange.classification.logreg.LogRegLearner.imputer_constructor`.

When given learning instances,
:obj:`~Orange.classification.logreg.LogRegLearner` will pass them to
:obj:`~Orange.classification.logreg.LogRegLearner.imputer_constructor` to get
an imputer and used it to impute the missing values in the learning data.
Imputed data is then used by the actual learning algorithm. Also, when a
classifier :obj:`Orange.classification.logreg.LogRegClassifier` is constructed,
the imputer is stored in its attribute
:obj:`Orange.classification.logreg.LogRegClassifier.imputer`. At
classification, the same imputer is used for imputation of missing values
in (testing) examples.

Details may vary from algorithm to algorithm, but this is how the imputation
is generally used. When writing user-defined learners,
it is recommended to use imputation according to the described procedure.

Wrapper for learning algorithms
===============================

Imputation is also used by learning algorithms and other methods that are not
capable of handling unknown values. It imputes missing values,
calls the learner and, if imputation is also needed by the classifier,
it wraps the classifier that imputes missing values in instances to classify.

.. literalinclude:: code/imputation-logreg.py
   :lines: 7-

The output of this code is::

    Without imputation: 0.945
    With imputation: 0.954

Even so, the module is somewhat redundant, as all learners that cannot handle
missing values should, in principle, provide the slots for imputer constructor.
For instance, :obj:`Orange.classification.logreg.LogRegLearner` has an
attribute
:obj:`Orange.classification.logreg.LogRegLearner.imputer_constructor`,
and even if you don't set it, it will do some imputation by default.

.. class:: ImputeLearner

    Wraps a learner and performs data discretization before learning.

    Most of Orange's learning algorithms do not use imputers because they can
    appropriately handle the missing values. Bayesian classifier, for instance,
    simply skips the corresponding attributes in the formula, while
    classification/regression trees have components for handling the missing
    values in various ways.

    If for any reason you want to use these algorithms to run on imputed data,
    you can use this wrapper. The class description is a matter of a separate
    page, but we shall show its code here as another demonstration of how to
    use the imputers - logistic regression is implemented essentially the same
    as the below classes.

    This is basically a learner, so the constructor will return either an
    instance of :obj:`ImputerLearner` or, if called with examples, an instance
    of some classifier. There are a few attributes that need to be set, though.

    .. attribute:: base_learner

    A wrapped learner.

    .. attribute:: imputer_constructor

    An instance of a class derived from :obj:`ImputerConstructor` (or a class
    with the same call operator).

    .. attribute:: dont_impute_classifier

    If given and set (this attribute is optional), the classifier will not be
    wrapped into an imputer. Do this if the classifier doesn't mind if the
    examples it is given have missing values.

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

    So "learning" goes like this. :obj:`ImputeLearner` will first construct
    the imputer (that is, call :obj:`self.imputer_constructor` to get a (trained)
    imputer. Than it will use the imputer to impute the data, and call the
    given :obj:`baseLearner` to construct a classifier. For instance,
    :obj:`baseLearner` could be a learner for logistic regression and the
    result would be a logistic regression model. If the classifier can handle
    unknown values (that is, if :obj:`dont_impute_classifier`, we return it as
    it is, otherwise we wrap it into :obj:`ImputeClassifier`, which is given
    the base classifier and the imputer which it can use to impute the missing
    values in (testing) examples.

.. class:: ImputeClassifier

    Objects of this class are returned by :obj:`ImputeLearner` when given data.

    .. attribute:: baseClassifier

    A wrapped classifier.

    .. attribute:: imputer

    An imputer for imputation of unknown values.

    .. method:: __call__

    This class is even more trivial than the learner. Its constructor accepts
    two arguments, the classifier and the imputer, which are stored into the
    corresponding attributes. The call operator which does the classification
    then looks like this::

        def __call__(self, ex, what=orange.GetValue):
            return self.base_classifier(self.imputer(ex), what)

    It imputes the missing values by calling the :obj:`imputer` and passes the
    class to the base classifier.

.. note::
   In this setup the imputer is trained on the training data - even if you do
   cross validation, the imputer will be trained on the right data. In the
   classification phase we again use the imputer which was classified on the
   training data only.

.. rubric:: Code of ImputeLearner and ImputeClassifier

:obj:`Orange.feature.imputation.ImputeLearner` puts the keyword arguments into
the instance's  dictionary. You are expected to call it like
:obj:`ImputeLearner(base_learner=<someLearner>,
imputer=<someImputerConstructor>)`. When the learner is called with
examples, it
trains the imputer, imputes the data, induces a :obj:`base_classifier` by the
:obj:`base_cearner` and constructs :obj:`ImputeClassifier` that stores the
:obj:`base_classifier` and the :obj:`imputer`. For classification, the missing
values are imputed and the classifier's prediction is returned.

Note that this code is slightly simplified, although the omitted details handle
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

.. rubric:: Example

Although most Orange's learning algorithms will take care of imputation
internally, if needed, it can sometime happen that an expert will be able to
tell you exactly what to put in the data instead of the missing values. In this
example we shall suppose that we want to impute the minimal value of each
feature. We will try to determine whether the naive Bayesian classifier with
its  implicit internal imputation works better than one that uses imputation by
minimal values.

:download:`imputation-minimal-imputer.py <code/imputation-minimal-imputer.py>` (uses :download:`voting.tab <code/voting.tab>`):

.. literalinclude:: code/imputation-minimal-imputer.py
    :lines: 7-

Should ouput this::

    Without imputation: 0.903
    With imputation: 0.899

.. note::
   Note that we constructed just one instance of \
   :obj:`Orange.classification.bayes.NaiveLearner`, but this same instance is
   used twice in each fold, once it is given the examples as they are (and
   returns an instance of :obj:`Orange.classification.bayes.NaiveClassifier`.
   The second time it is called by :obj:`imba` and the \
   :obj:`Orange.classification.bayes.NaiveClassifier` it returns is wrapped
   into :obj:`Orange.feature.imputation.Classifier`. We thus have only one
   learner, but which produces two different classifiers in each round of
   testing.

Write your own imputer
======================

Imputation classes provide the Python-callback functionality (not all Orange
classes do so, refer to the documentation on `subtyping the Orange classes
in Python <callbacks.htm>`_ for a list). If you want to write your own
imputation constructor or an imputer, you need to simply program a Python
function that will behave like the built-in Orange classes (and even less,
for imputer, you only need to write a function that gets an example as
argument, imputation for example tables will then use that function).

You will most often write the imputation constructor when you have a special
imputation procedure or separate procedures for various attributes, as we've
demonstrated in the description of
:obj:`Orange.feature.imputation.ImputerConstructor_model`. You basically only
need to pack everything we've written there to an imputer constructor that
will accept a data set and the id of the weight meta-attribute (ignore it if
you will, but you must accept two arguments), and return the imputer (probably
:obj:`Orange.feature.imputation.Imputer_model`. The benefit of implementing an
imputer constructor as opposed to what we did above is that you can use such a
constructor as a component for Orange learners (like logistic regression) or
for wrappers from module orngImpute, and that way properly use the in
classifier testing procedures.
