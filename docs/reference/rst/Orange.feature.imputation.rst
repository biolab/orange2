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

:obj:`ImputerConstructor` is the abstract root in the hierarchy of classes
that get training data and construct an instance of a class derived from
:obj:`Imputer`. When an :obj:`Imputer` is called with an
:obj:`Orange.data.Instance` it will return a new example with the
missing values imputed (leaving the original example intact). If imputer is
called with an :obj:`Orange.data.Table` it will return a new example table
with imputed instances.

.. class:: ImputerConstructor

    .. attribute:: imputeClass

    Indicates whether to impute the class value. Default is True.

    .. attribute:: deterministic

    Indicates whether to initialize random by example's CRC. Default is False.

Simple imputation
=================

Simple imputers always impute the same value for a particular attribute,
disregarding the values of other attributes. They all use the same class
:obj:`Imputer_defaults`.

.. class:: Imputer_defaults

    .. attribute::  defaults

    An instance :obj:`Orange.data.Instance` with the default values to be
    imputed instead of missing. Examples to be imputed must be from the same
    domain as :obj:`defaults`.

Instances of this class can be constructed by
:obj:`Orange.feature.imputation.ImputerConstructor_minimal`,
:obj:`Orange.feature.imputation.ImputerConstructor_maximal`,
:obj:`Orange.feature.imputation.ImputerConstructor_average`.

For continuous features, they will impute the smallest,
largest or the average values encountered in the training examples.

For discrete, they will impute the lowest (the one with index 0,
e. g. attr.values[0]), the highest (attr.values[-1]),
and the most common value encountered in the data.

The first two imputers
will mostly be used when the discrete values are ordered according to their
impact on the class (for instance, possible values for symptoms of some
disease can be ordered according to their seriousness). The minimal and maximal
imputers will then represent optimistic and pessimistic imputations.

The following code will load the bridges data, and first impute the values
in a single examples and then in the whole table.

:download:`imputation-complex.py <code/imputation-complex.py>` (uses :download:`bridges.tab <code/bridges.tab>`):

.. literalinclude:: code/imputation-complex.py
    :lines: 9-23

This is example shows what the imputer does, not how it is to be used. Don't
impute all the data and then use it for cross-validation. As warned at the top
of this page, see the instructions for actual `use of
imputers <#using-imputers>`_.

.. note:: The :obj:`ImputerConstructor` are another class with schizophrenic
  constructor: if you give the constructor the data, it will return an \
  :obj:`Imputer` - the above call is equivalent to calling \
  :obj:`Orange.feature.imputation.ImputerConstructor_minimal()(data)`.

You can also construct the :obj:`Orange.feature.imputation.Imputer_defaults`
yourself and specify your own defaults. Or leave some values unspecified, in
which case the imputer won't impute them, as in the following example. Here,
the only attribute whose values will get imputed is "LENGTH"; the imputed value
will be 1234.

.. literalinclude:: code/imputation-complex.py
    :lines: 56-69

:obj:`Orange.feature.imputation.Imputer_defaults`'s constructor will accept an
argument of type :obj:`Orange.data.Domain` (in which case it will construct an
empty instance for :obj:`defaults`) or an example. (Be careful with this:
:obj:`Orange.feature.imputation.Imputer_defaults` will have a reference to the
instance and not a copy. But you can make a copy yourself to avoid problems:
instead of `Imputer_defaults(data[0])` you may want to write
`Imputer_defaults(Orange.data.Instance(data[0]))`.

Random imputation
=================

.. class:: Imputer_Random

    Imputes random values. The corresponding constructor is
    :obj:`ImputerConstructor_Random`.

    .. attribute:: impute_class

    Tells whether to impute the class values or not. Defaults to True.

    .. attribute:: deterministic

    If true (default is False), random generator is initialized for each
    example using the example's hash value as a seed. This results in same
    examples being always imputed the same values.

Model-based imputation
======================

.. class:: ImputerConstructor_model

    Model-based imputers learn to predict the attribute's value from values of
    other attributes. :obj:`ImputerConstructor_model` are given a learning
    algorithm (two, actually - one for discrete and one for continuous
    attributes) and they construct a classifier for each attribute. The
    constructed imputer :obj:`Imputer_model` stores a list of classifiers which
    are used when needed.

    .. attribute:: learner_discrete, learner_continuous

    Learner for discrete and for continuous attributes. If any of them is
    missing, the attributes of the corresponding type won't get imputed.

    .. attribute:: use_class

    Tells whether the imputer is allowed to use the class value. As this is
    most often undesired, this option is by default set to False. It can
    however be useful for a more complex design in which we would use one
    imputer for learning examples (this one would use the class value) and
    another for testing examples (which would not use the class value as this
    is unavailable at that moment).

.. class:: Imputer_model

    .. attribute: models

    A list of classifiers, each corresponding to one attribute of the examples
    whose values are to be imputed. The :obj:`classVar`'s of the models should
    equal the examples' attributes. If any of classifier is missing (that is,
    the corresponding element of the table is :obj:`None`, the corresponding
    attribute's values will not be imputed.

.. rubric:: Examples

The following imputer predicts the missing attribute values using
classification and regression trees with the minimum of 20 examples in a leaf.
Part of :download:`imputation-complex.py <code/imputation-complex.py>` (uses :download:`bridges.tab <code/bridges.tab>`):

.. literalinclude:: code/imputation-complex.py
    :lines: 74-76

We could even use the same learner for discrete and continuous attributes,
as :class:`Orange.classification.tree.TreeLearner` checks the class type
and constructs regression or classification trees accordingly. The
common parameters, such as the minimal number of
examples in leaves, are used in both cases.

You can also use different learning algorithms for discrete and
continuous attributes. Probably a common setup will be to use
:class:`Orange.classification.bayes.BayesLearner` for discrete and
:class:`Orange.regression.mean.MeanLearner` (which
just remembers the average) for continuous attributes. Part of
:download:`imputation-complex.py <code/imputation-complex.py>` (uses :download:`bridges.tab <code/bridges.tab>`):

.. literalinclude:: code/imputation-complex.py
    :lines: 91-94

You can also construct an :class:`Imputer_model` yourself. You will do
this if different attributes need different treatment. Brace for an
example that will be a bit more complex. First we shall construct an
:class:`Imputer_model` and initialize an empty list of models.
The following code snippets are from
:download:`imputation-complex.py <code/imputation-complex.py>` (uses :download:`bridges.tab <code/bridges.tab>`):

.. literalinclude:: code/imputation-complex.py
    :lines: 108-109

Attributes "LANES" and "T-OR-D" will always be imputed values 2 and
"THROUGH". Since "LANES" is continuous, it suffices to construct a
:obj:`DefaultClassifier` with the default value 2.0 (don't forget the
decimal part, or else Orange will think you talk about an index of a discrete
value - how could it tell?). For the discrete attribute "T-OR-D", we could
construct a :class:`Orange.classification.ConstantClassifier` and give the index of value
"THROUGH" as an argument. But we shall do it nicer, by constructing a
:class:`Orange.data.Value`. Both classifiers will be stored at the appropriate places
in :obj:`imputer.models`.

.. literalinclude:: code/imputation-complex.py
    :lines: 110-112


"LENGTH" will be computed with a regression tree induced from "MATERIAL",
"SPAN" and "ERECTED" (together with "LENGTH" as the class attribute, of
course). Note that we initialized the domain by simply giving a list with
the names of the attributes, with the domain as an additional argument
in which Orange will look for the named attributes.

.. literalinclude:: code/imputation-complex.py
    :lines: 114-119

We printed the tree just to see what it looks like.

::

    <XMP class=code>SPAN=SHORT: 1158
    SPAN=LONG: 1907
    SPAN=MEDIUM
    |    ERECTED<1908.500: 1325
    |    ERECTED>=1908.500: 1528
    </XMP>

Small and nice. Now for the "SPAN". Wooden bridges and walkways are short,
while the others are mostly medium. This could be done by
:class:`Orange.classifier.ClassifierByLookupTable` - this would be faster
than what we plan here. See the corresponding documentation on lookup
classifier. Here we are going to do it with a Python function.

.. literalinclude:: code/imputation-complex.py
    :lines: 121-128

:obj:`compute_span` could also be written as a class, if you'd prefer
it. It's important that it behaves like a classifier, that is, gets an example
and returns a value. The second element tells, as usual, what the caller expect
the classifier to return - a value, a distribution or both. Since the caller,
:obj:`Imputer_model`, always wants values, we shall ignore the argument
(at risk of having problems in the future when imputers might handle
distribution as well).

Missing values as special values
================================

Missing values sometimes have a special meaning. The fact that something was
not measured can sometimes tell a lot. Be, however, cautious when using such
values in decision models; it the decision not to measure something (for
instance performing a laboratory test on a patient) is based on the expert's
knowledge of the class value, such unknown values clearly should not be used
in models.

.. class:: ImputerConstructor_asValue

    Constructs a new domain in which each
    discrete attribute is replaced with a new attribute that has one value more:
    "NA". The new attribute will compute its values on the fly from the old one,
    copying the normal values and replacing the unknowns with "NA".

    For continuous attributes, it will
    construct a two-valued discrete attribute with values "def" and "undef",
    telling whether the continuous attribute was defined or not. The attribute's
    name will equal the original's with "_def" appended. The original continuous
    attribute will remain in the domain and its unknowns will be replaced by
    averages.

    :class:`ImputerConstructor_asValue` has no specific attributes.

    It constructs :class:`Imputer_asValue` (I bet you
    wouldn't guess). It converts the example into the new domain, which imputes
    the values for discrete attributes. If continuous attributes are present, it
    will also replace their values by the averages.

.. class:: Imputer_asValue

    .. attribute:: domain

        The domain with the new attributes constructed by
        :class:`ImputerConstructor_asValue`.

    .. attribute:: defaults

        Default values for continuous attributes. Present only if there are any.

The following code shows what this imputer actually does to the domain.
Part of :download:`imputation-complex.py <code/imputation-complex.py>` (uses :download:`bridges.tab <code/bridges.tab>`):

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

Seemingly, the two examples have the same attributes (with
:samp:`imputed` having a few additional ones). If you check this by
:samp:`original.domain[0] == imputed.domain[0]`, you shall see that this
first glance is False. The attributes only have the same names,
but they are different attributes. If you read this page (which is already a
bit advanced), you know that Orange does not really care about the attribute
names).

Therefore, if we wrote :samp:`imputed[i]` the program would fail
since :samp:`imputed` has no attribute :samp:`i`. But it has an
attribute with the same name (which even usually has the same value). We
therefore use :samp:`i.name` to index the attributes of
:samp:`imputed`. (Using names for indexing is not fast, though; if you do
it a lot, compute the integer index with
:samp:`imputed.domain.index(i.name)`.)</P>

For continuous attributes, there is an additional attribute with "_def"
appended; we get it by :samp:`i.name+"_def"`.

The first continuous attribute, "ERECTED" is defined. Its value remains 1874
and the additional attribute "ERECTED_def" has value "def". Not so for
"LENGTH". Its undefined value is replaced by the average (1567) and the new
attribute has value "undef". The undefined discrete attribute "CLEAR-G" (and
all other undefined discrete attributes) is assigned the value "NA".

Using imputers
==============

To properly use the imputation classes in learning process, they must be
trained on training examples only. Imputing the missing values and subsequently
using the data set in cross-validation will give overly optimistic results.

Learners with imputer as a component
------------------------------------

Orange learners that cannot handle missing values will generally provide a slot
for the imputer component. An example of such a class is
:obj:`Orange.classification.logreg.LogRegLearner` with an attribute called
:obj:`Orange.classification.logreg.LogRegLearner.imputerConstructor`. To it you
can assign an imputer constructor - one of the above constructors or a specific
constructor you wrote yourself. When given learning examples,
:obj:`Orange.classification.logreg.LogRegLearner` will pass them to
:obj:`Orange.classification.logreg.LogRegLearner.imputerConstructor` to get an
imputer (again some of the above or a specific imputer you programmed). It will
immediately use the imputer to impute the missing values in the learning data
set, so it can be used by the actual learning algorithm. Besides, when the
classifier :obj:`Orange.classification.logreg.LogRegClassifier` is constructed,
the imputer will be stored in its attribute
:obj:`Orange.classification.logreg.LogRegClassifier.imputer`. At
classification, the imputer will be used for imputation of missing values in
(testing) examples.

Although details may vary from algorithm to algorithm, this is how the
imputation is generally used in Orange's learners. Also, if you write your own
learners, it is recommended that you use imputation according to the described
procedure.

Wrapper for learning algorithms
===============================

Imputation is used by learning algorithms and other methods that are not
capable of handling unknown values. It will impute missing values,
call the learner and, if imputation is also needed by the classifier,
it will wrap the classifier into a wrapper that imputes missing values in
examples to classify.

.. literalinclude:: code/imputation-logreg.py
   :lines: 7-

The output of this code is::

    Without imputation: 0.945
    With imputation: 0.954

Even so, the module is somewhat redundant, as all learners that cannot handle
missing values should, in principle, provide the slots for imputer constructor.
For instance, :obj:`Orange.classification.logreg.LogRegLearner` has an attribute
:obj:`Orange.classification.logreg.LogRegLearner.imputerConstructor`, and even
if you don't set it, it will do some imputation by default.

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
imputer=<someImputerConstructor>)`. When the learner is called with examples, it
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
