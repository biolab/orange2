"""

.. index:: imputation

.. index:: 
   single: feature; value imputation


Imputation is a procedure of replacing the missing feature values with some 
appropriate values. Imputation is needed because of the methods (learning 
algorithms and others) that are not capable of handling unknown values, for 
instance logistic regression.

Missing values sometimes have a special meaning, so they need to be replaced
by a designated value. Sometimes we know what to replace the missing value
with; for instance, in a medical problem, some laboratory tests might not be
done when it is known what their results would be. In that case, we impute 
certain fixed value instead of the missing. In the most complex case, we assign
values that are computed based on some model; we can, for instance, impute the
average or majority value or even a value which is computed from values of
other, known feature, using a classifier.

In a learning/classification process, imputation is needed on two occasions.
Before learning, the imputer needs to process the training examples.
Afterwards, the imputer is called for each example to be classified.

In general, imputer itself needs to be trained. This is, of course, not needed
when the imputer imputes certain fixed value. However, when it imputes the
average or majority value, it needs to compute the statistics on the training
examples, and use it afterwards for imputation of training and testing
examples.

While reading this document, bear in mind that imputation is a part of the
learning process. If we fit the imputation model, for instance, by learning
how to predict the feature's value from other features, or even if we 
simply compute the average or the minimal value for the feature and use it
in imputation, this should only be done on learning data. If cross validation
is used for sampling, imputation should be done on training folds only. Orange
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

Wrapper for learning algorithms
===============================

This wrapper can be used with learning algorithms that cannot handle missing
values: it will impute the missing examples using the imputer, call the 
earning and, if the imputation is also needed by the classifier, wrap the
resulting classifier into another wrapper that will impute the missing values
in examples to be classified.

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

    .. attribute:: baseLearner 
    
    A wrapped learner.

    .. attribute:: imputerConstructor
    
    An instance of a class derived from :obj:`ImputerConstructor` (or a class
    with the same call operator).

    .. attribute:: dontImputeClassifier

    If given and set (this attribute is optional), the classifier will not be
    wrapped into an imputer. Do this if the classifier doesn't mind if the
    examples it is given have missing values.

    The learner is best illustrated by its code - here's its complete
    :obj:`__call__` method::

        def __call__(self, data, weight=0):
            trained_imputer = self.imputerConstructor(data, weight)
            imputed_data = trained_imputer(data, weight)
            baseClassifier = self.baseLearner(imputed_data, weight)
            if self.dontImputeClassifier:
                return baseClassifier
            else:
                return ImputeClassifier(baseClassifier, trained_imputer)

    So "learning" goes like this. :obj:`ImputeLearner` will first construct
    the imputer (that is, call :obj:`self.imputerConstructor` to get a (trained)
    imputer. Than it will use the imputer to impute the data, and call the
    given :obj:`baseLearner` to construct a classifier. For instance,
    :obj:`baseLearner` could be a learner for logistic regression and the
    result would be a logistic regression model. If the classifier can handle
    unknown values (that is, if :obj:`dontImputeClassifier`, we return it as 
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
            return self.baseClassifier(self.imputer(ex), what)

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
:obj:`ImputeLearner(baseLearner=<someLearner>,
imputer=<someImputerConstructor>)`. When the learner is called with examples, it
trains the imputer, imputes the data, induces a :obj:`baseClassifier` by the
:obj:`baseLearner` and constructs :obj:`ImputeClassifier` that stores the
:obj:`baseClassifier` and the :obj:`imputer`. For classification, the missing
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
            trained_imputer = self.imputerConstructor(data, weight)
            imputed_data = trained_imputer(data, weight)
            baseClassifier = self.baseLearner(imputed_data, weight)
            return ImputeClassifier(baseClassifier, trained_imputer)
    
    class ImputeClassifier(orange.Classifier):
        def __init__(self, baseClassifier, imputer):
            self.baseClassifier = baseClassifier
            self.imputer = imputer
    
        def __call__(self, ex, what=orange.GetValue):
            return self.baseClassifier(self.imputer(ex), what)

.. rubric:: Example

Although most Orange's learning algorithms will take care of imputation
internally, if needed, it can sometime happen that an expert will be able to
tell you exactly what to put in the data instead of the missing values. In this
example we shall suppose that we want to impute the minimal value of each
feature. We will try to determine whether the naive Bayesian classifier with
its  implicit internal imputation works better than one that uses imputation by 
minimal values.

`imputation-minimal-imputer.py`_ (uses `voting.tab`_):

.. literalinclude:: code/imputation-minimal-imputer.py
    :lines: 7-
    
Should ouput this::

    Without imputation: 0.903
    With imputation: 0.899

.. note:: 
   Note that we constructed just one instance of
   :obj:`Orange.classification.bayes.NaiveLearner`, but this same instance is
   used twice in each fold, once it is given the examples as they are (and 
   returns an instance of :obj:`Orange.classification.bayes.NaiveClassifier`.
   The second time it is called by :obj:`imba` and the 
   :obj:`Orange.classification.bayes.NaiveClassifier` it returns is wrapped
   into :obj:`Orange.feature.imputation.Classifier`. We thus have only one
   learner, but which produces two different classifiers in each round of
   testing.

Abstract imputers
=================

As common in Orange, imputation is done by pairs of two classes: one that does
the work and another that constructs it. :obj:`ImputerConstructor` is an
abstract root of the hierarchy of classes that get the training data (with an 
optional id for weight) and constructs an instance of a class, derived from
:obj:`Imputer`. An :obj:`Imputer` can be called with an
:obj:`Orange.data.Instance` and it will return a new example with the missing
values imputed (it will leave the original example intact!). If imputer is
called with an :obj:`Orange.data.Table`, it will return a new example table
with imputed examples.

.. class:: ImputerConstructor

    .. attribute:: imputeClass
    
    Tell whether to impute the class value (default) or not.

Simple imputation
=================

The simplest imputers always impute the same value for a particular attribute,
disregarding the values of other attributes. They all use the same imputer
class, :obj:`Imputer_defaults`.
    
.. class:: Imputer_defaults

    .. attribute::  defaults
    
    An example with the default values to be imputed instead of the missing. 
    Examples to be imputed must be from the same domain as :obj:`defaults`.

    Instances of this class can be constructed by 
    :obj:`Orange.feature.imputation.ImputerConstructor_minimal`, 
    :obj:`Orange.feature.imputation.ImputerConstructor_maximal`,
    :obj:`Orange.feature.imputation.ImputerConstructor_average`. 

    For continuous features, they will impute the smallest, largest or the
    average  values encountered in the training examples. For discrete, they
    will impute the lowest (the one with index 0, e. g. attr.values[0]), the 
    highest (attr.values[-1]), and the most common value encountered in the
    data. The first two imputers will mostly be used when the discrete values
    are ordered according to their impact on the class (for instance, possible
    values for symptoms of some disease can be ordered according to their
    seriousness). The minimal and maximal imputers will then represent
    optimistic and pessimistic imputations.

    The following code will load the bridges data, and first impute the values
    in a single examples and then in the whole table.

`imputation-complex.py`_ (uses `bridges.tab`_):

.. literalinclude:: code/imputation-complex.py
    :lines: 9-23

This is example shows what the imputer does, not how it is to be used. Don't
impute all the data and then use it for cross-validation. As warned at the top
of this page, see the instructions for actual `use of
imputers <#using-imputers>`_.

.. note:: :obj:`ImputerConstructor` are another class with schizophrenic
  constructor: if you give the constructor the data, it will return an
  :obj:`Imputer` - the above call is equivalent to calling
  :obj:`Orange.feature.imputation.ImputerConstructor_minimal()(data)`.

You can also construct the :obj:`Orange.feature.imputation.Imputer_defaults`
yourself and specify your own defaults. Or leave some values unspecified, in
which case the imputer won't impute them, as in the following example. Here,
the only attribute whose values will get imputed is "LENGTH"; the imputed value
will be 1234.

`imputation-complex.py`_ (uses `bridges.tab`_):

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

    .. attribute:: imputeClass
    
    Tells whether to impute the class values or not. Defaults to :obj:`True`.

    .. attribute:: deterministic

    If true (default is :obj:`False`), random generator is initialized for each
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

    .. attribute:: learnerDiscrete, learnerContinuous
    
    Learner for discrete and for continuous attributes. If any of them is
    missing, the attributes of the corresponding type won't get imputed.

    .. attribute:: useClass
    
    Tells whether the imputer is allowed to use the class value. As this is
    most often undesired, this option is by default set to :obj:`False`. It can
    however be useful for a more complex design in which we would use one
    imputer for learning examples (this one would use the class value) and
    another for testing examples (which would not use the class value as this
    is unavailable at that moment).

..class:: Imputer_model

    .. attribute: models

    A list of classifiers, each corresponding to one attribute of the examples
    whose values are to be imputed. The :obj:`classVar`'s of the models should
    equal the examples' attributes. If any of classifier is missing (that is,
    the corresponding element of the table is :obj:`None`, the corresponding
    attribute's values will not be imputed.

.. rubric:: Examples

The following imputer predicts the missing attribute values using
classification and regression trees with the minimum of 20 examples in a leaf.

<P class="header">part of <A href="imputation.py">imputation.py</A> (uses <a
href="bridges.tab">bridges.tab</a>)</P> <XMP class=code>import orngTree imputer
= orange.ImputerConstructor_model() imputer.learnerContinuous =
imputer.learnerDiscrete = orngTree.TreeLearner(minSubset = 20) imputer =
imputer(data) </XMP>

<P>We could even use the same learner for discrete and continuous attributes!
(The way this functions is rather tricky. If you desire to know:
<CODE>orngTree.TreeLearner</CODE> is a learning algorithm written in Python -
Orange doesn't mind, it will wrap it into a C++ wrapper for a Python-written
learners which then call-backs the Python code. When given the examples to
learn from, <CODE>orngTree.TreeLearner</CODE> checks the class type. If it's
continuous, it will set the <CODE>orange.TreeLearner</CODE> to construct
regression trees, and if it's discrete, it will set the components for
classification trees. The common parameters, such as the minimal number of
examples in leaves, are used in both cases.)</P>

<P>You can of course use different learning algorithms for discrete and
continuous attributes. Probably a common setup will be to use
<CODE>BayesLearner</CODE> for discrete and <CODE>MajorityLearner</CODE> (which
just remembers the average) for continuous attributes, as follows.</P>

<P class="header">part of <A href="imputation.py">imputation.py</A> (uses <a
href="bridges.tab">bridges.tab</a>)</P> <XMP class=code>imputer =
orange.ImputerConstructor_model() imputer.learnerContinuous =
orange.MajorityLearner() imputer.learnerDiscrete = orange.BayesLearner()
imputer = imputer(data) </XMP>

<P>You can also construct an <CODE>Imputer_model</CODE> yourself. You will do this if different attributes need different treatment. Brace for an example that will be a bit more complex. First we shall construct an <CODE>Imputer_model</CODE> and initialize an empty list of models.</P>

<P class="header">part of <A href="imputation.py">imputation.py</A> (uses <a href="bridges.tab">bridges.tab</a>)</P>
<XMP class=code>imputer = orange.Imputer_model()
imputer.models = [None] * len(data.domain)
</XMP>

<P>Attributes "LANES" and "T-OR-D" will always be imputed values 2 and
"THROUGH". Since "LANES" is continuous, it suffices to construct a
<CODE>DefaultClassifier</CODE> with the default value 2.0 (don't forget the
decimal part, or else Orange will think you talk about an index of a discrete
value - how could it tell?). For the discrete attribute "T-OR-D", we could
construct a <CODE>DefaultClassifier</CODE> and give the index of value
"THROUGH" as an argument. But we shall do it nicer, by constructing a
<CODE>Value</CODE>. Both classifiers will be stored at the appropriate places
in <CODE>imputer.models</CODE>.</P>

<XMP class=code>imputer.models[data.domain.index("LANES")] = orange.DefaultClassifier(2.0)

tord = orange.DefaultClassifier(orange.Value(data.domain["T-OR-D"], "THROUGH"))
imputer.models[data.domain.index("T-OR-D")] = tord
</XMP>

<P>"LENGTH" will be computed with a regression tree induced from "MATERIAL", "SPAN" and "ERECTED" (together with "LENGTH" as the class attribute, of course). Note that we initialized the domain by simply giving a list with the names of the attributes, with the domain as an additional argument in which Orange will look for the named attributes.</P>

<XMP class=code>import orngTree
len_domain = orange.Domain(["MATERIAL", "SPAN", "ERECTED", "LENGTH"], data.domain)
len_data = orange.ExampleTable(len_domain, data)
len_tree = orngTree.TreeLearner(len_data, minSubset=20)
imputer.models[data.domain.index("LENGTH")] = len_tree
orngTree.printTxt(len_tree)
</XMP>

<P>We printed the tree just to see what it looks like.</P>

<XMP class=code>SPAN=SHORT: 1158
SPAN=LONG: 1907
SPAN=MEDIUM
|    ERECTED<1908.500: 1325
|    ERECTED>=1908.500: 1528
</XMP>

<P>Small and nice. Now for the "SPAN". Wooden bridges and walkways are short, while the others are mostly medium. This could be done by <a href="lookup.htm"><CODE>ClassifierByLookupTable</CODE></A> - this would be faster than what we plan here. See the corresponding documentation on lookup classifier. Here we are gonna do it with a Python function.</P>

<XMP class=code>spanVar = data.domain["SPAN"]

def computeSpan(ex, returnWhat):
    if ex["TYPE"] == "WOOD" or ex["PURPOSE"] == "WALK":
        span = "SHORT"
    else:
        span = "MEDIUM"
    return orange.Value(spanVar, span)

imputer.models[data.domain.index("SPAN")] = computeSpan
</XMP>


<P><CODE>computeSpan</CODE> could also be written as a class, if you'd prefer
it. It's important that it behaves like a classifier, that is, gets an example
and returns a value. The second element tells, as usual, what the caller expect
the classifier to return - a value, a distribution or both. Since the caller,
<CODE>Imputer_model</CODE>, always wants values, we shall ignore the argument
(at risk of having problems in the future when imputers might handle
distribution as well).</P>


Treating the missing values as special values
=============================================

<P>Missing values sometimes have a special meaning. The fact that something was
not measured can sometimes tell a lot. Be, however, cautious when using such
values in decision models; it the decision not to measure something (for
instance performing a laboratory test on a patient) is based on the expert's
knowledge of the class value, such unknown values clearly should not be used in
models.</P>

<P><CODE><INDEX name="classes/ImputerConstructor_asValue">ImputerConstructor_asValue</INDEX></CODE> constructs a new domain in which each discrete attribute is replaced with a new attribute that has one value more: "NA". The new attribute will compute its values on the fly from the old one, copying the normal values and replacing the unknowns with "NA".</P>

<P>For continuous attributes, <CODE>ImputerConstructor_asValue</CODE> will
construct a two-valued discrete attribute with values "def" and "undef",
telling whether the continuous attribute was defined or not. The attribute's
name will equal the original's with "_def" appended. The original continuous
attribute will remain in the domain and its unknowns will be replaced by
averages.</P>

<P><CODE>ImputerConstructor_asValue</CODE> has no specific attributes.</P>

<P>The constructed imputer is named <CODE>Imputer_asValue</CODE> (I bet you
wouldn't guess). It converts the example into the new domain, which imputes the
values for discrete attributes. If continuous attributes are present, it will
also replace their values by the averages.</P>

<P class=section>Attributes of <CODE>Imputer_asValue</CODE></P>
<DL class=attributes>
<DT>domain</DT>
<DD>The domain with the new attributes constructed by <CODE>ImputerConstructor_asValue</CODE>.</DD>

<DT>defaults</DT>
<DD>Default values for continuous attributes. Present only if there are any.</DD>
</DL>

<P>Here's a script that shows what this imputer actually does to the domain.</P>

<P class="header">part of <A href="imputation.py">imputation.py</A> (uses <a href="bridges.tab">bridges.tab</a>)</P>
<XMP class=code>imputer = orange.ImputerConstructor_asValue(data)

original = data[19]
imputed = imputer(data[19])

print original.domain
print
print imputed.domain
print

for i in original.domain:
    print "%s: %s -> %s" % (original.domain[i].name, original[i], imputed[i.name]),
    if original.domain[i].varType == orange.VarTypes.Continuous:
        print "(%s)" % imputed[i.name+"_def"]
    else:
        print
print
</XMP>

<P>The script's output looks like this.</P>

<XMP class=code>[RIVER, ERECTED, PURPOSE, LENGTH, LANES, CLEAR-G, T-OR-D,
MATERIAL, SPAN, REL-L, TYPE]

[RIVER, ERECTED_def, ERECTED, PURPOSE, LENGTH_def, LENGTH,
LANES_def, LANES, CLEAR-G, T-OR-D,
MATERIAL, SPAN, REL-L, TYPE]


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
</XMP>

<P>Seemingly, the two examples have the same attributes (with
<CODE>imputed</CODE> having a few additional ones). If you check this by
<CODE>original.domain[0] == imputed.domain[0]</CODE>, you shall see that this
first glance is <CODE>False</CODE>. The attributes only have the same names,
but they are different attributes. If you read this page (which is already a
bit advanced), you know that Orange does not really care about the attribute
names).</P>

<P>Therefore, if we wrote "<CODE>imputed[i]</CODE>" the program would fail
since <CODE>imputed</CODE> has no attribute <CODE>i</CODE>. But it has an
attribute with the same name (which even usually has the same value). We
therefore use <CODE>i.name</CODE> to index the attributes of
<CODE>imputed</CODE>. (Using names for indexing is not fast, though; if you do
it a lot, compute the integer index with
<CODE>imputed.domain.index(i.name)</CODE>.)</P>

<P>For continuous attributes, there is an additional attribute with "_def"
appended; we get it by <CODE>i.name+"_def"</CODE>. Not really nice, but it
works.</P>

<P>The first continuous attribute, "ERECTED" is defined. Its value remains 1874
and the additional attribute "ERECTED_def" has value "def". Not so for
"LENGTH". Its undefined value is replaced by the average (1567) and the new
attribute has value "undef". The undefined discrete attribute "CLEAR-G" (and
all other undefined discrete attributes) is assigned the value "NA".</P>

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

.. _imputation-minimal-imputer.py: code/imputation-minimal-imputer.py
.. _imputation-complex.py: code/imputation-complex.py
.. _voting.tab: code/voting.tab
.. _bridges.tab: code/bridges.tab

"""

import Orange.core as orange
from orange import ImputerConstructor_minimal 
from orange import ImputerConstructor_maximal
from orange import ImputerConstructor_average
from orange import Imputer_defaults
from orange import ImputerConstructor_model
from orange import Imputer_model
from orange import ImputerConstructor_asValue 

class ImputeLearner(orange.Learner):
    def __new__(cls, examples = None, weightID = 0, **keyw):
        self = orange.Learner.__new__(cls, **keyw)
        self.dontImputeClassifier = False
        self.__dict__.update(keyw)
        if examples:
            return self.__call__(examples, weightID)
        else:
            return self
        
    def __call__(self, data, weight=0):
        trained_imputer = self.imputerConstructor(data, weight)
        imputed_data = trained_imputer(data, weight)
        baseClassifier = self.baseLearner(imputed_data, weight)
        if self.dontImputeClassifier:
            return baseClassifier
        else:
            return ImputeClassifier(baseClassifier, trained_imputer)

class ImputeClassifier(orange.Classifier):
    def __init__(self, baseClassifier, imputer, **argkw):
        self.baseClassifier = baseClassifier
        self.imputer = imputer
        self.__dict__.update(argkw)

    def __call__(self, ex, what=orange.GetValue):
        return self.baseClassifier(self.imputer(ex), what)
