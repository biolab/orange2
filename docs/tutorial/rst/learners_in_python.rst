Build your own learner
======================

.. index::
   single: classifiers; in Python

This part of tutorial will show how to build learners and classifiers
in Python, that is, how to build your own learners and
classifiers. Especially for those of you that want to test some of
your methods or want to combine existing techniques in Orange, this is
a very important topic. Developing your own learners in Python makes
prototyping of new methods fast and enjoyable.

There are different ways to build learners/classifiers in Python. We
will take the route that shows how to do this correctly, in a sense
that you will be able to use your learner as it would be any learner
that Orange originally provides. Distinct to Orange learners is the
way how they are invoked and what the return. Let us start with an
example. Say that we have a Learner(), which is some learner in
Orange. The learner can be called in two different ways::

   learner = Learner()
   classifier = Learner(data)

In the first line, the learner is invoked without the data set and
in that case it should return an instance of learner, such that later
you may say ``classifier = learner(data)`` or you may call
some validation procedure with a ``learner`` itself (say
``orngEval.CrossValidation([learner], data)``). In the second
line, learner is called with the data and returns a classifier.

Classifiers should be called with a data instance to classify,
and should return either a class value (by default), probability of
classes or both::

   value = classifier(instance)
   value = classifier(instance, orange.GetValue)
   probabilities = classifier(instance, orange.GetProbabilities)
   value, probabilities = classifier(instance, orange.GetBoth)

Here is a short example::

   > python
   >>> import orange
   >>> data = orange.ExampleTable("voting")
   >>> learner = orange.BayesLearner()
   >>> classifier = learner(data)
   >>> classifier(data[0])
   republican
   >>> classifier(data[0], orange.GetBoth)
   (republican, [0.99999994039535522, 7.9730767765795463e-008])
   >>> classifier(data[0], orange.GetProbabilities)
   [0.99999994039535522, 7.9730767765795463e-008]
   >>> 
   >>> c = orange.BayesLearner(data)
   >>> c(data[12])
   democrat
   >>>

We will here assume that our learner and the corresponding classifier
will be defined in a single file (module) that will not contain any
other code. This helps for code reuse, so that if you want to use your
new method anywhere else, you just import it from that file. Each such
module will contain a class ``Learner_Class`` and a class
``Classifier``. We will use this schema to define a learner that will
use naive Bayesian classifier with embeded categorization of training
data. Then we will show how to write naive Bayesian classifier in
Python (that is, how to do this from scratch). We conclude with Python
implementation of bagging.

.. _naive bayes with discretization:

Naive Bayes with discretization
-------------------------------

Let us build a learner/classifier that is an extension of build-in
naive Bayes and which before learning categorizes the data. We will
define a module :download:`nbdisc.py <code/nbdisc.py>` that will implement two classes, Learner
and Classifier. Following is a Python code for a Learner class (part
of :download:`nbdisc.py <code/nbdisc.py>`)::

   class Learner(object):
       def __new__(cls, examples=None, name='discretized bayes', **kwds):
           learner = object.__new__(cls, **kwds)
           if examples:
               learner.__init__(name) # force init
               return learner(examples)
           else:
               return learner  # invokes the __init__
   
       def __init__(self, name='discretized bayes'):
           self.name = name
   
       def __call__(self, data, weight=None):
           disc = orange.Preprocessor_discretize( \
               data, method=orange.EntropyDiscretization())
           model = orange.BayesLearner(disc, weight)
           return Classifier(classifier = model)

``Learner_Class`` has three methods. Method ``__new__`` creates the
object and returns a learner or classifier, depending if examples
where passed to the call. If the examples were passed as an argument
than the method called the learner (invoking ``__call__``
method). Method ``__init__`` is invoked every time the class is called
for the first time. Notice that all it does is remembers the only
argument that this class can be called with, i.e. the argument
``name`` which defaults to discretized bayes. If you would expect any
other arguments for your learners, you should handle them here (store
them as class' attributes using the keyword ``self``).

If we have created an instance of the learner (and did not pass the
examples as attributes), the next call of this learner will invoke a
method ``__call__``, where the essence of our learner is
implemented. Notice also that we have included an attribute for vector
of instance weights, which is passed to naive Bayesian learner. In our
learner, we first discretize the data using Fayyad &amp; Irani's
entropy-based discretization, then build a naive Bayesian model and
finally pass it to a class ``Classifier``. You may expect that at its
first invocation the ``Classifier`` will just remember the model we
have called it with (part of :download:`nbdisc.py <code/nbdisc.py>`)::

   class Classifier:
       def __init__(self, **kwds):
           self.__dict__.update(kwds)
   
       def __call__(self, example, resultType = orange.GetValue):
           return self.classifier(example, resultType)
   
The method ``__init__`` in ``Classifier`` is rather general: it makes
``Classifier`` remember all arguments it was called with. They are
then accessed through ``Classifiers``' arguments
(``self.argument_name``). When Classifier is called, it expects an
example and an optional argument that specifies the type of result to
be returned.

This completes our code for naive Bayesian classifier with
discretization. You can see that the code is fairly short (fewer than
20 lines), and it can be easily extended or changed if we want to do
something else as well (like feature subset selection, ...).

Here are now a few lines to test our code::

   >>> import orange, nbdisc
   >>> data = orange.ExampleTable("iris")
   >>> classifier = nbdisc.Learner(data)
   >>> print classifier(data[100])
   Iris-virginica
   >>> classifier(data[100], orange.GetBoth)
   (<orange.Value 'iris'='Iris-virginica'>, <0.000, 0.001, 0.999>)
   >>>

For a more elaborate test that also shows the use of a learner (that
is not given the data at its initialization), here is a script that
does 10-fold cross validation (:download:`nbdisc_test.py <code/nbdisc_test.py>`, uses :download:`iris.tab <code/iris.tab>` and
:download:`nbdisc.py <code/nbdisc.py>`)::

   import orange, orngEval, nbdisc
   data = orange.ExampleTable("iris")
   results = orngEval.CrossValidation([nbdisc.Learner()], data)
   print "Accuracy = %5.3f" % orngEval.CA(results)[0]

The accuracy on this data set is about 92%. You may try to obtain a
better accuracy by using some other type of discretization, or try
some other learner on this data (hint: k-NN should perform better).

Python implementation of naive Bayesian classifier
--------------------------------------------------

.. index::
   single: naive Bayesian classifier; in Python

The naive Bayesian classifier we will implement in this lesson uses
standard naive Bayesian algorithm also described in Michell: Machine
Learning, 1997 (pages 177-180). Essentially, if a data instance is
described with :math:`n` features :math:`a_i`, then the
class that instance is classified to a class :math:`c` from set of possible
classes :math:`V`. According to naive Bayes classifier:

.. math::
   c=\arg\max_{c_i\in V} P(v_j)\prod_{i=1}^n P(a_i|v_j)

We will also compute a vector of elements:

.. math::
   p_j = P(v_j)\prod_{i=1}^n P(a_i, v_j)

which, after normalization such that :math:`\sum_j p_j` is
equal to 1, represent class probabilities. The class probabilities and
conditional probabilities (priors) in above formulas are estimated
from training data: class probability is equal to the relative class
frequency, while the conditional probability of attribute value given
class is computed by figuring out the proportion of instances with a
value of :math:`i`-th attribute equal to :math:`a_i` among instances that
from class :math:`v_j`.

To complicate things just a little bit, :math:`m`-estimate (see
Mitchell, and Cestnik IJCAI-1990) will be used instead of relative
frequency when computing prior conditional probabilities. So
(following the example in Mitchell), when assessing :math:`P=P({\rm
Wind}={\rm strong}|{\rm PlayTennis}={\rm no})` we find that the total
number of training examples with PlayTennis=no is :math:`n=5`, and of
these there are :math:`n_c=3` for which Wind=strong, than using
relative frequency the corresponding probability would be:

.. math::
   P={n_c\over n}

Relative frequency has a problem when number of instance is
small, and to alleviate that m-estimate assumes that there are m
imaginary cases (m is also referred to as equivalent sample size)
with equal probability of class values p. Our conditional
probability using m-estimate is then computed as:

.. math::
   P={n_c+m p\over n+m}

Often, instead of uniform class probability :math:`p`, a relative class
frequency as estimated from training data is taken.

We will develop a module called bayes.py that will implement our naive
Bayes learner and classifier. The structure of the module will be as
with `naive bayes with discretization`_.  Again, we will implement two classes, one for
learning and the other on for classification. Here is a ``Learner``:
class (part of :download:`bayes.py <code/bayes.py>`)::

   class Learner_Class:
     def __init__(self, m=0.0, name='std naive bayes', **kwds):
       self.__dict__.update(kwds)
       self.m = m
       self.name = name
   
     def __call__(self, examples, weight=None, **kwds):
       for k in kwds.keys():
         self.__dict__[k] = kwds[k]
       domain = examples.domain
   
       # first, compute class probabilities
       n_class = [0.] * len(domain.classVar.values)
       for e in examples:
         n_class[int(e.getclass())] += 1
   
       p_class = [0.] * len(domain.classVar.values)
       for i in range(len(domain.classVar.values)):
         p_class[i] = n_class[i] / len(examples)
   
       # count examples with specific attribute and
       # class value, pc[attribute][value][class]
   
       # initialization of pc
       pc = []
       for i in domain.attributes:
         p = [[0.]*len(domain.classVar.values) for i in range(len(i.values))]
         pc.append(p)
   
       # count instances, store them in pc
       for e in examples:
         c = int(e.getclass())
         for i in range(len(domain.attributes)):
         if not e[i].isSpecial():
           pc[i][int(e[i])][c] += 1.0
   
       # compute conditional probabilities
       for i in range(len(domain.attributes)):
         for j in range(len(domain.attributes[i].values)):
           for k in range(len(domain.classVar.values)):
             pc[i][j][k] = (pc[i][j][k] + self.m * p_class[k])/ \
               (n_class[k] + self.m)
   
       return Classifier(m = self.m, domain=domain, p_class=p_class, \
                p_cond=pc, name=self.name)

Initialization of ``Learner_Class`` saves the two attributes, ``m``
and ``name`` of the classifier. Notice that both parameters are
optional, and the default value for ``m`` is 0, making naive Bayes
m-estimate equal to relative frequency unless the user specifies some
other value for m. Function ``__call__`` is called with the training
data set, computes class and conditional probabilities and calls
classifiers, passing the probabilities along with some other variables
required for classification (part of :download:`bayes.py <code/bayes.py>`)::

   class Classifier:
     def __init__(self, **kwds):
       self.__dict__.update(kwds)
   
     def __call__(self, example, result_type=orange.GetValue):
       # compute the class probabilities
       p = map(None, self.p_class)
       for c in range(len(self.domain.classVar.values)):
         for a in range(len(self.domain.attributes)):
           if not example[a].isSpecial():
             p[c] *= self.p_cond[a][int(example[a])][c]
   
       # normalize probabilities to sum to 1
       sum =0.
       for pp in p: sum += pp
       if sum>0:
         for i in range(len(p)): p[i] = p[i]/sum
   
       # find the class with highest probability
       v_index = p.index(max(p))
       v = orange.Value(self.domain.classVar, v_index)
   
       # return the value based on requested return type
       if result_type == orange.GetValue:
         return v
       if result_type == orange.GetProbabilities:
         return p
       return (v,p)
   
     def show(self):
       print 'm=', self.m
       print 'class prob=', self.p_class
       print 'cond prob=', self.p_cond
   
Upon first invocation, the classifier will store the values of the
parameters it was called with (``__init__``). When called with a data
instance, it will first compute the class probabilities using the
prior probabilities sent by the learner. The probabilities will be
normalized to sum to 1. The class will then be found that has the
highest probability, and the classifier will accordingly predict to
this class. Notice that we have also added a method called show, which
reports on m, class probabilities and conditional probabilities::

   >>> import orange, bayes
   >>> data = orange.ExampleTable("voting")
   >>> classifier = bayes.Learner(data)
   >>> classifier.show()
   m= 0.0
   class prob= [0.38620689655172413, 0.61379310344827587]
   cond prob= [[[0.79761904761904767, 0.38202247191011235], ...]]
   >>>

The following script tests our naive Bayes, and compares it to
10-nearest neighbors. Running the script (do you it yourself) reports
classification accuracies just about 90% (:download:`bayes_test.py <code/bayes_test.py>`, uses
:download:`bayes.py <code/bayes.py>` and :download:`voting.tab <code/voting.tab>`)::

   import orange, orngEval, bayes
   data = orange.ExampleTable("voting")
   
   bayes = bayes.Learner(m=2, name='my bayes')
   knn = orange.kNNLearner(k=10)
   knn.name = "knn"
   
   learners = [knn,bayes]
   results = orngEval.CrossValidation(learners, data)
   for i in range(len(learners)):
       print learners[i].name, orngEval.CA(results)[i]

Bagging
-------

Here we show how to use the schema that allows us to build our own
learners/classifiers for bagging. While you can find bagging,
boosting, and other ensemble-related stuff in :py:mod:`Orange.ensemble` module, we thought
explaining how to code bagging in Python may provide for a nice
example. The following pseudo-code (from
Whitten &amp; Frank: Data Mining) illustrates the main idea of bagging::

   MODEL GENERATION
   Let n be the number of instances in the training data.
   For each of t iterations:
      Sample n instances with replacement from training data.
      Apply the learning algorithm to the sample.
      Store the resulting model.
   
   CLASSIFICATION
   For each of the t models:
      Predict class of instance using model.
   Return class that has been predicted most often.

Using the above idea, this means that our ``Learner_Class`` will need
to develop t classifiers and will have to pass them to ``Classifier``,
which, once seeing a data instance, will use them for
classification. We will allow parameter t to be specified by the user,
10 being the default.

The code for the ``Learner_Class`` is therefore (part of
:download:`bagging.py <code/bagging.py>`)::

   class Learner_Class:
       def __init__(self, learner, t=10, name='bagged classifier'):
           self.t = t
           self.name = name
           self.learner = learner
   
       def __call__(self, examples, weight=None):
           n = len(examples)
           classifiers = []
           for i in range(self.t):
               selection = []
               for i in range(n):
                   selection.append(random.randrange(n))
               data = examples.getitems(selection)
               classifiers.append(self.learner(data))
               
           return Classifier(classifiers = classifiers, \
               name=self.name, domain=examples.domain)

Upon invocation, ``__init__`` stores the base learning (the one that
will be bagged), the value of the parameter t, and the name of the
classifier. Note that while the learner requires the base learner to
be specified, parameters t and name are optional.

When the learner is called with examples, a list of t classifiers is
build and stored in variable ``classifier``. Notice that for data
sampling with replacement, a list of data instance indices is build
(``selection``) and then used to sample the data from training
examples (``example.getitems``). Finally, a ``Classifier`` is called
with a list of classifiers, name and domain information (part of
:download:`bagging.py <code/bagging.py>`)::

   class Classifier:
       def __init__(self, **kwds):
           self.__dict__.update(kwds)
   
       def __call__(self, example, resultType = orange.GetValue):
           freq = [0.] * len(self.domain.classVar.values)
           for c in self.classifiers:
               freq[int(c(example))] += 1
           index = freq.index(max(freq))
           value = orange.Value(self.domain.classVar, index)
           for i in range(len(freq)):
               freq[i] = freq[i]/len(self.classifiers)
           if resultType == orange.GetValue: return value
           elif resultType == orange.GetProbabilities: return freq
           else: return (value, freq)
   
For initialization, ``Classifier`` stores all parameters it was
invoked with. When called with a data instance, a list freq is
initialized which is of length equal to the number of classes and
records the number of models that classify an instance to a specific
class. The class that majority of models voted for is returned. While
it may be possible to return classes index, or even a name, by
convention classifiers in Orange return an object ``Value`` instead.

Notice that while, originally, bagging was not intended to compute
probabilities of classes, we compute these as the proportion of models
that voted for a certain class (this is probably incorrect, but
suffice for our example, and does not hurt if only classes values and
not probabilities are used).

Here is the code that tests our bagging we have just implemented. It
compares a decision tree and its bagged variant.  Run it yourself to
see which one is better (:download:`bagging_test.py <code/bagging_test.py>`, uses :download:`bagging.py <code/bagging.py>` and
:download:`adult_sample.tab <code/adult_sample.tab>`)::

   import orange, orngTree, orngEval, bagging
   data = orange.ExampleTable("adult_sample")
   
   tree = orngTree.TreeLearner(mForPrunning=10, minExamples=30)
   tree.name = "tree"
   baggedTree = bagging.Learner(learner=tree, t=5)
   
   learners = [tree, baggedTree]
   
   results = orngEval.crossValidation(learners, data, folds=5)
   for i in range(len(learners)):
       print learners[i].name, orngEval.CA(results)[i]



