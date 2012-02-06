.. index:: Random classifier

.. index::
   single: classification; Random classifier

**********************************
Random classifier (``random``)
**********************************

Random classifier (class Orange.classification.RandomClassifier) disregards
the examples and returns random predictions. Curious enough though,
the classifier will always predict the same class for the same example.
Predictions can be distributed by the prescribed distribution.

.. class:: Orange.classification.RandomClassifier()

The following example demonstrates a straightforward invocation of
this algorithm:

.. literalinclude:: code/random_classifier.py

The script always prints

.. literalinclude:: code/random_classifier.res

Setting classVar is needed for nicer output. Remove it and see what happens.

Random classifier computes the hash value of example (equivalent to calling
hash(ex), where hash is a Python's built-in function), masks it by 0x7fffffff
and divides it by 0x7fffffff to get a floating point number between 0 and 1.
This value's position in the distribution determines the class. In our example,
random values below 0.5 give the first class, those between 0.5 and 0.8 give
the second and the rest give the third.
