"""

.. index: scoring

.. index:: 
   single: multilabel;  scoring for multilabel
   
This module contains various measures of quality for classification and
regression. Most functions require an argument named :obj:`res`, an instance of
:class:`Orange.multilabel.testing.ExperimentResults` as computed by
functions from :mod:`Orange.multilabel.testing` and which contains 
predictions obtained through cross-validation and leave one-out.

======================
Scoring for multilabel
======================

To prepare some data for examples on this page, we shall load the multidata data
set and evaluate Binary Relevance Learner, PPT and BMPLL using cross-validation.

General Measures of Quality
===========================

Multi-label classification requries different metrics than those used in traditional single-label 
classification. This module presents the various methrics that have been proposed in the literature. 
Let :math:`D` be a multi-label evaluation data set, conisting of :math:`|D|` multi-label examples 
:math:`(x_i,Y_i)`, :math:`i=1..|D|`, :math:`Y_i \\subseteq L`. Let :math:`H` be a multi-label classifier 
and :math:`Z_i=H(x_i)` be the set of labels predicted by :math:`H` for example :math:`x_i`.

.. autofunction:: hamming_loss 
.. autofunction:: accuracy
.. autofunction:: precision
.. autofunction:: recall

So, let's compute all this in part of 
(`ml-evaluator.py`_, uses `multidata.tab`_) and print it out:

.. literalinclude:: code/ml-evaluator.py
   :lines: 1-

.. _multidata.tab: code/multidata.tab
.. _ml-evaluator.py: code/ml-evaluator.py

The output should look like this::

    loss= [0.9375]
    accuracy= [0.875]
    precision= [1.0]
    recall= [0.875]

References
==========

Boutell, M.R., Luo, J., Shen, X. & Brown, C.M. (2004), 'Learning multi-label scene classification',
Pattern Recogintion, vol.37, no.9, pp:1757-71

Godbole, S. & Sarawagi, S. (2004), 'Discriminative Methods for Multi-labeled Classification', paper 
presented to Proceedings of the 8th Pacific-Asia Conference on Knowledge Discovery and Data Mining 
(PAKDD 2004)
 
Schapire, R.E. & Singer, Y. (2000), 'Boostexter: a bossting-based system for text categorization', 
Machine Learning, vol.39, no.2/3, pp:135-68.

"""

import statc, operator, math
from operator import add
import numpy

import Orange


#### Private stuff

def log2(x):
    """Calculate logarithm in base 2."""
    return math.log(x)/math.log(2)

def check_non_zero(x):
    """Throw Value Error when x = 0.0."""
    if x==0.0:
        raise ValueError, "Cannot compute the score: no examples or sum of weights is 0.0."

def gettotweight(res):
    """Sum all the weights"""
    totweight = reduce(lambda x, y: x+y.weight, res.results, 0)
    if totweight==0.0:
        raise ValueError, "Cannot compute the score: sum of weights is 0.0."
    return totweight

def gettotsize(res):
    """ Get number of result instances """
    if len(res.results):
        return len(res.results)
    else:
        raise ValueError, "Cannot compute the score: no examples."

def hamming_loss(res, **argkw):
    """
    Schapire and Singer (2000) presented Hamming Loss, which id defined as: 
    
    :math:`HammingLoss(H,D)=\\frac{1}{|D|} \\sum_{i=1}^{|D|} \\frac{Y_i \\vartriangle Z_i}{|L|}`
    """
    losses = [0.0]*res.numberOfLearners
    label_num = len(res.classValues)
    example_num = gettotsize(res)
    
    for e in res.results:
        aclass = e.actualClass
        for i in range(len(e.classes)):
            labels = e.classes[i] 
            if len(labels) <> len(aclass):
                raise ValueError, "The dimensions of the classified output and the actual class array do not match."
            for j in range(label_num):
                if labels[j] == aclass[j]:
                    losses[i] = losses[i]+1
            
    return [x/label_num/example_num for x in losses]

def accuracy(res, forgiveness_rate = 1.0, **argkw):
    """
    Godbole & Sarawagi, 2004 uses the metrics accuracy, precision, recall as follows:
     
    :math:`Accuracy(H,D)=\\frac{1}{|D|} \\sum_{i=1}^{|D|} \\frac{|Y_i \\cap Z_i|}{|Y_i \\cup Z_i|}`
    
    Boutell et al. (2004) give a more generalized version using a parameter :math:`\\alpha \\ge 0`, 
    called forgiveness rate:
    
    :math:`Accuracy(H,D)=\\frac{1}{|D|} \\sum_{i=1}^{|D|} (\\frac{|Y_i \\cap Z_i|}{|Y_i \\cup Z_i|})^{\\alpha}`
    """
    accuracies = [0.0]*res.numberOfLearners
    label_num = len(res.classValues)
    example_num = gettotsize(res)
    
    for e in res.results:
        aclass = e.actualClass
        for i in range(len(e.classes)):
            labels = e.classes[i] 
            if len(labels) <> len(aclass):
                raise ValueError, "The dimensions of the classified output and the actual class array do not match."
            
            intersection = 0.0
            union = 0.0
            for j in range(label_num):
                if labels[j]=='1' and aclass[j]=='1':
                    intersection = intersection+1
                if labels[j]=='1' or aclass[j]=='1':
                    union = union+1
            #print intersection, union
            if union <> 0:
                accuracies[i] = accuracies[i] + intersection/union
            
    return [math.pow(x/example_num,forgiveness_rate) for x in accuracies]

def precision(res, **argkw):
    """
    :math:`Precision(H,D)=\\frac{1}{|D|} \\sum_{i=1}^{|D|} \\frac{|Y_i \\cap Z_i|}{|Z_i|}`
    """
    precisions = [0.0]*res.numberOfLearners
    label_num = len(res.classValues)
    example_num = gettotsize(res)
    
    for e in res.results:
        aclass = e.actualClass
        for i in range(len(e.classes)):
            labels = e.classes[i] 
            if len(labels) <> len(aclass):
                raise ValueError, "The dimensions of the classified output and the actual class array do not match."
            
            intersection = 0.0
            predicted = 0.0
            for j in range(label_num):
                if labels[j]=='1' and aclass[j]=='1':
                    intersection = intersection+1
                if labels[j] == '1':
                    predicted = predicted + 1
            if predicted <> 0:
                precisions[i] = precisions[i] + intersection/predicted
            
    return [x/example_num for x in precisions]

def recall(res, **argkw):
    """
    :math:`Recall(H,D)=\\frac{1}{|D|} \\sum_{i=1}^{|D|} \\frac{|Y_i \\cap Z_i|}{|Y_i|}`
    """
    recalls = [0.0]*res.numberOfLearners
    label_num = len(res.classValues)
    example_num = gettotsize(res)
    
    for e in res.results:
        aclass = e.actualClass
        for i in range(len(e.classes)):
            labels = e.classes[i] 
            if len(labels) <> len(aclass):
                raise ValueError, "The dimensions of the classified output and the actual class array do not match."
            
            intersection = 0.0
            actual = 0.0
            for j in range(label_num):
                if labels[j]=='1' and aclass[j]=='1':
                    intersection = intersection+1
                if aclass[j] == '1':
                    actual = actual + 1
            if actual <> 0:
                recalls[i] = recalls[i] + intersection/actual
            
    return [x/example_num for x in recalls]

def ranking_loss(res, **argkw):
    pass

def average_precision(res, **argkw):
    pass

def hierarchical_loss(res, **argkw):
    pass

