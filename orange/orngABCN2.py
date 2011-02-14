""" This module implements argument based rule learning.
The main learner class is ABCN2. The first few classes are some variants of ABCN2 with reasonable settings.  """


from Orange.classification.rules import DefaultLearner
from Orange.classification.rules import ABCN2
from Orange.classification.rules import ABCN2Ordered
from Orange.classification.rules import ABCN2M
from Orange.classification.rules import ABBeamFilter
from Orange.classification.rules import ruleCoversArguments
from Orange.classification.rules import SelectorAdder
from Orange.classification.rules import ArgFilter
from Orange.classification.rules import SelectorArgConditions
from Orange.classification.rules import CovererAndRemover_Prob
from Orange.classification.rules import avg
from Orange.classification.rules import var
from Orange.classification.rules import perc
from Orange.classification.rules import EVDFitter
from Orange.classification.rules import CrossValidation
from Orange.classification.rules import PILAR
from Orange.classification.rules import CN2UnorderedClassifier
from orngABML import *