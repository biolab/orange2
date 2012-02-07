"""
.. index:: regression tree

.. index::
   single: regression; tree

****************
Regression trees
****************

Regression tree shares its implementation with Orange.classification.tree.TreeLearner,
but uses a different set of functions to evaluate node splitting and stop
criteria. Usage of regression trees is straightforward as demonstrated on the
following example (:download:`regression-tree-run.py <code/regression-tree-run.py>`):

.. literalinclude:: code/regression-tree-run.py
   :lines: 7-

.. autoclass:: TreeLearner
    :members:

.. autoclass:: TreeClassifier
    :members:

=================
SimpleTreeLearner
=================

.. include:: /SimpleTreeLearner.txt

A basic example of using a SimpleTreeLearner is shown below:

.. literalinclude:: code/simple_tree_regression.py

"""


from Orange.core import \
         TreeLearner as _TreeLearner, \
         TreeClassifier as _TreeClassifier, \
         TreeDescender as Descender, \
              TreeDescender_UnknownMergeAsBranchSizes as Descender_UnknownMergeAsBranchSizes, \
              TreeDescender_UnknownMergeAsSelector as Descender_UnknownMergeAsSelector, \
              TreeDescender_UnknownToBranch as Descender_UnknownToBranch, \
              TreeDescender_UnknownToCommonBranch as Descender_UnknownToCommonBranch, \
              TreeDescender_UnknownToCommonSelector as Descender_UnknownToCommonSelector, \
         TreeExampleSplitter as Splitter, \
              TreeExampleSplitter_IgnoreUnknowns as Splitter_IgnoreUnknowns, \
              TreeExampleSplitter_UnknownsAsBranchSizes as Splitter_UnknownsAsBranchSizes, \
              TreeExampleSplitter_UnknownsAsSelector as Splitter_UnknownsAsSelector, \
              TreeExampleSplitter_UnknownsToAll as Splitter_UnknownsToAll, \
              TreeExampleSplitter_UnknownsToBranch as Splitter_UnknownsToBranch, \
              TreeExampleSplitter_UnknownsToCommon as Splitter_UnknownsToCommon, \
              TreeExampleSplitter_UnknownsToRandom as Splitter_UnknownsToRandom, \
         TreeNode as Node, \
         TreeNodeList as NodeList, \
         TreePruner as Pruner, \
              TreePruner_SameMajority as Pruner_SameMajority, \
              TreePruner_m as Pruner_m, \
         TreeSplitConstructor as SplitConstructor, \
              TreeSplitConstructor_Combined as SplitConstructor_Combined, \
              TreeSplitConstructor_Measure as SplitConstructor_Measure, \
                   TreeSplitConstructor_Attribute as SplitConstructor_Feature, \
                   TreeSplitConstructor_ExhaustiveBinary as SplitConstructor_ExhaustiveBinary, \
                   TreeSplitConstructor_OneAgainstOthers as SplitConstructor_OneAgainstOthers, \
                   TreeSplitConstructor_Threshold as SplitConstructor_Threshold, \
         TreeStopCriteria as StopCriteria, \
              TreeStopCriteria_Python as StopCriteria_Python, \
              TreeStopCriteria_common as StopCriteria_common, \
         SimpleTreeLearner, \
         SimpleTreeClassifier

              
from Orange.classification.tree import TreeLearner, TreeClassifier

