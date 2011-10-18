"""

.. index:: classification tree

.. index::
   single: classification; tree

*******************************
Classification trees (``tree``)
*******************************

The following code builds a :obj:`TreeClassifier` on the Iris data set
(with the depth limited to three levels):

.. literalinclude:: code/orngTree1.py
   :lines: 1-4

.. _orngTree1.py: code/orngTree1.py

See `Decision tree learning
<http://en.wikipedia.org/wiki/Decision_tree_learning>`_ on Wikipedia
for introduction to classification trees.

======================
Learner and Classifier
======================

.. autoclass:: TreeLearner
    :members:

.. autoclass:: TreeClassifier
    :members:

.. class:: Node

    Classification trees are a hierarchy of :obj:`Node` classes.

    Node stores the instances belonging to the node, a branch selector,
    a list of branches (if the node is not a leaf) with their descriptions
    and strengths, and a classifier.

    .. attribute:: distribution
    
        A distribution of learning instances.

    .. attribute:: contingency

        Complete contingency matrices for the learning instances.

    .. attribute:: examples, weightID

        Learning instances and the ID of weight meta attribute. The root
        of the tree actually stores all instances, while other nodes
        store only reference to instances in the root node.

    .. attribute:: node_classifier

        A classifier for instances coming to the node. If the node is a
        leaf, it chooses the class (or class distribution) of an instance.

    Internal nodes have additional attributes. The lists :obj:`branches`,
    :obj:`branch_descriptions` and :obj:`branch_sizes` are of the
    same length.

    .. attribute:: branches

        A list of subtrees. Each element is a :obj:`Node` or None.
        If None, the node is empty.

    .. attribute:: branch_descriptions

        A list with strings describing branches. They are constructed
        by :obj:`SplitConstructor`. A string can contain anything,
        for example 'red' or '>12.3'.

    .. attribute:: branch_sizes

        A (weighted) number of training instances for 
        each branch. It can be used, for instance, for modeling
        probabilities when classifying instances with unknown values.

    .. attribute:: branch_selector

        A :obj:`~Orange.classification.Classifier` that returns a branch
        for each instance (as
        :obj:`Orange.data.Value` in ``[0, len(branches)-1]``).  When an
        instance cannot be classified unambiguously, the selector can
        return a discrete distribution, which proposes how to divide
        the instance between the branches. Whether the proposition will
        be used depends upon the :obj:`Splitter` (for learning)
        or :obj:`Descender` (for classification).

    .. method:: tree_size()
        
        Return the number of nodes in the subtrees (including the node,
        excluding null-nodes).

========
Examples
========

Tree Structure
==============

This example works with the lenses data set:

.. literalinclude:: code/treestructure.py
   :lines: 7-10

The following function counts the number of nodes in a tree:

.. _lenses.tab: code/lenses.tab
.. _treestructure.py: code/treestructure.py

.. literalinclude:: code/treestructure.py
   :lines: 12-21

If node is None, the function above return 0. Otherwise, the size is 1
(this node) plus the sizes of all subtrees. The algorithm need to check
if a node is internal (it has a :obj:`~Node.branch_selector`), as leaves
don't have the :obj:`~Node.branches` attribute.

    >>> tree_size(tree_classifier.tree)
    10

Note that a :obj:`Node` already has a built-in method
:func:`~Node.tree_size`.

Trees can be printed with a simple recursive function:

.. literalinclude:: code/treestructure.py
   :lines: 26-41

The crux of the example is not in the formatting (\\n's etc.);
what matters is everything but the print statements. The code
separately handles three node types:

* For null nodes (a node to which no learning instances were classified),
  it just prints "<null node>".
* For internal nodes, it print a node description:
  the feature's name and distribution of classes. :obj:`Node`'s
  branch description is an :obj:`~Orange.classification.Classifier`,
  and its ``class_var`` is the feature whose name is printed.  Class
  distributions are printed as well (they are assumed to be stored).
  The :obj:`printTree0` with a level increased by 1 to increase the
  indent is recursively called for each branch.
* If the node is a leaf, it prints the distribution of learning instances
  in the node and the class to which the instances in the node would
  be classified. We assume that the :obj:`~Node.node_classifier` is a
  :obj:`DefaultClassifier`. A better print function should be aware of
  possible alternatives.

The wrapper function that accepts either a
:obj:`TreeClassifier` or a :obj:`Node` can be written as follows:

.. literalinclude:: code/treestructure.py
   :lines: 43-49

It's straightforward: if ``x`` is a
:obj:`TreeClassifier`, it prints ``x.tree``; if it's :obj:`Node` it
print ``x``. If it's of some other type,
an exception is raised. The output::

    >>> print_tree(tree_classifier)
    tear_rate (<15.000, 5.000, 4.000>)
    : reduced --> none (<12.000, 0.000, 0.000>)
    : normal
       astigmatic (<3.000, 5.000, 4.000>)
       : no
          age (<1.000, 5.000, 0.000>)
          : young --> soft (<0.000, 2.000, 0.000>)
          : pre-presbyopic --> soft (<0.000, 2.000, 0.000>)
          : presbyopic --> none (<1.000, 1.000, 0.000>)
       : yes
          prescription (<2.000, 0.000, 4.000>)
          : myope --> hard (<0.000, 0.000, 3.000>)
          : hypermetrope --> none (<2.000, 0.000, 1.000>)

The tree structure examples conclude with a simple pruning function,
written entirely in Python and unrelated to any :class:`Pruner`. It limits
the tree depth (the number of internal nodes on any path down the tree).
For example, to get a two-level tree, call cut_tree(root, 2). The function
is recursive, with the second argument (level) decreasing at each call;
when zero, the current node will be made a leaf:

.. literalinclude:: code/treestructure.py
   :lines: 54-62

The function acts only when :obj:`node` and :obj:`node.branch_selector`
are defined. If the level is not zero, is recursively calls  the function
for each branch. Otherwise, it clears the selector, branches and branch
descriptions.

    >>> cutTree(tree.tree, 2)
    >>> printTree(tree)
    tear_rate (<15.000, 5.000, 4.000>)
    : reduced --> none (<12.000, 0.000, 0.000>)
    : normal
       astigmatic (<3.000, 5.000, 4.000>)
       : no --> soft (<1.000, 5.000, 0.000>)
       : yes --> hard (<2.000, 0.000, 4.000>)

Setting learning parameters
===========================

Let us construct a :obj:`TreeLearner` to play with:

.. literalinclude:: code/treelearner.py
   :lines: 7-10

There are three crucial components in learning: the
:obj:`~TreeLearner.split` and :obj:`~TreeLearner.stop` criteria, and the
example :obj:`~TreeLearner.splitter`. The default ``stop`` is set with:

    >>> learner.stop = Orange.classification.tree.StopCriteria_common()

The default stopping parameters are:

    >>> print learner.stop.max_majority, learner.stop.min_examples
    1.0 0.0

The defaults only stop splitting when no instances are left or all of
them are in the same class.

If the minimal subset that is allowed to be split further is set to five
instances, the resulting tree is smaller.

    >>> learner.stop.min_instances = 5.0
    >>> tree = learner(data)
    >>> print tree.dump()
    tear_rate=reduced: none (100.00%)
    tear_rate=normal
    |    astigmatic=no
    |    |    age=pre-presbyopic: soft (100.00%)
    |    |    age=presbyopic: none (50.00%)
    |    |    age=young: soft (100.00%)
    |    astigmatic=yes
    |    |    prescription=hypermetrope: none (66.67%)
    |    |    prescription=myope: hard (100.00%)

We can also limit the maximal proportion of majority class.

    >>> learner.stop.max_majority = 0.5
    >>> tree = learner(data)
    >>> print tree.dump()
    none (62.50%)

Redefining tree induction components
====================================

This example shows how to use a custom stop function.  First, the
``def_stop`` function defines the default stop function. The first tree
has some added randomness; the induction also stops in 20% of the
cases when ``def_stop`` returns False. The stopping criteria for the
second tree is completely random: it stops induction in 20% of cases.
Note that in the second case lambda function still has three parameters,
even though in does not need any, since so many are necessary
for :obj:`~TreeLearner.stop`.

.. _tree3.py: code/tree3.py

.. literalinclude:: code/tree3.py
   :lines: 8-23

=================================
Learner and Classifier Components
=================================

Split constructors
=====================

.. class:: SplitConstructor

    Decide how to divide learning instances.
    
    The :obj:`SplitConstructor` should use the domain
    contingency when possible, both for speed and adaptability
    (:obj:`TreeLearner.contingency`).  Sometimes domain contingency does
    not suffice, for example if ReliefF score is used.

    A :obj:`SplitConstructor` can veto further tree induction by returning
    no classifier. This is generally related to the number of learning
    instances that would go in each branch. If there are no splits with
    more than :obj:`SplitConstructor.min_subset` instances in the branches
    (null nodes are allowed), the induction is stopped.

    Split constructors that cannot handle a particular feature
    type (discrete, continuous) quietly skip them. When in doubt, use
    :obj:`SplitConstructor_Combined`, which delegates features to
    specialized split constructors.

    The same split constructors can be used both for classification and
    regression, if the chosen score (for :obj:`SplitConstructor_Score`
    and derived classes) supports both.

    .. attribute:: min_subset

        The minimal (weighted) number in non-null leaves.

    .. method:: __call__(data, [ weightID, contingency, apriori_distribution, candidates, clsfr]) 

        :param data: in any acceptable form.
        :param weightID: Optional; the default of 0 means that all
            instances have a weight of 1.0. 
        :param contingency: a domain contingency
        :param apriori_distribution: apriori class probabilities.
        :type apriori_distribution: :obj:`Orange.statistics.distribution.Distribution`
        :param candidates: only consider these 
            features (one boolean for each feature).
        :param clsfr: a node classifier (if it was constructed, that is, 
            if :obj:`store_node_classifier` is True) 

        Construct a split. Return a tuple (:obj:`branch_selector`,
        :obj:`branch_descriptions` (a list), :obj:`subset_sizes`
        (the number of instances for each branch, may also be
        empty), :obj:`quality` (higher numbers mean better splits),
        :obj:`spent_feature`). If no split is constructed,
        the :obj:`selector`, :obj:`branch_descriptions` and
        :obj:`subset_sizes` are None, while :obj:`quality` is 0.0 and
        :obj:`spent_feature` is -1.

        If the chosen feature will be useless in the future and
        should not be considered for splitting in any of the subtrees
        (typically, when discrete features are used as-they-are, without
        any binarization or subsetting), then it should return the index
        of this feature as :obj:`spent_feature`. If no features are spent,
        :obj:`spent_feature` is -1.

.. class:: SplitConstructor_Score

    Bases: :class:`SplitConstructor`

    An abstract base class that compare splits
    with a :class:`Orange.feature.scoring.Score`.  All split
    constructors except for :obj:`SplitConstructor_Combined` are derived
    from this class.

    .. attribute:: measure

        A :class:`Orange.feature.scoring.Score` for split evaluation. It
        has to handle the class type - for example, you cannot use
        :class:`~Orange.feature.scoring.GainRatio` for regression or
        :class:`~Orange.feature.scoring.MSE` for classification.

    .. attribute:: worst_acceptable

        The lowest allowed split quality.  The value strongly depends
        on chosen :obj:`measure` component. Default is 0.0.

.. class:: SplitConstructor_Feature

    Bases: :class:`SplitConstructor_Score`

    Each value of a discrete feature corresponds to a branch.  The feature
    with the highest score (:obj:`~Measure.measure`) is selected. When
    tied, a random feature is selected.

    The constructed :obj:`branch_selector` is an instance of
    :obj:`orange.ClassifierFromVarFD`, that returns a value of the selected
    feature. :obj:`branch_description` contains the feature's
    values. The feature is marked as spent (it cannot reappear in the
    node's subtrees).

.. class:: SplitConstructor_ExhaustiveBinary

    Bases: :class:`SplitConstructor_Score`

    Finds the binarization with the highest score among all features. In
    case of ties, a random feature is selected.

    The constructed :obj:`branch_selector` is an instance
    :obj:`orange.ClassifierFromVarFD`, that returns a value of the
    selected feature. Its :obj:`transformer` contains a ``MapIntValue``
    that maps values of the feature into a binary feature. Branches
    with a single feature value are described with that value and
    branches with more than one are described with ``[<val1>, <val2>,
    ...<valn>]``. Only binary features are marked as spent.

.. class:: SplitConstructor_Threshold

    Bases: :class:`SplitConstructor_Score`

    The only split constructor for continuous features. It divides the
    range of feature values with a threshold that maximizes the split's
    quality. In case of ties, a random feature is selected.  The feature
    that yields the best binary split is returned.

    The constructed :obj:`branch_selector` is an instance of
    :obj:`orange.ClassifierFromVarFD` with an attached :obj:`transformer`,
    of type :obj:`Orange.feature.discretization.ThresholdDiscretizer`. The
    branch descriptions are "<threshold" and ">=threshold". The feature
    is not spent.

.. class:: SplitConstructor_OneAgainstOthers
    
    Bases: :class:`SplitConstructor_Score`

    Undocumented.

.. class:: SplitConstructor_Combined

    Bases: :class:`SplitConstructor`

    Uses different split constructors for discrete and continuous
    features. Each split constructor is called with appropriate
    features. Both construct a candidate for a split; the better of them
    is used.

    The choice of the split is not probabilistically fair, when
    multiple candidates have the same score. For example, if there
    are nine discrete features with the highest score the split
    constructor for discrete features will select one of them. Now,
    if there is also a single continuous feature with the same score,
    :obj:`SplitConstructor_Combined` would randomly select between the
    proposed discrete feature and continuous feature, unaware that the
    discrete feature  has already competed with eight others.  So,
    the probability for selecting (each) discrete feature would be
    1/18 instead of 1/10. Although incorrect, this should not affect
    the performance.

    .. attribute: discrete_split_constructor

        Split constructor for discrete features; 
        for instance, :obj:`SplitConstructor_Feature` or
        :obj:`SplitConstructor_ExhaustiveBinary`.

    .. attribute: continuous_split_constructor

        Split constructor for continuous features; it 
        can be either :obj:`SplitConstructor_Threshold` or a 
        a custom split constructor.


StopCriteria and StopCriteria_common
============================================

:obj:`StopCriteria` determines when to stop the induction of subtrees. 

.. class:: StopCriteria

    Provides the basic stopping criteria: the tree induction stops
    when there is at most one instance left (the actual, not weighted,
    number). The induction also stops when all instances are in the
    same class (for discrete problems) or have the same outcome value
    (for regression problems).

    .. method:: __call__(instances[, weightID, domain contingencies])

        Return True (stop) of False (continue the induction).
        Contingencies are not used for counting. Derived classes should
        use the contingencies whenever possible.

.. class:: StopCriteria_common

    Pre-pruning with additional criteria.

    .. attribute:: max_majority

        Maximum proportion of majority class. When exceeded,
        induction stops.

    .. attribute:: min_instances

        Minimum number of instances for splitting. Subsets with less
        than :obj:`min_instances` instances are not split further.
        The sample count is weighed.


Splitters
=================

Splitters sort learning instances into branches (the branches are selected
with a :obj:`SplitConstructor`, while a :obj:`Descender` decides the
branch for an instance during classification.

Most splitters call :obj:`Node.branch_selector` and assign
instances correspondingly. When the value is unknown they choose a
particular branch or skip the instance.

Some splitters can also split instances: a weighed instance is 
used in more than than one subset. Each branch has a weight ID (usually,
each its own ID) and all instances in that branch should have this meta attribute. 

An instance that 
hasn't been split has only one additional attribute (weight
ID corresponding to the subset to which it went). Instance that is split
between, say, three subsets, has three new meta attributes, one for each
subset. The weights are used only when needed; when there is no
splitting - no weight IDs are returned.

.. class:: Splitter

    An abstract base class that splits instances
    into subsets.

    .. method:: __call__(node, instances[, weightID])

        :param node: a node.
        :type node: :obj:`Node`
        :param instances: a set of instances
        :param weightID: weight ID. 
        
        Use the information in :obj:`Node` (particularly the
        :obj:`~Node.branch_selector`) to split the given set of instances into
        subsets.  Return a tuple with a list of instance subsets and
        a list of weights.  The list of weights is either a
        list of integers or None when no weights are added.

.. class:: Splitter_IgnoreUnknowns

    Bases: :class:`Splitter`

    Ignores the instances for which no single branch can be determined.

.. class:: Splitter_UnknownsToCommon

    Bases: :class:`Splitter`

    Places all ambiguous instances to a branch with the highest number of
    instances. If there is more than one such branch, one is selected at
    random and then used for all instances.

.. class:: Splitter_UnknownsToAll

    Bases: :class:`Splitter`

    Splits instances with an unknown value of the feature into all branches.

.. class:: Splitter_UnknownsToRandom

    Bases: :class:`Splitter`

    Selects a random branch for ambiguous instances.

.. class:: Splitter_UnknownsToBranch

    Bases: :class:`Splitter`

    Constructs an additional branch for ambiguous instances. 
    The branch's description is "unknown".

.. class:: Splitter_UnknownsAsBranchSizes

    Bases: :class:`Splitter`

    Splits instances with unknown value of the feature according to
    proportions of instances in each branch.

.. class:: Splitter_UnknownsAsSelector

    Bases: :class:`Splitter`

    Splits instances with unknown value of the feature according to
    distribution proposed by selector (usually the same as proportions
    of instances in branches).

Descenders
=============================

Descenders decide where should the instances that cannot be unambiguously
put in a single branch go during classification (the branches are selected
with a :obj:`SplitConstructor`, while a :obj:`Splitter` sorts instances
during learning).

.. class:: Descender

    An abstract base tree descender. It descends a
    an instance as deep as possible, according to the values
    of instance's features. The :obj:`Descender`: calls the node's
    :obj:`~Node.branch_selector` to get the branch index. If it's a
    simple index, the corresponding branch is followed. If not, the
    descender decides what to do. A descender can choose a single
    branch (for instance, the one that is the most recommended by the
    :obj:`~Node.branch_selector`) or it can let the branches vote.

    Three are possible outcomes of a descent:

    #. The descender reaches a leaf. This happens when
       there were no unknown or out-of-range values, or when the
       descender selected a single branch and continued the descend
       despite them. The descender returns the :obj:`Node` it has reached.
    #. Node's :obj:`~Node.branch_selector` returned a distribution and
       :obj:`Descender` decided to stop the descend at this (internal)
       node. It returns the current :obj:`Node`.
    #. Node's :obj:`~Node.branch_selector` returned a distribution and the
       :obj:`Node` wants to split the instance (i.e., to decide the class
       by voting). It returns a :obj:`Node` and the vote-weights for
       the branches.  The weights can correspond, for example,  to the
       distribution returned by node's :obj:`~Node.branch_selector`, or to
       the number of learning instances that were assigned to each branch.

    .. method:: __call__(node, instance)

        Descends until it reaches a leaf or a node in
        which a vote of subtrees is required. A tuple
        of two elements is returned. If it reached a leaf, the tuple contains
        the leaf node and None. If not, it contains a node and
        a list of floats (weights of votes).

.. class:: Descender_UnknownToNode

    Bases: :obj:`Descender`

    When instance cannot be classified into a single branch, the current
    node is returned. Thus, the node's :obj:`~Node.node_classifier`
    will be used to make a decision. Therefore, internal nodes
    need to have :obj:`Node.node_classifier` defined.

.. class:: Descender_UnknownToBranch

    Bases: :obj:`Descender`

    Classifies instances with unknown value to a special branch. This
    makes sense only if the tree itself was constructed with
    :obj:`Splitter_UnknownsToBranch`.

.. class:: Descender_UnknownToCommonBranch

    Bases: :obj:`Descender`

    Classifies instances with unknown values to the branch with the
    highest number of instances. If there is more than one such branch,
    random branch is chosen for each instance.

.. class:: Descender_UnknownToCommonSelector

    Bases: :obj:`Descender`

    Classifies instances with unknown values to the branch which received
    the highest recommendation by the selector.

.. class:: Descender_UnknownMergeAsBranchSizes

    Bases: :obj:`Descender`

    The subtrees vote for the instance's class; the vote is weighted
    according to the sizes of the branches.

.. class:: Descender_UnknownMergeAsSelector

    Bases: :obj:`Descender`

    The subtrees vote for the instance's class; the vote is weighted
    according to the selectors proposal.

Pruning
=======

.. index::
    pair: classification trees; pruning

The pruners construct a shallow copy of a tree. The pruned tree's
:obj:`Node` contain references to the same contingency matrices,
node classifiers, branch selectors, ...  as the original tree.

Pruners cannot construct a new :obj:`~Node.node_classifier`.  Thus, for
pruning, internal nodes must have :obj:`~Node.node_classifier` defined
(the default).

.. class:: Pruner

    An abstract base tree pruner.

    .. method:: __call__(tree)

        :param tree: either
            a :obj:`Node` or (the C++ version of the classifier,
            saved in :obj:`TreeClassfier.base_classifier`).

        The resulting pruned tree is of the same type as the argument.
        The original tree remains intact.

.. class:: Pruner_SameMajority

    Bases: :class:`Pruner`

    A tree can have a subtrees where all the leaves have
    the same majority class. This is allowed because leaves can still
    have different class distributions and thus predict different
    probabilities.  The :obj:`Pruner_SameMajority` prunes the tree so
    that there is no subtree in which all the nodes would have the same
    majority class.

    This pruner will only prune the nodes in which the node classifier
    is a :obj:`~Orange.classification.majority.ConstantClassifier`
    (or a derived class).

    The pruning works from leaves to the root.
    It siblings have (at least one) common majority class, they can be pruned.

.. class:: Pruner_m

    Bases: :class:`Pruner`

    Prunes a tree by comparing m-estimates of static and dynamic 
    error as defined in (Bratko, 2002).

    .. attribute:: m

        Parameter m for m-estimation.

Printing the tree
=================

The tree printing functions are very flexible. They can print, for
example, numbers of instances, proportions of majority class in nodes
and similar, or more complex statistics like the proportion of instances
in a particular class divided by the proportion of instances of this
class in a parent node. Users may also pass their own functions to print
certain elements.

The easiest way to print the tree is to call :func:`TreeClassifier.dump`
without any arguments::

    >>> print tree.dump()
    petal width<0.800: Iris-setosa (100.00%)
    petal width>=0.800
    |    petal width<1.750
    |    |    petal length<5.350: Iris-versicolor (94.23%)
    |    |    petal length>=5.350: Iris-virginica (100.00%)
    |    petal width>=1.750
    |    |    petal length<4.850: Iris-virginica (66.67%)
    |    |    petal length>=4.850: Iris-virginica (100.00%)


Format string
-------------

Format strings are printed at every leaf or internal node with the certain
format specifiers replaced by data from the tree node. Specifiers are
generally of form **%[^]<precision><quantity><divisor>**.

**^** at the start tells that the number should be multiplied by 100,
which is useful for proportions like percentages.

**<precision>** is in the same format as in Python (or C) string
formatting. For instance, ``%N`` denotes the number of instances in
the node, hence ``%6.2N`` would mean output to two decimal digits
and six places altogether. If left out, a default format ``5.3`` is
used, unless the numbers are multiplied by 100, in which case the default
is ``.0`` (no decimals, the number is rounded to the nearest integer).

**<divisor>** tells what to divide the quantity in that node with.
``bP`` means division by the same quantity in the parent node; for instance,
``%NbP`` gives the number of instances in the node divided by the
number of instances in parent node. Precision formatting can be added,
e.g. ``%6.2NbP``. ``bA`` denotes division by the same quantity over the entire
data set, so ``%NbA`` will tell you the proportion of instances (out
of the entire training data set) that fell into that node. If division is
impossible since the parent node does not exist or some data is missing,
a dot is printed out instead.

**<quantity>** defines what to print and is the only required element. 
It can be:

``V``
    The predicted value at that node. Precision 
    or divisor can not be defined here.

``N``
    The number of instances in the node.

``M``
    The number of instances in the majority class (that is, the class 
    predicted by the node).

``m``
    The proportion of instances in the majority class.

``A``
    The average class for instances the node; this is available only for 
    regression trees.

``E``
    Standard error for class of instances in the node; available only for
    regression trees.

``I``
    Print out the confidence interval. The modifier is used as 
    ``%I(95)`` of (more complicated) ``%5.3I(95)bP``.

``C``
    The number of instances in the given class.  For a classification
    example, ``%5.3C="Iris-virginica"bP`` denotes the number of instances
    of Iris-virginica by the number of instances this class in the parent
    node ( instances that are *not* Iris-virginica could be described with
    ``%5.3CbP!="Iris-virginica"``).

    For regression trees, use operators =, !=, <, <=, >, and >=, as in
    ``%C<22``, with optional precision and divisor. Intervals are also
    possible: ``%C[20, 22]`` gives the number of instances between
    20 and 22 (inclusive) and ``%C(20, 22)`` gives the number of such
    instances excluding the boundaries. Mixing of parentheses is allowed,
    e.g. ``%C(20, 22]``.  Add ``!`` for instances outside the interval,
    like ``%C!(20, 22]``.

``c``
    Same as ``C``, except that it computes the proportion of the class
    instead of the number of instances.

``D``
    The number of instances in each class. Precision and the divisor
    are applied to each number in the distribution.  This quantity can
    not be computed for regression trees.

``d``
    Same as ``D``, except that it shows proportions of instances.

<user defined formats>
    Instructions and examples of added formats are at the end of this
    section.

.. rubric:: Examples on classification trees

A tree on the iris data set with the depth limited to three
levels is built as follows:
    
.. literalinclude:: code/orngTree1.py
   :lines: 1-4

.. _orngTree1.py: code/orngTree1.py

Printing the predicted class at each node, the number
of instances in the majority class with the total number of instances in
the node requires a custom format string::

    >>> print tree.dump(leaf_str="%V (%M out of %N)")
    petal width<0.800: Iris-setosa (50.000 out of 50.000)
    petal width>=0.800
    |    petal width<1.750
    |    |    petal length<5.350: Iris-versicolor (49.000 out of 52.000)
    |    |    petal length>=5.350: Iris-virginica (2.000 out of 2.000)
    |    petal width>=1.750
    |    |    petal length<4.850: Iris-virginica (2.000 out of 3.000)
    |    |    petal length>=4.850: Iris-virginica (43.000 out of 43.000)

The number of instances as
compared to the entire data set and to the parent node::

    >>> print tree.dump(leaf_str="%V (%^MbA%, %^MbP%)")
    petal width<0.800: Iris-setosa (100%, 100%)
    petal width>=0.800
    |    petal width<1.750
    |    |    petal length<5.350: Iris-versicolor (98%, 100%)
    |    |    petal length>=5.350: Iris-virginica (4%, 40%)
    |    petal width>=1.750
    |    |    petal length<4.850: Iris-virginica (4%, 4%)
    |    |    petal length>=4.850: Iris-virginica (86%, 96%)

``%M`` is the number of instances in the majority class. Dividing by
the number of all instances from this class on the entire data set
is described with ``%MbA``. Add ``^`` in front for mutiplication with
100. The percent sign *after* that is printed out literally, just as the
comma and parentheses. For the proportion of this class in the parent the
``bA`` is replaced with ``bA``.

To print the number of versicolors in each node, together with the
proportion of versicolors among the instances in this particular node
and among all versicolors, use the following::

    '%C="Iris-versicolor" (%^c="Iris-versicolor"% of node, %^CbA="Iris-versicolor"% of versicolors)

It gives::

    petal width<0.800: 0.000 (0% of node, 0% of versicolors)
    petal width>=0.800
    |    petal width<1.750
    |    |    petal length<5.350: 49.000 (94% of node, 98% of versicolors)
    |    |    petal length>=5.350: 0.000 (0% of node, 0% of versicolors)
    |    petal width>=1.750
    |    |    petal length<4.850: 1.000 (33% of node, 2% of versicolors)
    |    |    petal length>=4.850: 0.000 (0% of node, 0% of versicolors)

Finally, to print the distributions using a format string ``%D``::

    petal width<0.800: [50.000, 0.000, 0.000]
    petal width>=0.800
    |    petal width<1.750
    |    |    petal length<5.350: [0.000, 49.000, 3.000]
    |    |    petal length>=5.350: [0.000, 0.000, 2.000]
    |    petal width>=1.750
    |    |    petal length<4.850: [0.000, 1.000, 2.000]
    |    |    petal length>=4.850: [0.000, 0.000, 43.000]

As the order of classes is the same as in ``data.domain.class_var.values``
(setosa, versicolor, virginica), there are 49 versicolors and 3 virginicae
in the node at ``petal length<5.350``. To print the proportions within
nodes rounded to two decimals use ``%.2d``::

    petal width<0.800: [1.00, 0.00, 0.00]
    petal width>=0.800
    |    petal width<1.750
    |    |    petal length<5.350: [0.00, 0.94, 0.06]
    |    |    petal length>=5.350: [0.00, 0.00, 1.00]
    |    petal width>=1.750
    |    |    petal length<4.850: [0.00, 0.33, 0.67]
    |    |    petal length>=4.850: [0.00, 0.00, 1.00]

The most trivial format string for internal nodes is for printing
node predictions. ``.`` in the following example specifies
that the node_str should be the same as leaf_str.

::

    tree.dump(leaf_str="%V", node_str=".")
 
The output::

    root: Iris-setosa
    |    petal width<0.800: Iris-setosa
    |    petal width>=0.800: Iris-versicolor
    |    |    petal width<1.750: Iris-versicolor
    |    |    |    petal length<5.350: Iris-versicolor
    |    |    |    petal length>=5.350: Iris-virginica
    |    |    petal width>=1.750: Iris-virginica
    |    |    |    petal length<4.850: Iris-virginica
    |    |    |    petal length>=4.850: Iris-virginica

A node *root* has appeared and the tree looks one level
deeper. This is needed to also print the data for tree root.

To observe how the number
of virginicas decreases down the tree try::

    print tree.dump(leaf_str='%^.1CbA="Iris-virginica"% (%^.1CbP="Iris-virginica"%)', node_str='.')

Interpretation: ``CbA="Iris-virginica"`` is 
the number of instances from virginica, divided by the total number
of instances in this class. Add ``^.1`` and the result will be
multiplied and printed with one decimal. The trailing ``%`` is printed
out. In parentheses the same thing was divided by
the instances in the parent node. The single quotes were used for strings, so
that double quotes inside the string can specify the class.

::

    root: 100.0% (.%)
    |    petal width<0.800: 0.0% (0.0%)
    |    petal width>=0.800: 100.0% (100.0%)
    |    |    petal width<1.750: 10.0% (10.0%)
    |    |    |    petal length<5.350: 6.0% (60.0%)
    |    |    |    petal length>=5.350: 4.0% (40.0%)
    |    |    petal width>=1.750: 90.0% (90.0%)
    |    |    |    petal length<4.850: 4.0% (4.4%)
    |    |    |    petal length>=4.850: 86.0% (95.6%)

If :meth:`~TreeClassifier.dump` cannot compute something, in this case
because the root has no parent, it prints out a dot.

The final example with classification trees prints the distributions in
nodes, the distribution compared to the parent, the proportions compared
to the parent and the predicted class in the leaves::

    >>> print tree.dump(leaf_str='"%V   %D %.2DbP %.2dbP"', node_str='"%D %.2DbP %.2dbP"')
    root: [50.000, 50.000, 50.000] . .
    |    petal width<0.800: [50.000, 0.000, 0.000] [1.00, 0.00, 0.00] [3.00, 0.00, 0.00]:
    |        Iris-setosa   [50.000, 0.000, 0.000] [1.00, 0.00, 0.00] [3.00, 0.00, 0.00]
    |    petal width>=0.800: [0.000, 50.000, 50.000] [0.00, 1.00, 1.00] [0.00, 1.50, 1.50]
    |    |    petal width<1.750: [0.000, 49.000, 5.000] [0.00, 0.98, 0.10] [0.00, 1.81, 0.19]
    |    |    |    petal length<5.350: [0.000, 49.000, 3.000] [0.00, 1.00, 0.60] [0.00, 1.04, 0.62]:
    |    |    |        Iris-versicolor   [0.000, 49.000, 3.000] [0.00, 1.00, 0.60] [0.00, 1.04, 0.62]
    |    |    |    petal length>=5.350: [0.000, 0.000, 2.000] [0.00, 0.00, 0.40] [0.00, 0.00, 10.80]:
    |    |    |        Iris-virginica   [0.000, 0.000, 2.000] [0.00, 0.00, 0.40] [0.00, 0.00, 10.80]
    |    |    petal width>=1.750: [0.000, 1.000, 45.000] [0.00, 0.02, 0.90] [0.00, 0.04, 1.96]
    |    |    |    petal length<4.850: [0.000, 1.000, 2.000] [0.00, 1.00, 0.04] [0.00, 15.33, 0.68]:
    |    |    |        Iris-virginica   [0.000, 1.000, 2.000] [0.00, 1.00, 0.04] [0.00, 15.33, 0.68]
    |    |    |    petal length>=4.850: [0.000, 0.000, 43.000] [0.00, 0.00, 0.96] [0.00, 0.00, 1.02]:
    |    |    |        Iris-virginica   [0.000, 0.000, 43.000] [0.00, 0.00, 0.96] [0.00, 0.00, 1.02]


.. rubric:: Examples on regression trees

The regression trees examples use a tree induced from the housing data
set. Without other argumets, :meth:`TreeClassifier.dump` prints the
following::

    RM<6.941
    |    LSTAT<14.400
    |    |    DIS<1.385: 45.6
    |    |    DIS>=1.385: 22.9
    |    LSTAT>=14.400
    |    |    CRIM<6.992: 17.1
    |    |    CRIM>=6.992: 12.0
    RM>=6.941
    |    RM<7.437
    |    |    CRIM<7.393: 33.3
    |    |    CRIM>=7.393: 14.4
    |    RM>=7.437
    |    |    TAX<534.500: 45.9
    |    |    TAX>=534.500: 21.9

To add the standard error in both internal nodes and leaves, and
the 90% confidence intervals in the leaves, use::

    >>> print tree.dump(leaf_str="[SE: %E]\t %V %I(90)", node_str="[SE: %E]")
    root: [SE: 0.409]
    |    RM<6.941: [SE: 0.306]
    |    |    LSTAT<14.400: [SE: 0.320]
    |    |    |    DIS<1.385: [SE: 4.420]:
    |    |    |        [SE: 4.420]   45.6 [38.331-52.829]
    |    |    |    DIS>=1.385: [SE: 0.244]:
    |    |    |        [SE: 0.244]   22.9 [22.504-23.306]
    |    |    LSTAT>=14.400: [SE: 0.333]
    |    |    |    CRIM<6.992: [SE: 0.338]:
    |    |    |        [SE: 0.338]   17.1 [16.584-17.691]
    |    |    |    CRIM>=6.992: [SE: 0.448]:
    |    |    |        [SE: 0.448]   12.0 [11.243-12.714]
    |    RM>=6.941: [SE: 1.031]
    |    |    RM<7.437: [SE: 0.958]
    |    |    |    CRIM<7.393: [SE: 0.692]:
    |    |    |        [SE: 0.692]   33.3 [32.214-34.484]
    |    |    |    CRIM>=7.393: [SE: 2.157]:
    |    |    |        [SE: 2.157]   14.4 [10.862-17.938]
    |    |    RM>=7.437: [SE: 1.124]
    |    |    |    TAX<534.500: [SE: 0.817]:
    |    |    |        [SE: 0.817]   45.9 [44.556-47.237]
    |    |    |    TAX>=534.500: [SE: 0.000]:
    |    |    |        [SE: 0.000]   21.9 [21.900-21.900]

The predicted value (``%V``) and the average (``%A``) may differ because
a regression tree does not always predict the leaf average, but whatever
the :obj:`~Node.node_classifier` in a leaf returns.  As ``%V`` uses the
:obj:`Orange.data.variable.Continuous`' function for printing the
value, the number has the same number of decimals as in the data file.

Regression trees cannot print the distributions in the same way
as classification trees. They instead offer a set of operators for
observing the number of instances within a certain range. For instance,
to print the number of instances with values below 22 and compare
it with values in the parent nodes use::

    >>> print tree.dump(leaf_str="%C<22 (%cbP<22)", node_str=".")
    root: 277.000 (.)
    |    RM<6.941: 273.000 (1.160)
    |    |    LSTAT<14.400: 107.000 (0.661)
    |    |    |    DIS<1.385: 0.000 (0.000)
    |    |    |    DIS>=1.385: 107.000 (1.020)
    |    |    LSTAT>=14.400: 166.000 (1.494)
    |    |    |    CRIM<6.992: 93.000 (0.971)
    |    |    |    CRIM>=6.992: 73.000 (1.040)
    |    RM>=6.941: 4.000 (0.096)
    |    |    RM<7.437: 3.000 (1.239)
    |    |    |    CRIM<7.393: 0.000 (0.000)
    |    |    |    CRIM>=7.393: 3.000 (15.333)
    |    |    RM>=7.437: 1.000 (0.633)
    |    |    |    TAX<534.500: 0.000 (0.000)
    |    |    |    TAX>=534.500: 1.000 (30.000)</xmp>

The last line, for instance, says the the number of instances with the
class below 22 is among those with tax above 534 is 30 times higher than
the number of such instances in its parent node.

To count the same for all instances *outside*
interval [20, 22] and print out the proportions as percents use::

    >>> print tree.dump(leaf_str="%C![20,22] (%^cbP![20,22]%)", node_str=".")

The format string  ``%c![20, 22]`` denotes the proportion of instances
(within the node) whose values are below 20 or above 22. ``%cbP![20,
22]`` derives same statistics computed on the parent. A ``^`` is added
for percentages.

::

    root: 439.000 (.%)
    |    RM<6.941: 364.000 (98%)
    |    |    LSTAT<14.400: 200.000 (93%)
    |    |    |    DIS<1.385: 5.000 (127%)
    |    |    |    DIS>=1.385: 195.000 (99%)
    |    |    LSTAT>=14.400: 164.000 (111%)
    |    |    |    CRIM<6.992: 91.000 (96%)
    |    |    |    CRIM>=6.992: 73.000 (105%)
    |    RM>=6.941: 75.000 (114%)
    |    |    RM<7.437: 46.000 (101%)
    |    |    |    CRIM<7.393: 43.000 (100%)
    |    |    |    CRIM>=7.393: 3.000 (100%)
    |    |    RM>=7.437: 29.000 (98%)
    |    |    |    TAX<534.500: 29.000 (103%)
    |    |    |    TAX>=534.500: 0.000 (0%)


Defining custom printouts
-------------------------

:meth:`TreeClassifier.dump`'s argument :obj:`user_formats` can be used to
print other information.  :obj:`~TreeClassifier.dump.user_formats` should
contain a list of tuples with a regular expression and a function to be
called when that expression is found in the format string. Expressions
from :obj:`user_formats` are checked before the built-in expressions
discussed above.

The regular expression should describe a string like used above,
for instance ``%.2DbP``. When a leaf or internal node
is printed, the format string (:obj:`leaf_str` or :obj:`node_str`)
is checked for these regular expressions and when the match is found,
the corresponding callback function is called.

The passed function will get five arguments: the format string 
(:obj:`leaf_str` or :obj:`node_str`), the match object, the node which is
being printed, its parent (can be None) and the tree classifier.
The function should return the format string in which the part described
by the match object (that is, the part that is matched by the regular
expression) is replaced by whatever information your callback function
is supposed to give.

The function can use several utility function provided in the module.

.. autofunction:: insert_str

.. autofunction:: insert_dot

.. autofunction:: insert_num

.. autofunction:: by_whom

The module also includes reusable regular expressions: 

.. autodata:: fs

.. autodata:: by

For a trivial example, ``%V`` is implemented with the
following tuple::

    (re.compile("%V"), replaceV)

And ``replaceV`` is defined by::

    def replaceV(strg, mo, node, parent, tree):
        return insert_str(strg, mo, str(node.node_classifier.default_value))

``replaceV`` takes the value predicted at the node
(``node.node_classifier.default_value`` ), converts it to a string
and passes it to :func:`insert_str`.

A more complex regular expression is the one for the proportion of
majority class, defined as ``"%"+fs+"M"+by``. It uses the two partial
expressions defined above (:obj:`fs` and :obj:`by`).

The following code prints the classification margin for each node,
that is, the difference between the proportion of the largest and the
second largest class in the node:

.. _orngTree2.py: code/orngTree2.py

.. literalinclude:: code/orngTree2.py
   :lines: 7-31

``get_margin`` computes the margin from the distribution. The replacing
function, ``replaceB``, computes the margin for the node.  If :data:`by`
group is present, we call :func:`by_whom` to get the node with whose
margin this node's margin is to be divided. If this node (usually the
parent) does not exist of if its margin is zero, :func:`insert_dot`
inserts dot, otherwise :func:`insert_num` is called which inserts the
number in the user-specified format.  ``my_format`` contains the regular
expression and the callback function.

Printing the tree with

.. literalinclude:: code/orngTree2.py
    :lines: 33

yields::

    petal width<0.800: Iris-setosa 100% (100.00%)
    petal width>=0.800
    |    petal width<1.750
    |    |    petal length<5.350: Iris-versicolor 88% (108.57%)
    |    |    petal length>=5.350: Iris-virginica 100% (122.73%)
    |    petal width>=1.750
    |    |    petal length<4.850: Iris-virginica 33% (34.85%)
    |    |    petal length>=4.850: Iris-virginica 100% (104.55%)

Plotting with Dot
---------------------------

To produce images of trees, first create a .dot file with
:meth:`TreeClassifier.dot`. If it was saved to "tree5.dot", plot a gif
with the following command::

    dot -Tgif tree5.dot -otree5.gif

Check GraphViz's dot documentation for more options and
output formats.


===========================
C4.5 Classifier and Learner
===========================

C4.5 is, as  a standard benchmark in machine learning, incorporated in
Orange. The implementation uses the original C4.5 code, so the resulting
tree is exactly like the one that would be build by standalone C4.5. The
tree build is only made accessible in Python.

:class:`C45Learner` and :class:`C45Classifier` behave
like any other Orange learner and classifier. Unlike most of Orange 
learning algorithms, C4.5 does not accepts weighted instances.

Building the C4.5 plug-in
=========================

C4.5 is not distributed with Orange, but it can be added as a
plug-in. A C compiler is needed for the procedure: on Windows MS Visual C
(CL.EXE and LINK.EXE must be on the PATH), on Linux and OS X gcc (OS X
users can download it from Apple).

Orange must be installed prior to building C4.5.

#. Download 
   `C4.5 (Release 8) sources <http://www.rulequest.com/Personal/c4.5r8.tar.gz>`_
   from the `Rule Quest's site <http://www.rulequest.com/>`_ and extract
   them. The files will be modified in the
   further process.
#. Download
   `buildC45.zip <http://orange.biolab.si/orange/download/buildC45.zip>`_
   and unzip its contents into the directory R8/Src of the C4.5 sources
   (this directory contains, for instance, the file average.c).
#. Run buildC45.py, which will build the plug-in and put it next to 
   orange.pyd (or orange.so on Linux/Mac).
#. Run python, type ``import Orange`` and
   create ``Orange.classification.tree.C45Learner()``. This should
   succeed.
#. Finally, you can remove C4.5 sources.

The buildC45.py creates .h files that wrap Quinlan's .i files and
ensure that they are not included twice. It modifies C4.5 sources to
include .h's instead of .i's (this step can hardly fail). Then it compiles
ensemble.c into c45.dll or c45.so and puts it next to Orange. In the end
it checks if the built C4.5 gives the same results as the original.

.. autoclass:: C45Learner
    :members:

.. autoclass:: C45Classifier
    :members:

.. class:: C45Node

    This class is a reimplementation of the corresponding *struct* from
    Quinlan's C4.5 code.

    .. attribute:: node_type

        Type of the node:  :obj:`C45Node.Leaf` (0), 
        :obj:`C45Node.Branch` (1), :obj:`C45Node.Cut` (2),
        :obj:`C45Node.Subset` (3). "Leaves" are leaves, "branches"
        split instances based on values of a discrete attribute,
        "cuts" cut them according to a threshold value of a continuous
        attributes and "subsets" use discrete attributes but with subsetting
        so that several values can go into the same branch.

    .. attribute:: leaf

        Value returned by that leaf. The field is defined for internal 
        nodes as well.

    .. attribute:: items

        Number of (learning) instances in the node.

    .. attribute:: class_dist

        Class distribution for the node (of type 
        :obj:`Orange.statistics.distribution.Discrete`).

    .. attribute:: tested
        
        The attribute used in the node's test. If node is a leaf,
        obj:`tested` is None, if node is of type :obj:`Branch` or :obj:`Cut`
        :obj:`tested` is a discrete attribute, and if node is of type
        :obj:`Cut` then :obj:`tested` is a continuous attribute.

    .. attribute:: cut

        A threshold for continuous attributes, if node is of type :obj:`Cut`.
        Undefined otherwise.

    .. attribute:: mapping

        Mapping for nodes of type :obj:`Subset`. Element ``mapping[i]``
        gives the index for an instance whose value of :obj:`tested` is *i*. 
        Here, *i* denotes an index of value, not a :class:`Orange.data.Value`.

    .. attribute:: branch
        
        A list of branches stemming from this node.

Examples
========

.. _tree_c45.py: code/tree_c45.py
.. _iris.tab: code/iris.tab

This
script constructs the same learner as you would get by calling
the usual C4.5:

.. literalinclude:: code/tree_c45.py
   :lines: 7-14

Both C4.5 command-line symbols and variable names can be used. The 
following lines produce the same result::

    tree = Orange.classification.tree.C45Learner(data, m=100)
    tree = Orange.classification.tree.C45Learner(data, min_objs=100)

A veteran C4.5 might prefer :func:`C45Learner.commandline`::

    lrn = Orange.classification.tree.C45Learner()
    lrn.commandline("-m 1 -s")
    tree = lrn(data)

The following script prints out the tree same format as C4.5 does.

.. literalinclude:: code/tree_c45_printtree.py

For the leaves just the value in ``node.leaf`` in printed. Since
:obj:`C45Node` does not know to which attribute it belongs, we need to
convert it to a string through ``classvar``, which is passed as an extra
argument to the recursive part of printTree.

For discrete splits without subsetting, we print out all attribute values
and recursively call the function for all branches. Continuous splits
are equally easy to handle.

For discrete splits with subsetting, we iterate through branches,
retrieve the corresponding values that go into each branch to inset,
turn the values into strings and print them out, separately treating
the case when only a single value goes into the branch.

=================
SimpleTreeLearner
=================

:obj:`SimpleTreeLearner` is an implementation of regression and classification
trees. It is faster than :obj:`TreeLearner` at the expense of flexibility.
It uses gain ratio for classification and mse for regression.

:obj:`SimpleTreeLearner` was developed for speeding up the construction
of random forests, but can also be used as a standalone tree.

.. class:: SimpleTreeLearner

    .. attribute:: max_majority

        Maximal proportion of majority class. When this is exceeded,
        induction stops.

    .. attribute:: min_instances

        Minimal number of instances in leaves. Instance count is weighed.

    .. attribute:: max_depth

        Maximal depth of tree.

    .. attribute:: skip_prob
        
        At every split an attribute will be skipped with probability ``skip_prob``.
        Useful for building random forests.
        
Examples
========

:obj:`SimpleTreeLearner` is used in much the same way as :obj:`TreeLearner`.
A typical example of using :obj:`SimpleTreeLearner` would be to build a random
forest:

.. literalinclude:: code/simple_tree_random_forest.py


References
==========

Bratko, I. (2002). `Prolog Programming for Artificial Intelligence`, Addison 
Wesley, 2002.

E Koutsofios, SC North. Drawing Graphs with dot. AT&T Bell Laboratories,
Murray Hill NJ, U.S.A., October 1993.

`Graphviz - open source graph drawing software <http://www.research.att.com/sw/tools/graphviz/>`_
A home page of AT&T's dot and similar software packages.

"""

"""
TODO C++ aliases

SplitConstructor.discrete/continuous_split_constructor -> SplitConstructor.discrete 
Node.examples -> Node.instances
"""

from Orange.core import \
     TreeLearner as _TreeLearner, \
         TreeClassifier as _TreeClassifier, \
         SimpleTreeLearner, \
         SimpleTreeClassifier, \
         C45Learner as _C45Learner, \
         C45Classifier as _C45Classifier, \
         C45TreeNode as C45Node, \
         C45TreeNodeList as C45NodeList, \
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
              TreeSplitConstructor_Measure as SplitConstructor_Score, \
                   TreeSplitConstructor_Attribute as SplitConstructor_Feature, \
                   TreeSplitConstructor_ExhaustiveBinary as SplitConstructor_ExhaustiveBinary, \
                   TreeSplitConstructor_OneAgainstOthers as SplitConstructor_OneAgainstOthers, \
                   TreeSplitConstructor_Threshold as SplitConstructor_Threshold, \
         TreeStopCriteria as StopCriteria, \
              TreeStopCriteria_Python as StopCriteria_Python, \
              TreeStopCriteria_common as StopCriteria_common

import Orange.core
import operator
import base64
import re
import Orange.data
import Orange.feature.scoring
import warnings

class C45Learner(Orange.classification.Learner):
    """
    :class:`C45Learner`'s attributes have double names - those that
    you know from C4.5 command lines and the corresponding names of C4.5's
    internal variables. All defaults are set as in C4.5; if you change
    nothing, you are running C4.5.

    .. attribute:: gain_ratio (g)
        
        Determines whether to use information gain (false, default)
        or gain ratio for selection of attributes (true).

    .. attribute:: batch (b)

        Turn on batch mode (no windows, no iterations); this option is
        not documented in C4.5 manuals. It conflicts with "window",
        "increment" and "trials".

    .. attribute:: subset (s)
        
        Enables subsetting (default: false, no subsetting),
 
    .. attribute:: prob_thresh (p)

        Probabilistic threshold for continuous attributes (default: false).

    .. attribute:: min_objs (m)
        
        Minimal number of objects (instances) in leaves (default: 2).

    .. attribute:: window (w)

        Initial windows size (default: maximum of 20% and twice the
        square root of the number of data objects).

    .. attribute:: increment (i)

        The maximum number of objects that can be added to the window
        at each iteration (default: 20% of the initial window size).

    .. attribute:: cf (c)

        Prunning confidence level (default: 25%).

    .. attribute:: trials (t)

        Set the number of trials in iterative (i.e. non-batch) mode (default: 10).

    .. attribute:: prune
        
        Return pruned tree (not an original C4.5 option) (default: true)
    """

    _rename_new_old = { "min_objs": "minObjs", "probTresh": "prob_tresh",
            "gain_ratio": "gainRatio" }
    #_rename_new_old = {}
    _rename_old_new = dict((a,b) for b,a in _rename_new_old.items())

    @classmethod
    def _rename_dict(cls, dic):
        return dict((cls._rename_arg(a),b) for a,b in dic.items())

    @classmethod
    def _rename_arg(cls, a):
        if a in cls._rename_old_new:
            Orange.misc.deprecation_warning(a, cls._rename_old_new[a], stacklevel=4)
        return cls._rename_new_old.get(a, a)

    def __new__(cls, instances = None, weightID = 0, **argkw):
        self = Orange.classification.Learner.__new__(cls, **cls._rename_dict(argkw))
        if instances:
            self.__init__(**cls._rename_dict(argkw))
            return self.__call__(instances, weightID)
        else:
            return self
        
    def __init__(self, **kwargs):
        self.base = _C45Learner(**self._rename_dict(kwargs))

    def __setattr__(self, name, value):
        nn = self._rename_arg(name)
        if name != "base" and nn in self.base.__dict__:
            self.base.__dict__[nn] = value
        elif name == "base":
            self.__dict__["base"] = value
        else:
            settingAttributesNotSuccessful

    def __getattr__(self, name):
        nn = self._rename_arg(name)
        if name != "base":
            return self.base.__dict__[nn]
        else:
            return self.base

    def __call__(self, *args, **kwargs):
        return C45Classifier(self.base(*args, **self._rename_dict(kwargs)))

    def commandline(self, ln):
        """
        Set the arguments with a C4.5 command line.
        """
        self.base.commandline(ln)
    
 
class C45Classifier(Orange.classification.Classifier):
    """
    A faithful reimplementation of Quinlan's function from C4.5, but
    uses a tree composed of :class:`C45Node` instead of C4.5's original
    tree structure.

    .. attribute:: tree

        C4.5 tree stored as :obj:`C45Node`.
    """

    def __init__(self, base_classifier):
        self.nativeClassifier = base_classifier
        for k, v in self.nativeClassifier.__dict__.items():
            self.__dict__[k] = v
  
    def __call__(self, instance, result_type=Orange.classification.Classifier.GetValue,
                 *args, **kwdargs):
        """Classify a new instance.
        
        :param instance: instance to be classified.
        :type instance: :class:`Orange.data.Instance`
        :param result_type: 
              :class:`Orange.classification.Classifier.GetValue` or \
              :class:`Orange.classification.Classifier.GetProbabilities` or
              :class:`Orange.classification.Classifier.GetBoth`
        
        :rtype: :class:`Orange.data.Value`, 
              :class:`Orange.statistics.Distribution` or a tuple with both
        """
        return self.nativeClassifier(instance, result_type, *args, **kwdargs)

    def __setattr__(self, name, value):
        if name == "nativeClassifier":
            self.__dict__[name] = value
            return
        if name in self.nativeClassifier.__dict__:
            self.nativeClassifier.__dict__[name] = value
        self.__dict__[name] = value

    def __str__(self):
        return self.dump()
   
    def dump(self):  
        """
        Print the tree in the same form as Ross Quinlan's 
        C4.5 program.

        ::

            import Orange

            data = Orange.data.Table("voting")
            c45 = Orange.classification.tree.C45Learner(data)
            print c45.dump()

        prints

        ::

            physician-fee-freeze = n: democrat (253.4)
            physician-fee-freeze = y:
            |   synfuels-corporation-cutback = n: republican (145.7)
            |   synfuels-corporation-cutback = y:
            |   |   mx-missile = y: democrat (6.0)
            |   |   mx-missile = n:
            |   |   |   adoption-of-the-budget-resolution = n: republican (22.6)
            |   |   |   adoption-of-the-budget-resolution = y:
            |   |   |   |   anti-satellite-test-ban = n: democrat (5.0)
            |   |   |   |   anti-satellite-test-ban = y: republican (2.2)


        The standalone C4.5 would print::

            physician-fee-freeze = n: democrat (253.4/5.9)
            physician-fee-freeze = y:
            |   synfuels-corporation-cutback = n: republican (145.7/6.2)
            |   synfuels-corporation-cutback = y:
            |   |   mx-missile = y: democrat (6.0/2.4)
            |   |   mx-missile = n:
            |   |   |   adoption-of-the-budget-resolution = n: republican (22.6/5.2)
            |   |   |   adoption-of-the-budget-resolution = y:
            |   |   |   |   anti-satellite-test-ban = n: democrat (5.0/1.2)
            |   |   |   |   anti-satellite-test-ban = y: republican (2.2/1.0)

        C4.5 also prints out the number of errors on learning data in
        each node.
        """
        return  _c45_printTree0(self.tree, self.class_var, 0)

def _c45_showBranch(node, classvar, lev, i):
    var = node.tested
    str_ = ""
    if node.node_type == 1:
        str_ += "\n"+"|   "*lev + "%s = %s:" % (var.name, var.values[i])
        str_ += _c45_printTree0(node.branch[i], classvar, lev+1)
    elif node.node_type == 2:
        str_ += "\n"+"|   "*lev + "%s %s %.1f:" % (var.name, ["<=", ">"][i], node.cut)
        str_ += _c45_printTree0(node.branch[i], classvar, lev+1)
    else:
        inset = filter(lambda a:a[1]==i, enumerate(node.mapping))
        inset = [var.values[j[0]] for j in inset]
        if len(inset)==1:
            str_ += "\n"+"|   "*lev + "%s = %s:" % (var.name, inset[0])
        else:
            str_ +=  "\n"+"|   "*lev + "%s in {%s}:" % (var.name, ", ".join(inset))
        str_ += _c45_printTree0(node.branch[i], classvar, lev+1)
    return str_
        
        
def _c45_printTree0(node, classvar, lev):
    var = node.tested
    str_ = ""
    if node.node_type == 0:
        str_ += "%s (%.1f)" % (classvar.values[int(node.leaf)], node.items) 
    else:
        for i, branch in enumerate(node.branch):
            if not branch.node_type:
                str_ += _c45_showBranch(node, classvar, lev, i)
        for i, branch in enumerate(node.branch):
            if branch.node_type:
                str_ += _c45_showBranch(node, classvar, lev, i)
    return str_

def _printTreeC45(tree):
    print _c45_printTree0(tree.tree, tree.class_var, 0)


import Orange.feature.scoring as fscoring

class TreeLearner(Orange.core.Learner):
    """
    A classification or regression tree learner. If a set of instances
    is given on initialization, a :class:`TreeClassifier` is built and
    returned instead. All attributes can also be set on initialization.

    **The tree induction process**

    #. The learning instances are copied to a table, unless
       :obj:`store_instances` is `False` and they already are in table.
    #. Apriori class probabilities are computed. A list of
       candidate features for the split is compiled; in the beginning,
       all features are candidates.
    #. The recursive part. The contingency matrix is computed by
       :obj:`contingency_computer`. Contingencies are used by :obj:`split`,
       :obj:`stop` and :obj:`splitter`.
    #. If the induction should :obj:`stop`, a :obj:`~Node.node_classifier`
       is built by calling :obj:`node_learner` with the given instances,
       weight ID and the contingency matrix. As the learner uses
       contingencies whenever possible, the :obj:`contingency_computer`
       will affect the :obj:`~Node.node_classifier`. The node is returned.
    #. If the induction continues, a :obj:`split` is called.
       If :obj:`split` fails to return a branch selector, induction stops 
       and the :obj:`Node` is returned.
    #. The feature spent (if any) is removed from the candidate list.
    #. Instances are divided into child nodes with :obj:`splitter`.
       The process recursively continues with step 3 for
       each of the non-empty subsets. If the splitter returned weights,
       they are used for each branch.

    **Attributes**

    .. attribute:: node_learner

        Induces a classifier from instances in a node. It is used both
        for internal nodes and leaves. The default is
        :obj:`Orange.classification.majority.MajorityLearner`.

    .. attribute:: descender

        The descender that the induced :obj:`TreeClassifier` will
        use. The default is :obj:`Descender_UnknownMergeAsSelector`.
        It votes with the :obj:`branch_selector`'s distribution for
        vote weights.

    .. attribute:: contingency_computer

        Defines the computation of contingency matrices (used by
        :obj:`split`, :obj:`stop`, :obj:`splitter`). It can be used,
        for example, to change the treatment of unknown values. By
        default ordinary contingency matrices are computed.

    **Split construction**

    .. attribute:: split
        
        A :obj:`SplitConstructor` or a function with the same signature as
        :obj:`SplitConstructor.__call__`. It is useful for prototyping
        new tree induction algorithms. If :obj:`split` is defined, other 
        arguments that affect split construction are ignored. These include
        :obj:`binarization`, :obj:`measure`, :obj:`worst_acceptable` and
        :obj:`min_subset`. Default: :class:`SplitConstructor_Combined`
        with separate constructors for discrete and continuous
        features. Discrete features are used as they are, while
        continuous are binarized. Features are scored with gain ratio.
        At least two instances in a leaf are required for
        discrete and five for continuous features.

    .. attribute:: binarization

        If 1, :class:`SplitConstructor_ExhaustiveBinary` is used.
        If 2, use :class:`SplitConstructor_OneAgainstOthers`. If
        0, do not use binarization (use :class:`SplitConstructor_Feature`).
        Default: 0.

    .. attribute:: measure
    
        A score to evaluate features for splitting instances in a
        node.  A subclass of :class:`Orange.feature.scoring.Score`
        (perhaps :class:`~Orange.feature.scoring.InfoGain`,
        :class:`~Orange.feature.scoring.GainRatio`,
        :class:`~Orange.feature.scoring.Gini`,
        :class:`~Orange.feature.scoring.Relief`, or
        :class:`~Orange.feature.scoring.MSE`). Default:
        :class:`Orange.feature.scoring.GainRatio`.

    .. attribute:: relief_m, relief_k

        Set `m` and `k` for :class:`~Orange.feature.scoring.Relief`,
        if chosen.

    .. attribute:: splitter

        :class:`Splitter` or a function with the same
        signature as :obj:`Splitter.__call__`. The default is
        :class:`Splitter_UnknownsAsSelector` that splits the
        learning instances according to distributions given by the
        selector.

    **Pruning**

    .. attribute:: worst_acceptable

        The lowest required feature score. If the score of the best
        feature is below this margin, the tree is not grown further
        (default: 0).

    .. attribute:: min_subset

        The lowest required number of instances in non-null leaves (default: 0).

    .. attribute:: min_instances

        Data subsets with less than :obj:`min_instances`
        instances are not split any further. Therefore, all leaves in the tree
        will contain at least that many instances (default: 0).

    .. attribute:: max_depth

        Maximal tree depth. If 0, only root is generated. 
        The default is 100. 

    .. attribute:: max_majority

        Induction stops when the proportion of majority class in the
        node exceeds the value set by this parameter (default: 1.0). 
  
    .. attribute:: stop

        :class:`StopCriteria` or a function with the same signature as
        :obj:`StopCriteria.__call__`. Useful for prototyping new tree
        induction algorithms.  When used, parameters  :obj:`max_majority`
        and :obj:`min_instances` will not be  considered.  The default
        stopping criterion stops induction when all instances in a node
        belong to the same class.

    .. attribute:: m_pruning

        If non-zero, invokes an error-based bottom-up post-pruning,
        where m-estimate is used to estimate class probabilities 
        (default: 0).

    .. attribute:: same_majority_pruning

        If true, invokes a bottom-up post-pruning by removing the
        subtrees of which all leaves classify to the same class
        (default: False).

    **Record keeping**

    .. attribute:: store_distributions 
    
    .. attribute:: store_contingencies
    
    .. attribute:: store_instances
    
    .. attribute:: store_node_classifier

        Determines whether to store class distributions,
        contingencies and instances in :class:`Node`, and whether the
        :obj:`Node.node_classifier` should be build for internal nodes
        also (it is needed by the :obj:`Descender` or for post-pruning).
        Not storing distributions but storing contingencies does not
        save any memory, since distributions actually points to the
        same distribution that is stored in :obj:`contingency.classes`.
        By default everything except :obj:`store_instances` is enabled.

    """
    def __new__(cls, data=None, weightID=0, **argkw):
        self = Orange.core.Learner.__new__(cls, **argkw)
        if data:
            self.__init__(**argkw)
            return self.__call__(data, weightID)
        else:
            return self
    
    def __init__(self, **kw):

        #name, buildfunction, parameters
        #buildfunctions are not saved as function references
        #because that would make problems with object copies
        for n,(fn,_) in self._built_fn.items():
            self.__dict__["_handset_" + n] = False

        #measure has to be before split
        self.measure = None
        self.split = None
        self.stop = None
        self.splitter = None
        
        for n,(fn,_) in self._built_fn.items():
            self.__dict__[n] = fn(self)

        for k,v in kw.items():
            self.__setattr__(k,v)
      
    def __call__(self, instances, weight=0):
        """
        Return a classifier from the given instances.
        """
        bl = self._base_learner()

        #set the scoring criteria for regression if it was not
        #set by the user
        if not self._handset_split and not self.measure:
            measure = fscoring.GainRatio() \
                if instances.domain.class_var.var_type == Orange.data.Type.Discrete \
                else fscoring.MSE()
            bl.split.continuous_split_constructor.measure = measure
            bl.split.discrete_split_constructor.measure = measure
         
        if self.splitter != None:
            bl.example_splitter = self.splitter

        #post pruning
        tree = bl(instances, weight)
        if getattr(self, "same_majority_pruning", 0):
            tree = Pruner_SameMajority(tree)
        if getattr(self, "m_pruning", 0):
            tree = Pruner_m(tree, m=self.m_pruning)

        return TreeClassifier(base_classifier=tree) 

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        for n,(fn,v) in self._built_fn.items():
            if name in v:
                if not self.__dict__["_handset_" + n]:
                    self.__dict__[n] = fn(self)
                else:
                    warnings.warn("Changing \"" + name + "\" does not have any effect as \"" + n + "\" was already set", UserWarning, 2)
            elif n == name:
                if value == None:
                    self.__dict__[n] = fn(self)
                    self.__dict__["_handset_" + n] = False
                    #print n, "was now disabled by hand"
                else:
                    self.__dict__["_handset_" + n] = True
                    #print n, "is now handset"
        #print self.__dict__

    def __delattr__(self, name):
        self.__setattr__(name, None) #use the __setattr__
        del self.__dict__[name]

    def instance(self):
        """
        DEPRECATED. Return a base learner - an object 
        of :class:`_TreeLearner`. 
        This method is left for backwards compatibility.
        """
        return self._base_learner()

    def _build_split(self):
        """
        Return the split constructor built according to object attributes.
        """
        split = SplitConstructor_Combined()
        split.continuous_split_constructor = \
            SplitConstructor_Threshold()
        binarization = getattr(self, "binarization", 0)
        if binarization == 1:
            split.discrete_split_constructor = \
                SplitConstructor_ExhaustiveBinary()
        elif binarization == 2:
            split.discrete_split_constructor = \
                SplitConstructor_OneAgainstOthers()
        else:
            split.discrete_split_constructor = \
                SplitConstructor_Feature()

        measures = {"infoGain": fscoring.InfoGain,
            "gainRatio": fscoring.GainRatio,
            "gini": fscoring.Gini,
            "relief": fscoring.Relief,
            "retis": fscoring.MSE
            }

        measure = self.measure
        if isinstance(measure, str):
            measure = measures[measure]()
        if not measure:
            measure = fscoring.GainRatio()

        measureIsRelief = isinstance(measure, fscoring.Relief)
        relM = getattr(self, "relief_m", None)
        if relM and measureIsRelief:
            measure.m = relM
        
        relK = getattr(self, "relief_k", None)
        if relK and measureIsRelief:
            measure.k = relK

        split.continuous_split_constructor.measure = measure
        split.discrete_split_constructor.measure = measure

        wa = getattr(self, "worst_acceptable", 0)
        if wa:
            split.continuous_split_constructor.worst_acceptable = wa
            split.discrete_split_constructor.worst_acceptable = wa

        ms = getattr(self, "min_subset", 0)
        if ms:
            split.continuous_split_constructor.min_subset = ms
            split.discrete_split_constructor.min_subset = ms

        return split

    def _build_stop(self):
        """
        Return the stop criteria built according to object's attributes.
        """
        stop = Orange.classification.tree.StopCriteria_common()
        mm = getattr(self, "max_majority", 1.0)
        if mm < 1.0:
            stop.max_majority = self.max_majority
        me = getattr(self, "min_instances", 0)
        if me:
            stop.min_examples = self.min_instances
        return stop

    def _base_learner(self):
        learner = _TreeLearner()

        learner.split = self.split
        learner.stop = self.stop

        for a in ["store_distributions", "store_contingencies",
            "store_node_classifier", "node_learner", "max_depth", "contingency_computer", "descender" ]:
            if hasattr(self, a):
                setattr(learner, a, getattr(self, a))

        if hasattr(self, "store_instances"):
            learner.store_examples = self.store_instances

        return learner

    _built_fn = { 
            "split": [ _build_split, [ "binarization", "measure", "relief_m", "relief_k", "worst_acceptable", "min_subset" ] ], \
            "stop": [ _build_stop, ["max_majority", "min_instances" ] ] 
        }



TreeLearner = Orange.misc.deprecated_members({
          "mForPruning": "m_pruning",
          "sameMajorityPruning": "same_majority_pruning",
          "reliefM": "relief_m",
          "reliefK": "relief_k",
          "storeDistributions": "store_distributions",
          "storeContingencies": "store_contingencies",
          "storeExamples": "store_instances",
          "store_examples": "store_instances",
          "storeNodeClassifier": "store_node_classifier",
          "worstAcceptable": "worst_acceptable",
          "minSubset": "min_subset",
          "maxMajority": "max_majority",
          "minExamples": "min_instances",
          "maxDepth": "max_depth",
          "nodeLearner": "node_learner",
          "min_examples": "min_instances"
}, wrap_methods=[])(TreeLearner)

#
# the following is for the output
#

fs = r"(?P<m100>\^?)(?P<fs>(\d*\.?\d*)?)"
""" Defines the multiplier by 100 (``^``) and the format
for the number of decimals (e.g. ``5.3``). The corresponding 
groups are named ``m100`` and ``fs``. """

by = r"(?P<by>(b(P|A)))?"
""" Defines bP or bA or nothing; the result is in groups by. """

bysub = r"((?P<bysub>b|s)(?P<by>P|A))?"
opc = r"(?P<op>=|<|>|(<=)|(>=)|(!=))(?P<num>\d*\.?\d+)"
opd = r'(?P<op>=|(!=))"(?P<cls>[^"]*)"'
intrvl = r'((\((?P<intp>\d+)%?\))|(\(0?\.(?P<intv>\d+)\))|)'
fromto = r"(?P<out>!?)(?P<lowin>\(|\[)(?P<lower>\d*\.?\d+)\s*,\s*(?P<upper>\d*\.?\d+)(?P<upin>\]|\))"
re_V = re.compile("%V")
re_N = re.compile("%"+fs+"N"+by)
re_M = re.compile("%"+fs+"M"+by)
re_m = re.compile("%"+fs+"m"+by)
re_Ccont = re.compile("%"+fs+"C"+by+opc)
re_Cdisc = re.compile("%"+fs+"C"+by+opd)
re_ccont = re.compile("%"+fs+"c"+by+opc)
re_cdisc = re.compile("%"+fs+"c"+by+opd)
re_Cconti = re.compile("%"+fs+"C"+by+fromto)
re_cconti = re.compile("%"+fs+"c"+by+fromto)
re_D = re.compile("%"+fs+"D"+by)
re_d = re.compile("%"+fs+"d"+by)
re_AE = re.compile("%"+fs+"(?P<AorE>A|E)"+bysub)
re_I = re.compile("%"+fs+"I"+intrvl)

def insert_str(s, mo, sub):
    """ Replace the part of s which is covered by mo 
    with the string sub. """
    return s[:mo.start()] + sub + s[mo.end():]

def insert_dot(s, mo):
    """ Replace the part of s which is covered by mo 
    with a dot.  You should use this when the 
    function cannot compute the desired quantity; it is called, for instance, 
    when it needs to divide by something in the parent, but the parent 
    doesn't exist.
    """
    return s[:mo.start()] + "." + s[mo.end():]

def insert_num(s, mo, n):
    """ Replace the part of s matched by mo with the number n, 
    formatted as specified by the user, that is, it multiplies 
    it by 100, if needed, and prints with the right number of 
    places and decimals. It does so by checking the mo
    for a group named m100 (representing the ``^`` in the format string) 
    and a group named fs representing the part giving the number o
    f decimals (e.g. ``5.3``).
    """
    grps = mo.groupdict()
    m100 = grps.get("m100", None)
    if m100:
        n *= 100
    fs = grps.get("fs") or (m100 and ".0" or "5.3")
    return s[:mo.start()] + ("%%%sf" % fs % n) + s[mo.end():]

def by_whom(by, parent, tree):
    """ If by equals bp, return parent, else return
    ``tree.tree``. This is used to find what to divide the quantity 
    with, when division is required.
    """
    if by=="bP":
        return parent
    else:
        return tree.tree

def replaceV(strg, mo, node, parent, tree):
    return insert_str(strg, mo, str(node.node_classifier.default_value))

def replaceN(strg, mo, node, parent, tree):
    by = mo.group("by")
    N = node.distribution.abs
    if by:
        whom = by_whom(by, parent, tree)
        if whom and whom.distribution:
            if whom.distribution.abs > 1e-30:
                N /= whom.distribution.abs
        else:
            return insert_dot(strg, mo)
    return insert_num(strg, mo, N)
        

def replaceM(strg, mo, node, parent, tree):
    by = mo.group("by")
    maj = int(node.node_classifier.default_value)
    N = node.distribution[maj]
    if by:
        whom = by_whom(by, parent, tree)
        if whom and whom.distribution:
            if whom.distribution[maj] > 1e-30:
                N /= whom.distribution[maj]
        else:
            return insert_dot(strg, mo)
    return insert_num(strg, mo, N)
        

def replacem(strg, mo, node, parent, tree):
    by = mo.group("by")
    maj = int(node.node_classifier.default_value)
    if node.distribution.abs > 1e-30:
        N = node.distribution[maj] / node.distribution.abs
        if by:
            if whom and whom.distribution:
                byN = whom.distribution[maj] / whom.distribution.abs
                if byN > 1e-30:
                    N /= byN
            else:
                return insert_dot(strg, mo)
    else:
        N = 0.
    return insert_num(strg, mo, N)


def replaceCdisc(strg, mo, node, parent, tree):
    if tree.class_var.var_type != Orange.data.Type.Discrete:
        return insert_dot(strg, mo)
    
    by, op, cls = mo.group("by", "op", "cls")
    N = node.distribution[cls]
    if op == "!=":
        N = node.distribution.abs - N
    if by:
        whom = by_whom(by, parent, tree)
        if whom and whom.distribution:
            if whom.distribution[cls] > 1e-30:
                N /= whom.distribution[cls]
        else:
            return insert_dot(strg, mo)
    return insert_num(strg, mo, N)

    
def replacecdisc(strg, mo, node, parent, tree):
    if tree.class_var.var_type != Orange.data.Type.Discrete:
        return insert_dot(strg, mo)
    
    op, by, cls = mo.group("op", "by", "cls")
    N = node.distribution[cls]
    if node.distribution.abs > 1e-30:
        N /= node.distribution.abs
        if op == "!=":
            N = 1 - N
    if by:
        whom = by_whom(by, parent, tree)
        if whom and whom.distribution:
            if whom.distribution[cls] > 1e-30:
                N /= whom.distribution[cls] / whom.distribution.abs
        else:
            return insert_dot(strg, mo)
    return insert_num(strg, mo, N)


__opdict = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "=": operator.eq, "!=": operator.ne}

def replaceCcont(strg, mo, node, parent, tree):
    if tree.class_var.var_type != Orange.data.Type.Continuous:
        return insert_dot(strg, mo)
    
    by, op, num = mo.group("by", "op", "num")
    op = __opdict[op]
    num = float(num)
    N = sum([x[1] for x in node.distribution.items() if op(x[0], num)], 0.)
    if by:
        whom = by_whom(by, parent, tree)
        if whom and whom.distribution:
            byN = sum([x[1] for x in whom.distribution.items() if op(x[0], num)], 0.)
            if byN > 1e-30:
                N /= byN
        else:
            return insert_dot(strg, mo)

    return insert_num(strg, mo, N)
    
    
def replaceccont(strg, mo, node, parent, tree):
    if tree.class_var.var_type != Orange.data.Type.Continuous:
        return insert_dot(strg, mo)
    
    by, op, num = mo.group("by", "op", "num")
    op = __opdict[op]
    num = float(num)
    N = sum([x[1] for x in node.distribution.items() if op(x[0], num)], 0.)
    if node.distribution.abs > 1e-30:
        N /= node.distribution.abs
    if by:
        whom = by_whom(by, parent, tree)
        if whom and whom.distribution:
            byN = sum([x[1] for x in whom.distribution.items() if op(x[0], num)], 0.)
            if byN > 1e-30:
                N /= byN/whom.distribution.abs # abs > byN, so byN>1e-30 => abs>1e-30
        else:
            return insert_dot(strg, mo)
    return insert_num(strg, mo, N)


def extractInterval(mo, dist):
    out, lowin, lower, upper, upin = mo.group("out", "lowin", "lower", "upper", "upin")
    lower, upper = float(lower), float(upper)
    if out:
        lop = lowin == "(" and operator.le or operator.lt
        hop = upin == ")" and operator.ge or operator.ge
        return filter(lambda x:lop(x[0], lower) or hop(x[0], upper), dist.items())
    else:
        lop = lowin == "(" and operator.gt or operator.ge
        hop = upin == ")" and operator.lt or operator.le
        return filter(lambda x:lop(x[0], lower) and hop(x[0], upper), dist.items())

    
def replaceCconti(strg, mo, node, parent, tree):
    if tree.class_var.var_type != Orange.data.Type.Continuous:
        return insert_dot(strg, mo)

    by = mo.group("by")
    N = sum([x[1] for x in extractInterval(mo, node.distribution)])
    if by:
        whom = by_whom(by, parent, tree)
        if whom and whom.distribution:
            byN = sum([x[1] for x in extractInterval(mo, whom.distribution)])
            if byN > 1e-30:
                N /= byN
        else:
            return insert_dot(strg, mo)
        
    return insert_num(strg, mo, N)

            
def replacecconti(strg, mo, node, parent, tree):
    if tree.class_var.var_type != Orange.data.Type.Continuous:
        return insert_dot(strg, mo)

    N = sum([x[1] for x in extractInterval(mo, node.distribution)])
    ab = node.distribution.abs
    if ab > 1e-30:
        N /= ab

    by = mo.group("by")
    if by:
        whom = by_whom(by, parent, tree)
        if whom and whom.distribution:
            byN = sum([x[1] for x in extractInterval(mo, whom.distribution)])
            if byN > 1e-30:
                N /= byN/whom.distribution.abs
        else:
            return insert_dot(strg, mo)
        
    return insert_num(strg, mo, N)

    
def replaceD(strg, mo, node, parent, tree):
    if tree.class_var.var_type != Orange.data.Type.Discrete:
        return insert_dot(strg, mo)

    fs, by, m100 = mo.group("fs", "by", "m100")
    dist = list(node.distribution)
    if by:
        whom = by_whom(by, parent, tree)
        if whom:
            for i, d in enumerate(whom.distribution):
                if d > 1e-30:
                    dist[i] /= d
        else:
            return insert_dot(strg, mo)
    mul = m100 and 100 or 1
    fs = fs or (m100 and ".0" or "5.3")
    return insert_str(strg, mo, "["+", ".join(["%%%sf" % fs % (N*mul) for N in dist])+"]")


def replaced(strg, mo, node, parent, tree):
    if tree.class_var.var_type != Orange.data.Type.Discrete:
        return insert_dot(strg, mo)

    fs, by, m100 = mo.group("fs", "by", "m100")
    dist = list(node.distribution)
    ab = node.distribution.abs
    if ab > 1e-30:
        dist = [d/ab for d in dist]
    if by:
        whom = by_whom(by, parent, tree)
        if whom:
            for i, d in enumerate(whom.distribution):
                if d > 1e-30:
                    dist[i] /= d/whom.distribution.abs # abs > d => d>1e-30 => abs>1e-30
        else:
            return insert_dot(strg, mo)
    mul = m100 and 100 or 1
    fs = fs or (m100 and ".0" or "5.3")
    return insert_str(strg, mo, "["+", ".join(["%%%sf" % fs % (N*mul) for N in dist])+"]")


def replaceAE(strg, mo, node, parent, tree):
    if tree.class_var.var_type != Orange.data.Type.Continuous:
        return insert_dot(strg, mo)

    AorE, bysub, by = mo.group("AorE", "bysub", "by")
    
    if AorE == "A":
        A = node.distribution.average()
    else:
        A = node.distribution.error()
    if by:
        whom = by_whom("b"+by, parent, tree)
        if whom:
            if AorE == "A":
                avg = whom.distribution.average()
            else:
                avg = whom.distribution.error()
            if bysub == "b":
                if avg > 1e-30:
                    A /= avg
            else:
                A -= avg
        else:
            return insert_dot(strg, mo)
    return insert_num(strg, mo, A)


Z = { 0.75:1.15, 0.80:1.28, 0.85:1.44, 0.90:1.64, 0.95:1.96, 0.99:2.58 }

def replaceI(strg, mo, node, parent, tree):
    if tree.class_var.var_type != Orange.data.Type.Continuous:
        return insert_dot(strg, mo)

    fs = mo.group("fs") or "5.3"
    intrvl = float(mo.group("intp") or mo.group("intv") or "95")/100.
    mul = mo.group("m100") and 100 or 1

    if not Z.has_key(intrvl):
        raise SystemError, "Cannot compute %5.3f% confidence intervals" % intrvl

    av = node.distribution.average()    
    il = node.distribution.error() * Z[intrvl]
    return insert_str(strg, mo, "[%%%sf-%%%sf]" % (fs, fs) % ((av-il)*mul, (av+il)*mul))


# This class is more a collection of function, merged into a class so 
# that they don't need to transfer too many arguments. It will be 
# constructed, used and discarded, it is not meant to store any information.
class _TreeDumper:
    defaultStringFormats = [(re_V, replaceV), (re_N, replaceN),
         (re_M, replaceM), (re_m, replacem), 
         (re_Cdisc, replaceCdisc), (re_cdisc, replacecdisc),
         (re_Ccont, replaceCcont), (re_ccont, replaceccont),
         (re_Cconti, replaceCconti), (re_cconti, replacecconti),
         (re_D, replaceD), (re_d, replaced), (re_AE, replaceAE), 
         (re_I, replaceI) ]

    def node(self):
        return self.tree.tree if "tree" in self.tree.__dict__ else self.tree

    def __init__(self, leafStr, nodeStr, stringFormats, minExamples, 
        maxDepth, simpleFirst, tree, **kw):
        self.stringFormats = stringFormats
        self.minExamples = minExamples
        self.maxDepth = maxDepth
        self.simpleFirst = simpleFirst
        self.tree = tree
        self.__dict__.update(kw)

        if leafStr:
            self.leafStr = leafStr
        else:
            if self.node().node_classifier.class_var.var_type == \
                    Orange.data.Type.Discrete:
                self.leafStr = "%V (%^.2m%)"
            else:
                self.leafStr = "%V"

        if nodeStr == ".":
            self.nodeStr = self.leafStr
        else:
            self.nodeStr = nodeStr
        

    def formatString(self, strg, node, parent):
        if hasattr(strg, "__call__"):
            return strg(node, parent, self.tree)
        
        if not node:
            return "<null node>"
        
        for rgx, replacer in self.stringFormats:
            if not node.distribution:
                strg = rgx.sub(".", strg)
            else:
                strt = 0
                while True:
                    mo = rgx.search(strg, strt)
                    if not mo:
                        break
                    strg = replacer(strg, mo, node, parent, self.tree)
                    strt = mo.start()+1
                        
        return strg
        

    def showBranch(self, node, parent, lev, i):
        bdes = node.branch_descriptions[i]
        bdes = node.branch_selector.class_var.name + \
            (bdes[0] not in "<=>" and "=" or "") + bdes
        if node.branches[i]:
            nodedes = self.nodeStr and ": " + \
                self.formatString(self.nodeStr, node.branches[i], node) or ""
        else:
            nodedes = "<null node>"
        return "|    "*lev + bdes + nodedes
        
        
    def dumpTree0(self, node, parent, lev):
        if node.branches:
            if node.distribution.abs < self.minExamples or \
                lev > self.maxDepth:
                return "|    "*lev + ". . .\n"
            
            res = ""
            if self.leafStr and self.nodeStr and self.leafStr != self.nodeStr:
                leafsep = "\n"+("|    "*lev)+"    "
            else:
                leafsep = ""
            if self.simpleFirst:
                for i, branch in enumerate(node.branches):
                    if not branch or not branch.branches:
                        if self.leafStr == self.nodeStr:
                            res += "%s\n" % \
                                self.showBranch(node, parent, lev, i)
                        else:
                            res += "%s: %s\n" % \
                                (self.showBranch(node, parent, lev, i),
                                 leafsep + 
                                 self.formatString(self.leafStr, branch, node))
            for i, branch in enumerate(node.branches):
                if branch and branch.branches:
                    res += "%s\n%s" % (self.showBranch(node, parent, lev, i),
                                       self.dumpTree0(branch, node, lev+1))
                elif not self.simpleFirst:
                    if self.leafStr == self.nodeStr:
                        res += "%s\n" % self.showBranch(node, parent, lev, i)
                    else:
                        res += "%s: %s\n" % \
                            (self.showBranch(node, parent, lev, i),
                             leafsep + 
                             self.formatString(self.leafStr, branch, node))
            return res
        else:
            return self.formatString(self.leafStr, node, parent)


    def dumpTree(self):
        node = self.node()
        if self.nodeStr:
            lev, res = 1, "root: %s\n" % \
                self.formatString(self.nodeStr, node, None)
            self.maxDepth += 1
        else:
            lev, res = 0, ""
        return res + self.dumpTree0(node, None, lev)
        

    def dotTree0(self, node, parent, internalName):
        if node.branches:
            if node.distribution.abs < self.minExamples or \
                len(internalName)-1 > self.maxDepth:
                self.fle.write('%s [ shape="plaintext" label="..." ]\n' % \
                    _quoteName(internalName))
                return
                
            label = node.branch_selector.class_var.name
            if self.nodeStr:
                label += "\\n" + self.formatString(self.nodeStr, node, parent)
            self.fle.write('%s [ shape=%s label="%s"]\n' % \
                (_quoteName(internalName), self.nodeShape, label))
            
            for i, branch in enumerate(node.branches):
                if branch:
                    internalBranchName = "%s-%d" % (internalName,i)
                    self.fle.write('%s -> %s [ label="%s" ]\n' % \
                        (_quoteName(internalName), 
                         _quoteName(internalBranchName), 
                         node.branch_descriptions[i]))
                    self.dotTree0(branch, node, internalBranchName)
                    
        else:
            self.fle.write('%s [ shape=%s label="%s"]\n' % \
                (_quoteName(internalName), self.leafShape, 
                self.formatString(self.leafStr, node, parent)))


    def dotTree(self, internalName="n"):
        self.fle.write("digraph G {\n")
        self.dotTree0(self.node(), None, internalName)
        self.fle.write("}\n")

def _quoteName(x):
    return '"%s"' % (base64.b64encode(x))

class TreeClassifier(Orange.classification.Classifier):
    """

    Classifies instances according to the tree stored in :obj:`tree`.

    **The classification process**

    :obj:`TreeClassifier` uses the :obj:`descender` to descend the
    instance from the root. If the :obj:`descender` returns only a
    :obj:`Node` and no distribution, the descend should stop as the node
    was unambiguously selected. The node's :obj:`~Node.node_classifier`
    decides the class.

    If the descender returns a :obj:`Node` and a distribution, as it
    happens, for example, if the instance's value for the :obj:`Node`'s
    feature is unknown, the same process repeats for all subtrees and
    their predictions are combined.

    **Attributes**

    .. attribute:: tree

        The root of the tree, as a :class:`Node`.

    .. attribute:: descender

        A :obj:`Descender` used to descend an instance from the root as
        deeply as possible according to the instance's feature values.
    """
    
    def __init__(self, base_classifier=None):
        if not base_classifier: base_classifier = _TreeClassifier()
        self.nativeClassifier = base_classifier
        for k, v in self.nativeClassifier.__dict__.items():
            self.__dict__[k] = v
  
    def __call__(self, instance, result_type=Orange.classification.Classifier.GetValue,
                 *args, **kwdargs):
        """Classify a new instance.

      
        :param instance: instance to be classified.
        :type instance: :class:`Orange.data.Instance`
        :param result_type: 
              :class:`Orange.classification.Classifier.GetValue` or \
              :class:`Orange.classification.Classifier.GetProbabilities` or
              :class:`Orange.classification.Classifier.GetBoth`
        
        :rtype: :class:`Orange.data.Value`, 
              :class:`Orange.statistics.Distribution` or a tuple with both
        """
        return self.nativeClassifier(instance, result_type, *args, **kwdargs)

    def __setattr__(self, name, value):
        if name == "nativeClassifier":
            self.__dict__[name] = value
            return
        if name in self.nativeClassifier.__dict__:
            self.nativeClassifier.__dict__[name] = value
        self.__dict__[name] = value
    
    def __str__(self):
        return self.dump()

    @Orange.misc.deprecated_keywords({"fileName": "file_name", \
        "leafStr": "leaf_str", "nodeStr": "node_str", \
        "userFormats": "user_formats", "minExamples": "min_examples", \
        "maxDepth": "max_depth", "simpleFirst": "simple_first"})
    def dump(self, leaf_str = "", node_str = "", \
            user_formats=[], min_examples=0, max_depth=1e10, \
            simple_first=True):
        """
        Return a string representation of a tree.

        :arg leaf_str: The format string for the tree leaves. If 
          left empty, "%V (%^.2m%)" will be used for classification trees
          and "%V" for regression trees.
        :type leaf_str: string
        :arg node_str: The format string for the internal nodes.
          If left empty (as it is by default), no data is printed out for
          internal nodes. If set to ``"."``, the same string is
          used as for leaves.
        :type node_str: string
        :arg max_depth: If set, it limits the depth to which the tree is
          printed out.
        :type max_depth: integer
        :arg min_examples: If set, the subtrees with less than the given 
          number of examples are not printed.
        :type min_examples: integer
        :arg simple_first: If True (default), the branches with a single 
          node are printed before the branches with larger subtrees. 
          If False, the branches are printed in order of
          appearance.
        :type simple_first: boolean
        :arg user_formats: A list of regular expressions and callback 
          function through which the user can print out other specific 
          information in the nodes.
        """
        return _TreeDumper(leaf_str, node_str, user_formats + 
            _TreeDumper.defaultStringFormats, min_examples, 
            max_depth, simple_first, self).dumpTree()

    @Orange.misc.deprecated_keywords({"fileName": "file_name", \
        "leafStr": "leaf_str", "nodeStr": "node_str", \
        "leafShape": "leaf_shape", "nodeShape": "node_shape", \
        "userFormats": "user_formats", "minExamples": "min_examples", \
        "maxDepth": "max_depth", "simpleFirst": "simple_first"})
    def dot(self, fileName, leaf_str = "", node_str = "", \
            leaf_shape="plaintext", node_shape="plaintext", \
            user_formats=[], min_examples=0, max_depth=1e10, \
            simple_first=True):
        """ Print the tree to a file in a format used by `GraphViz
        <http://www.research.att.com/sw/tools/graphviz>`_.  Uses the
        same parameters as :meth:`dump` plus two which define the shape
        of internal nodes and leaves of the tree:

        :param leaf_shape: Shape of the outline around leaves of the tree. 
            If "plaintext", no outline is used (default: "plaintext").
        :type leaf_shape: string
        :param node_shape: Shape of the outline around internal nodes 
            of the tree. If "plaintext", no outline is used (default: "plaintext")
        :type node_shape: string

        Check `Polygon-based Nodes <http://www.graphviz.org/doc/info/shapes.html>`_ 
        for various outlines supported by GraphViz.
        """
        fle = type(fileName) == str and open(fileName, "wt") or fileName

        _TreeDumper(leaf_str, node_str, user_formats + 
            _TreeDumper.defaultStringFormats, min_examples, 
            max_depth, simple_first, self,
            leafShape=leaf_shape, nodeShape=node_shape, fle=fle).dotTree()

    def count_nodes(self):
        """
        Return the number of nodes.
        """
        return _countNodes(self.tree)

    def count_leaves(self):
        """
        Return the number of leaves.
        """
        return _countLeaves(self.tree)

def _countNodes(node):
    count = 0
    if node:
        count += 1
        if node.branches:
            for node in node.branches:
                count += _countNodes(node)
    return count

def _countLeaves(node):
    count = 0
    if node:
        if node.branches: # internal node
            for node in node.branches:
                count += _countLeaves(node)
        else:
            count += 1
    return count

