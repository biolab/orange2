.. py:currentmodule:: Orange.classification.tree

.. index:: classification tree

.. index::
   single: classification; tree

*******************************
Classification trees (``tree``)
*******************************

Orange includes multiple implementations of classification tree learners:
a very flexible :class:`TreeLearner`, a fast :class:`SimpleTreeLearner`,
and a :class:`C45Learner`, which uses the C4.5 tree induction
algorithm.

The following code builds a :obj:`TreeClassifier` on the Iris data set
(with the depth limited to three levels):

.. literalinclude:: code/orngTree1.py
   :lines: 1-4

See `Decision tree learning
<http://en.wikipedia.org/wiki/Decision_tree_learning>`_ on Wikipedia
for introduction to classification trees.

==============================
Component-based Tree Inducer
==============================

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

    .. attribute:: instances, weightID

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

--------
Examples
--------

Tree Structure
==============

This example works with the lenses data set:

    >>> import Orange
    >>> lenses = Orange.data.Table("lenses")
    >>> tree_classifier = Orange.classification.tree.TreeLearner(lenses)

The following function counts the number of nodes in a tree:

    >>> def tree_size(node):
    ...    if not node:
    ...        return 0
    ...    size = 1
    ...    if node.branch_selector:
    ...        for branch in node.branches:
    ...            size += tree_size(branch)
    ...    return size

If node is None, the function above return 0. Otherwise, the size is 1
(this node) plus the sizes of all subtrees. The algorithm need to check
if a node is internal (it has a :obj:`~Node.branch_selector`), as leaves
don't have the :obj:`~Node.branches` attribute.

    >>> tree_size(tree_classifier.tree)
    15

Note that a :obj:`Node` already has a built-in method
:func:`~Node.tree_size`.

Trees can be printed with a simple recursive function:

    >>> def print_tree0(node, level):
    ...     if not node:
    ...         print " "*level + "<null node>"
    ...         return
    ...     if node.branch_selector:
    ...         node_desc = node.branch_selector.class_var.name
    ...         node_cont = node.distribution
    ...         print "\\n" + "   "*level + "%s (%s)" % (node_desc, node_cont),
    ...         for i in range(len(node.branches)):
    ...             print "\\n" + "   "*level + ": %s" % node.branch_descriptions[i],
    ...             print_tree0(node.branches[i], level+1)
    ...     else:
    ...         node_cont = node.distribution
    ...         major_class = node.node_classifier.default_value
    ...         print "--> %s (%s) " % (major_class, node_cont),

The crux of the example is not in the formatting (\\n's etc.);
what matters is everything but the print statements. The code
separately handles three node types:

* For null nodes (a node to which no learning instances were classified),
  it just prints "<null node>".
* For internal nodes, it prints a node description:
  the feature's name and distribution of classes. :obj:`Node`'s
  branch description is an :obj:`~Orange.classification.Classifier`,
  and its ``class_var`` is the feature whose name is printed.  Class
  distributions are printed as well (they are assumed to be stored).
  The :obj:`print_tree0` with a level increased by 1 to increase the
  indent is recursively called for each branch.
* If the node is a leaf, it prints the distribution of learning instances
  in the node and the class to which the instances in the node would
  be classified. We assume that the :obj:`~Node.node_classifier` is a
  :obj:`DefaultClassifier`. A better print function should be aware of
  possible alternatives.

The wrapper function that accepts either a
:obj:`TreeClassifier` or a :obj:`Node` can be written as follows:

    >>> def print_tree(x):
    ...     if isinstance(x, Orange.classification.tree.TreeClassifier):
    ...         print_tree0(x.tree, 0)
    ...     elif isinstance(x, Orange.classification.tree.Node):
    ...         print_tree0(x, 0)
    ...     else:
    ...         raise TypeError, "invalid parameter"

It's straightforward: if ``x`` is a
:obj:`TreeClassifier`, it prints ``x.tree``; if it's :obj:`Node` it
print ``x``. If it's of some other type,
an exception is raised. The output:

    >>> print_tree(tree_classifier)
    <BLANKLINE>
    tear_rate (<15.000, 4.000, 5.000>) 
    : normal 
       astigmatic (<3.000, 4.000, 5.000>) 
       : no 
          age (<1.000, 0.000, 5.000>) 
          : pre-presbyopic --> soft (<0.000, 0.000, 2.000>)  
          : presbyopic 
             prescription (<1.000, 0.000, 1.000>) 
             : hypermetrope --> soft (<0.000, 0.000, 1.000>)  
             : myope --> none (<1.000, 0.000, 0.000>)  
          : young --> soft (<0.000, 0.000, 2.000>)  
       : yes 
          prescription (<2.000, 4.000, 0.000>) 
          : hypermetrope 
             age (<2.000, 1.000, 0.000>) 
             : pre-presbyopic --> none (<1.000, 0.000, 0.000>)  
             : presbyopic --> none (<1.000, 0.000, 0.000>)  
             : young --> hard (<0.000, 1.000, 0.000>)  
          : myope --> hard (<0.000, 3.000, 0.000>)  
    : reduced --> none (<12.000, 0.000, 0.000>) 

The tree structure examples conclude with a simple pruning function,
written entirely in Python and unrelated to any :class:`Pruner`. It limits
the tree depth (the number of internal nodes on any path down the tree).
For example, to get a two-level tree, call cut_tree(root, 2). The function
is recursive, with the second argument (level) decreasing at each call;
when zero, the current node will be made a leaf:

    >>> def cut_tree(node, level):
    ...     if node and node.branch_selector:
    ...         if level:
    ...             for branch in node.branches:
    ...                 cut_tree(branch, level-1)
    ...         else:
    ...             node.branch_selector = None
    ...             node.branches = None
    ...             node.branch_descriptions = None

The function acts only when :obj:`node` and :obj:`node.branch_selector`
are defined. If the level is not zero, is recursively calls  the function
for each branch. Otherwise, it clears the selector, branches and branch
descriptions.

    >>> cut_tree(tree_classifier.tree, 2)
    >>> print_tree(tree_classifier)
    <BLANKLINE>
    tear_rate (<15.000, 4.000, 5.000>) 
    : normal 
       astigmatic (<3.000, 4.000, 5.000>) 
       : no --> soft (<1.000, 0.000, 5.000>)  
       : yes --> hard (<2.000, 4.000, 0.000>)  
    : reduced --> none (<12.000, 0.000, 0.000>) 

Setting learning parameters
===========================

Let us construct a :obj:`TreeLearner` to play with:

    >>> import Orange
    >>> lenses = Orange.data.Table("lenses")
    >>> learner = Orange.classification.tree.TreeLearner()

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

    >>> learner.stop.min_examples = 5.0
    >>> tree = learner(lenses)
    >>> print tree
    tear_rate=reduced: none (100.00%)
    tear_rate=normal
    |    astigmatic=no
    |    |    age=pre-presbyopic: soft (100.00%)
    |    |    age=presbyopic: none (50.00%)
    |    |    age=young: soft (100.00%)
    |    astigmatic=yes
    |    |    prescription=hypermetrope: none (66.67%)
    |    |    prescription=myope: hard (100.00%)
    <BLANKLINE>

We can also limit the maximal proportion of majority class.

    >>> learner.stop.max_majority = 0.5
    >>> tree = learner(lenses)
    >>> print tree
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

.. literalinclude:: code/tree3.py
   :lines: 8-23

---------------------------------
Learner and Classifier Components
---------------------------------

Split constructors
=====================

.. class:: SplitConstructor

    Decide how to divide learning instances, ie. define branching criteria.
    
    The :obj:`SplitConstructor` should use the domain
    contingency when possible, both for speed and adaptability. 
    Sometimes domain contingency does
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
    ..., <valn>]``. Only binary features are marked as spent.

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
branch for an instance during classification).

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
    is a :obj:`~Orange.classification.ConstantClassifier`
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

The easiest way to print the tree is to print :func:`TreeClassifier`::

    >>> print tree
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

Printing the predicted class at each node, the number
of instances in the majority class with the total number of instances in
the node requires a custom format string::

    >>> print tree.to_string(leaf_str="%V (%M out of %N)")
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

    >>> print tree.to_string(leaf_str="%V (%^MbA%, %^MbP%)")
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

    tree.to_string(leaf_str="%V", node_str=".")
 
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

    print tree.to_string(leaf_str='%^.1CbA="Iris-virginica"% (%^.1CbP="Iris-virginica"%)', node_str='.')

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

If :meth:`~TreeClassifier.to_string` cannot compute something, in this case
because the root has no parent, it prints out a dot.

The final example with classification trees prints the distributions in
nodes, the distribution compared to the parent, the proportions compared
to the parent and the predicted class in the leaves::

    >>> print tree.to_string(leaf_str='"%V   %D %.2DbP %.2dbP"', node_str='"%D %.2DbP %.2dbP"')
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
set. Without other argumets, :meth:`TreeClassifier.to_string` prints the
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

    >>> print tree.to_string(leaf_str="[SE: %E]\t %V %I(90)", node_str="[SE: %E]")
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
:obj:`Orange.feature.Continuous`' function for printing the
value, the number has the same number of decimals as in the data file.

Regression trees cannot print the distributions in the same way
as classification trees. They instead offer a set of operators for
observing the number of instances within a certain range. For instance,
to print the number of instances with values below 22 and compare
it with values in the parent nodes use::

    >>> print tree.to_string(leaf_str="%C<22 (%cbP<22)", node_str=".")
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

    >>> print tree.to_string(leaf_str="%C![20,22] (%^cbP![20,22]%)", node_str=".")

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

:meth:`TreeClassifier.to_string`'s argument :obj:`user_formats` can be used to
print other information.  :obj:`~TreeClassifier.format.user_formats` should
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
C4.5 Tree Inducer
===========================

C4.5 is, as  a standard benchmark in machine learning, incorporated in
Orange. The implementation uses the original C4.5 code, so the resulting
tree is exactly like the one that would be build by standalone C4.5. The
tree build is only made accessible in Python.

:class:`C45Learner` and :class:`C45Classifier` behave
like any other Orange learner and classifier. Unlike most of Orange 
learning algorithms, C4.5 does not accepts weighted instances.

-------------------------
Building the C4.5 plug-in
-------------------------

Due to copyright restrictions, C4.5 is not distributed with Orange,
but it can be added as a plug-in. A C compiler is needed for the
procedure: on Windows MS Visual C (CL.EXE and LINK.EXE must be on the
PATH), on Linux and OS X gcc (OS X users can download it from Apple).

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

--------
Examples
--------

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
argument to the recursive part of print_tree.

For discrete splits without subsetting, we print out all attribute values
and recursively call the function for all branches. Continuous splits
are equally easy to handle.

For discrete splits with subsetting, we iterate through branches,
retrieve the corresponding values that go into each branch to inset,
turn the values into strings and print them out, separately treating
the case when only a single value goes into the branch.

===================
Simple Tree Inducer
===================

.. include:: /SimpleTreeLearner.txt

--------        
Examples
--------

:obj:`SimpleTreeLearner` is used in much the same way as :obj:`TreeLearner`.
A typical example of using :obj:`SimpleTreeLearner` would be to build a random
forest:

.. literalinclude:: code/simple_tree_random_forest.py

==========
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
