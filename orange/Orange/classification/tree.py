"""



This page describes the Orange trees. It first describes the basic components and procedures: it starts with <A href="#structure">the structure</A> that represents the tree, then it defines <A href="#classification">how the tree is used for classification</A>, then <A href="#learning">how it is built</A> and <a href="#pruning">pruned</A>. The order might seem strange, but the things are rather complex and this order is perhaps a bit easier to follow. After you have some idea about what the principal components do, we described the <a href="#classes">concrete classes</A> that you can use as components for a tree learner.

Classification trees are represented as a tree-like hierarchy of :obj:`TreeNode` classes.


.. class:: TreeNode

    TreeNode stores information about the learning examples belonging 
    to the node, a branch selector, a list of branches (if the node is 
    not a leaf) with their descriptions and strengths, and a classifier.

    .. attribute:: distribution
    
        Stores a distribution for learning examples belonging to the node.
        Storing distributions can be disabled by setting the 
        :obj:`TreeLearner`'s storeDistributions flag to false.

    .. attribute:: contingency

        Stores complete contingency matrices for the learning examples 
        belonging to the node. Storing contingencies can be enabled by 
        setting :obj:`TreeLearner`'s :obj:`storeContingencies` 
        flag to <CODE>true</CODE>. Note that even when the flag is not 
        set, the contingencies get computed and stored to 
        :obj:`TreeNone`, but are removed shortly afterwards. 
        The details are given in the 
        description of the :obj:`TreeLearner`object.

    .. attribute:: examples, weightID

        Store a set of learning examples for the node and the
        corresponding ID of /weight meta attribute. The root of the
        tree stores a "master table" of examples, while other nodes'
        :obj:`orange.ExampleTable` contain reference to examples in
        the root's :obj:`orange.ExampleTable`. Examples are only stored
        if a corresponding flag (:obj:`storeExamples`) has been
        set while building the tree; to conserve the space, storing
        is disabled by default.

    .. attribute:: nodeClassifier

        A classifier (usually, but not necessarily, a
        :obj:`DefaultClassifier`) that can be used to classify
        examples coming to the node. If the node is a leaf, this is
        used to decide the final class (or class distribution) of an
        example. If it's an internal node, it is stored if
        :obj:`TreeNode`'s flag :obj:`storeNodeClassifier`
        is set. Since the :obj:`nodeClassifier` is needed by
        :obj:`TreeDescender` and for pruning (see far below),
        this is the default behaviour; space consumption of the default
        :obj:`DefaultClassifier` is rather small. You should
        never disable this if you intend to prune the tree later.

    If the node is a leaf, the remaining fields are <code>None</code>. If it's an internal node, there are several additional fields.

    .. attribute:: branches

        Stores a list of subtrees, given as :obj:`TreeNode`.
        An element can be <code>None</code>; in this case the node is empty.

    .. attribute:: branchDescriptions

        A list with string descriptions for branches, constructed by
        :obj:`TreeSplitConstructor`. It can contain different kinds
        of descriptions, but basically, expect things like 'red' or '>12.3'.

    .. attribute:: branchSizes

        Gives a (weighted) number of training examples that went into
        each branch. This can be used later, for instance, for
        modeling probabilities when classifying examples with
        unknown values.

    .. attribute:: branchSelector

        Gives a branch for each example. The same object is used during
        learning and classifying. The :obj:`branchSelector` is of
        type :obj:`orange.Classifier`, since its job is similar to that
        of a classifier: it gets an example and returns discrete
        :obj:`orange.Value` in range [0, <CODE>len(branches)-1</CODE>].
        When an example cannot be classified to any branch, the selector
        can return a :obj:`orange.Value` containing a special value
        (<code>sVal</code>) which should be a discrete distribution
        (<code>DiscDistribution</code>). This should represent a
        :obj:`branchSelector`'s opinion of how to divide the
        example between the branches. Whether the proposition will be
        used or not depends upon the chosen :obj:`TreeExampleSplitter`
        (when learning) or :obj:`TreeDescender` (when classifying).

    The lists :obj:`branches`, :obj:`branchDescriptions` and :obj:`branchSizes` are of the same length; all of them are defined if the node is internal and none if it is a leaf.

    .. method:: treeSize()
        
        Return the number of nodes in the subtrees (including the node, excluding null-nodes).

==============
Classification
==============

.. class:: TreeClassifier

    Classifies examples according to a tree stored in :obj:`tree`.

    Classification would be straightforward if there were no unknown 
    values or, in general, examples that cannot be placed into a 
    single branch. The response in such cases is determined by a
    component :obj:`descender`.

    :obj:`TreeDescender` is an abstract object which is given an example
    and whose basic job is to descend as far down the tree as possible,
    according to the values of example's attributes. The
    :obj:`TreeDescender`: calls the node's :obj:`branchSelector` to get 
    the branch index. If it's a simple index, the corresponding branch 
    is followed. If not, it's up to descender to decide what to do, and
    that's where descenders differ. A :obj:`descender` can choose 
    a single branch (for instance, the one that is the most recommended 
    by the :obj:`branchSelector`) or it can let the branches vote.

    In general there are three possible outcomes of a descent.

    #. Descender reaches a leaf. This happens when nothing went wrong 
       (there are no unknown or out-of-range values in the example) or 
       when things went wrong, but the descender smoothed them by 
       selecting a single branch and continued the descend. In this
       case, the descender returns the reached :obj:`TreeNode`.
    #. :obj:`branchSelector` returned a distribution and the 
       :obj:`TreeDescender` decided to stop the descend at this 
       (internal) node.  Again, descender returns the current 
       :obj:`TreeNode` and nothing else.
    #. :obj:`branchSelector` returned a distribution and the 
       :obj:`TreeNode` wants to split the example (i.e., to decide the 
       class by voting). 

    It returns a :obj:`TreeNode` and the vote-weights for the branches. 
    The weights can correspond to the distribution returned by
    :obj:`branchSelector`, to the number of learning examples that
    were assigned to each branch, or to something else.

    :obj:`TreeClassifier` uses the descender to descend from the root. 
    If it returns only a :obj:`TreeNode` and no distribution, the 
    descend should stop; it does not matter whether it's a leaf (the
    first case above) or an internal node (the second case). The node's
    :obj:`nodeClassifier` is used to decide the class. If the descender
    returns a :obj:`TreeNode` and a distribution, the :obj:`TreeClassifier`
    recursively calls itself for each of the subtrees and the 
    predictions are weighted as requested by the descender.

    When voting, subtrees do not predict the class but probabilities 
    of classes. The predictions are multiplied by weights, summed and 
    the most probable class is returned.



The rest of this section is only for those interested in the C++ code.
======================================================================

If you'd like to understand how the classification works in C++, 
start reading at <code>TTreeClassifier::vote</code>. It gets a 
:obj:`TreeNode`, an :obj:`orange.Example`> and a distribution of 
vote weights. For each node, it calls the 
<code>TTreeClassifier::classDistribution</code> and then multiplies 
and sums the distribution. <code>vote</code> returns a normalized 
distribution of predictions.

A new overload of <code>TTreeClassifier::classDistribution</code> gets
an additional parameter, a :obj:`TreeNode`. This is done 
for the sake of recursion. The normal version of 
<code>classDistribution</code> simply calls the overloaded with a 
tree root as an additional parameter. <code>classDistribution</code> 
uses <code>descender</code>. If descender reaches a leaf, it calls 
:obj:`nodeClassifier`, otherwise it calls <CODE>vote</CODE>.

Thus, the :obj:`TreeClassifier`'s <code>vote</code> and 
<code>classDistribution</code> are written in a form of double 
recursion. The recursive calls do not happen at each node of the 
tree but only at nodes where a vote is needed (that is, at nodes 
where the descender halts).

For predicting a class, <code>operator()</code>, calls the
descender. If it reaches a leaf, the class is predicted by the 
leaf's :obj:`nodeClassifier`. Otherwise, it calls 
<code>vote</code>. From now on, <code>vote</code> and 
<code>classDistribution</code> interweave down the tree and return 
a distribution of predictions. <code>operator()</code> then simply 
chooses the most probable class.


========
Learning
========

The main learning object is :obj:`TreeLearner`. It is basically 
a skeleton into which the user must plug the components for particular 
functions. For easier use, defaults are provided.

<p>Components that govern the structure of the tree are <code>split</code> 
(of type :obj:`TreeSplitConstructor`), <code>stop</code> (of 
type :obj:`TreeStopCriteria` and <code>exampleSplitter</code> 
(of type :obj:`TreeExampleSplitter`).</p>

.. class:: TreeSplitConstructor

    Finds a suitable criteria for dividing the learning (and later testing)
    examples coming to the node. The data it gets is a set of examples
    (and, optionally, an ID of weight meta-attribute), a domain
    contingency computed from examples, apriori class probabilities,
    a list of candidate attributes it should consider and a node
    classifier (if it was constructed, that is, if 
    :obj:`storeNodeClassifier` is left true).

    The :obj:`TreeSplitConstructor` should use the domain contingency
    when possible. The reasons are two-fold; one is that it's faster
    and the other is that the contingency matrices are not necessarily
    constructed by simply counting the examples. Why and how is
    explained later. There are, however, cases, when domain contingency
    does not suffice, for examples, when ReliefF is used as a measure
    of quality of attributes. In this case, there's no other way but
    to use the examples and ignore the precomputed contingencies.

    The split constructor should consider only the attributes in the
    candidate list (the list is a vector of booleans, one for each
    attribute).

    :obj:`TreeSplitConstructor` returns most of the data we talked
    about when describing the :obj:`TreeNode`. It returns a classifier
    to be used as :obj:`TreeNode`'s :obj:`branchSelector`, a list of
    branch descriptions and a list with the number of examples that
    go into each branch. Just what we need for the :obj:`TreeNode`.
    It can return an empty list for the number of examples in branches;
    in this case, the :obj:`TreeLearner` will find the number itself
    after splitting the example set into subsets. However, if a split
    constructors can provide the numbers at no extra computational
    cost, it should do so.

    In addition, it returns a quality of the split; a number without
    any fixed meaning except that higher numbers mean better splits.

    If the constructed splitting criterion uses an attribute in such
    a way that the attribute is 'completely spent' and should not be
    considered as a split criterion in any of the subtrees (the
    typical case of this are discrete attributes that are used
    as-they-are, that is, without any binarization or subsetting),
    then it should report the index of this attribute. Some splits
    do not spend any attribute; this is indicated by returning a
    negative index.

    A :obj:`TreeSplitConstructor` can veto the further tree induction
    by returning no classifier. This can happen for many reasons.
    A general one is related to number of examples in the branches.
    :obj:`TreeSplitConstructor` has a field <code>minSubset</code>,
    which sets the minimal number of examples in a branch; null nodes,
    however, are allowed. If there is no split where this condition
    is met, :obj:`TreeSplitConstructor` stops the induction.

    .. attribute:: minSubset

        Sets the minimal number of examples in non-null leaves. As
        always in Orange (where not specified otherwise), "number of 
        examples" refers to the weighted number of examples.
    
    .. method:: test()

        Bla bla.

    .. method:: __call__(examples, [weightID=0, apriori_distribution, candidates]) 

        Construct a split. Returns a tuple (:obj:`branchSelector`,
        :obj:`branchDescriptions`, :obj:`subsetSizes`, :obj:`quality`,
        :obj:`spentAttribute`). :obj:`SpentAttribute` is -1 if no
        attribute is completely spent by the split criterion. If no
        split is constructed, the :obj:`selector`, :obj:`branchDescriptions`
        and :obj:`subsetSizes` are None, while :obj:`quality` is 0.0 and
        :obj:`spentAttribute` is -1.

        :param examples:  Examples can be given in any acceptable form
            (an :obj:`ExampleGenerator`, such as :obj:`ExampleTable`, or a
            list of examples).
        :param weightID: Optional; the default of 0 means that all
            examples have a weight of 1.0. 
        :param apriori-distribution: Should be of type 
            :obj:`orange.Distribution` and candidates should be a Python 
            list of objects which are interpreted as booleans.


.. class:: TreeStopCriteria

    Given a set of examples, weight ID and contingency matrices, decide
    whether to continue the induction or not. The basic criterion checks
    whether there are any examples and whether they belong to at least
    two different classes (if the class is discrete). Derived components
    check things like the number of examples and the proportion of
    majority classes.


.. class:: TreeExampleSplitter

    Analoguous to the :obj:`TreeDescender` described about a while ago. 
    Just like the :obj:`TreeDescender` decides the branch for an
    example during classification, the :obj:`TreeExampleSplitter`
    sorts the learning examples into branches.

    :obj:`TreeExampleSplitter` is given a :obj:`TreeNode` (from which 
    it can use different stuff, but most of splitters only use the 
    :obj:`branchSelector`), a set of examples to be divided, and 
    the weight ID. The result is a list of subsets of examples
    and, optionally, a list of new weight ID's.

    Subsets are usually stored as :obj:`ExamplePointerTable`'s. Most 
    of :obj:`TreeExampleSplitters` simply call the node's 
    :obj:`branchSelector` and assign examples to corresponding 
    branches. When the value is unknown they choose a particular 
    branch or simply skip the example.

    Some enhanced splitters can split examples. An example (actually, 
    a pointer to it) is copied to more than one subset. To facilitate
    real splitting, weights are needed. Each branch is assigned a
    weight ID (each would usually have its own ID) and all examples
    that are in that branch (either completely or partially) should
    have this meta attribute. If an example hasn't been split, it
    has only one additional attribute - with weight ID corresponding
    to the subset to which it went. Example that is split between,
    say, three subsets, has three new meta attributes, one for each
    subset. ID's of weight meta attributes are returned by the
    :obj:`TreeExampleSplitter` to be used at induction of the
    corresponding subtrees.

    Note that weights are used only when needed. When no splitting
    occured - because the splitter is not able to do it or becauser
    there was no need for splitting - no weight ID's are returned.

.. class:: TreeLearner

    TreeLearner has a number of components.

    .. attribute:: split

        Object of type :obj:`TreeSplitConstructor`. Default value, 
        provided by :obj:`TreeLearner`, is :obj:`SplitConstructor_Combined`
        with separate constructors for discrete and continuous attributes. 
        Discrete attributes are used as are, while continuous attributes
        are binarized. Gain ratio is used to select attributes.
        A minimum of two examples in a leaf is required for discreter
        and five examples in a leaf for continuous attributes.</DD>
    
    .. attribute:: stop

        Object of type :obj:`TreeStopCriteria`. The default stopping
        criterion stops induction when all examples in a node belong 
        to the same class.

    .. attribute:: splitter

        Object of type :obj:`TreeExampleSplitter`. The default splitter
        is :obj:`TreeExampleSplitter_UnknownsAsSelector` that splits
        the learning examples according to distributions given by the
        selector.

    .. attribute:: contingencyComputer
    
        By default, this slot is left empty and ordinary contingency
        matrices are computed for examples at each node. If need arises,
        one can change the way the matrices are computed. This can be
        used to change the way that unknown values are treated when
        assessing qualities of attributes. As mentioned earlier,
        the computed matrices can be used by split constructor and by
        stopping criteria. On the other hand, they can be (and are)
        ignored by some splitting constructors.

    .. attribute:: nodeLearner

        Induces a classifier from examples belonging to a node. The
        same learner is used for internal nodes and for leaves. The
        default :obj:`nodeLearner` is :obj:`MajorityLearner`.

    .. attribute:: descender

        Descending component that the induces :obj:`TreeClassifier`
        will use. Default descender is 
        :obj:`TreeDescender_UnknownMergeAsSelector` which votes using 
        the :obj:`branchSelector`'s distribution for vote weights.

    .. attribute:: maxDepth

        Gives maximal tree depth; 0 means that only root is generated. 
        The default is 100 to prevent any infinite tree induction due
        to missettings in stop criteria. If you are sure you need
        larger trees, increase it. If you, on the other hand, want
        to lower this hard limit, you can do so as well.

    .. attribute:: storeDistributions, storeContingencies, storeExamples, storeNodeClassifier

        Decides whether to store class distributions, contingencies 
        and examples in :obj:`TreeNode`, and whether the 
        :obj:`nodeClassifier` should be build for internal nodes. 
        By default, distributions and node classifiers are stored, 
        while contingencies and examples are not. You won't save any 
        memory by not storing distributions but storing contingencies,
        since distributions actually points to the same distribution
        that is stored in <code>contingency.classes</code>.

    The :obj:`TreeLearner` first sets the defaults for missing
    components. Although stored in the actual :obj:`TreeLearner`'s
    fields, they are removed when the induction is finished.

    Then it ensures that examples are stored in a table. This is needed
    because the algorithm juggles with pointers to examples. If
    examples are in a file or are fed through a filter, they are copied
    to a table. Even if they are already in a table, they are copied
    if :obj:`storeExamples` is set. This is to assure that pointers
    remain pointing to examples even if the user later changes the
    example table. If they are in the table and the :obj:`storeExamples`
    flag is clear, we just use them as they are. This will obviously
    crash in a multi-threaded system if one changes the table during
    the tree induction. Well... don't do it.

    Apriori class probabilities are computed. At this point we check
    the sum of example weights; if it's zero, there are no examples and 
    we cannot proceed. A list of candidate attributes is set; in the
    beginning, all attributes are candidates for the split criterion.

    Now comes the recursive part of the :obj:`TreeLearner`. Its arguments 
    are a set of examples, a weight meta-attribute ID (a tricky thing,
    it can be always the same as the original or can change to 
    accomodate splitting of examples among branches), apriori class
    distribution and a list of candidates (represented as a vector
    of Boolean values).

    <code>Contingency matrix</code> is computed next. This happens
    even if the flag :obj:`storeContingencies` is <CODE>false</CODE>.
    If the <code>contingencyComputer</code> is given we use it,
    otherwise we construct just an ordinary contingency matrix.

    A :obj:`stop` is called to see whether it's worth to continue. If 
    not, a :obj:`nodeClassifier` is built and the :obj:`TreeNode` is 
    returned. Otherwise, a :obj:`nodeClassifier` is only built if 
    :obj:`forceNodeClassifier` flag is set.

    To get a :obj:`TreeNode`'s :obj:`nodeClassifier`, the 
    :obj:`nodeLearner`'s :obj:`smartLearn` function is called with 
    the given examples, weight ID and the just computed matrix. If 
    the learner can use the matrix (and the default, 
    :obj:`MajorityLearner`, can), it won't touch the examples. Thus,
    a choice of :obj:`contingencyComputer` will, in many cases, 
    affect the :obj:`nodeClassifier`. The :obj:`nodeLearner` can
    return no classifier; if so and if the classifier would be 
    needed for classification, the :obj:`TreeClassifier`'s function
    returns DK or an empty distribution. If you're writing your own
    tree classifier - pay attention.

    If the induction is to continue, a :obj:`split` component is called. 
    If it fails to return a branch selector, induction stops and the 
    :obj:`TreeNode` is returned.

    :obj:`TreeLearner` than uses :obj:`ExampleSplitter` to divide 
    the examples as described above.

    The contingency gets removed at this point if it is not to be 
    stored. Thus, the :obj:`split`, :obj:`stop` and 
    :obj:`exampleSplitter` can use the contingency matrices if they will.

    The :obj:`TreeLearner` then recursively calls itself for each of 
    the non-empty subsets. If the splitter returnes a list of weights, 
    a corresponding weight is used for each branch. Besides, the 
    attribute spent by the splitter (if any) is removed from the 
    list of candidates for the subtree.

    A subset of examples is stored in its corresponding tree node, 
    if so requested. If not, the new weight attributes are removed (if 
    any were created).

Pruning
=======

Tree pruners derived from :obj:`TreePrune` can be given either a
:obj:`TreeNode` (presumably, but not necessarily a root) or a
:obj:`TreeClassifier`. The result is a new, pruned :obj:`TreeNode`
or a new :obj:`TreeClassifier` with a pruned tree. The original
tree remains intact.

Note however that pruners construct only a shallow copy of a tree.
The pruned tree's :obj:`TreeNode` contain references to the same
contingency matrices, node classifiers, branch selectors, ...
as the original tree. Thus, you may modify a pruned tree structure
(manually cut it, add new nodes, replace components) but modifying,
for instance, some node's :obj:`nodeClassifier` (a
:obj:`nodeClassifier` itself, not a reference to it!) would modify
the node's :obj:`nodeClassifier` in the corresponding node of
the original tree.

Talking about node classifiers - pruners cannot construct a
:obj:`nodeClassifier` nor merge :obj:`nodeClassifier` of the pruned
subtrees into classifiers for new leaves. Thus, if you want to build
a prunable tree, internal nodes must have their :obj:`nodeClassifier`
defined. Fortunately, all you need to do is nothing; if you leave
the :obj:`TreeLearner`'s flags as they are by default, the
:obj:`nodeClassifier` are created.

=======
Classes
=======

Several classes described above are already functional and can
(and mostly will) be used as they are. Those classes are :obj:`TreeNode`,
:obj:`TreeLearner` and :obj:`TreeClassifier`. This section describe 
the other classes.

Classes :obj:`TreeSplitConstructor`, :obj:`TreeStopCriteria`, 
:obj:`TreeExampleSplitter`, :obj:`TreeDescender`, :obj:`orange.Learner`
and :obj:`Classifier` are among the Orange classes that can be subtyped 
in Python and have the call operator overloadedd in such a way that it
is callbacked from C++ code. You can thus program your own components
for :obj:`orange.TreeLearner` and :obj:`TreeClassifier`. The detailed 
information on how this is done and what can go wrong, is given in a 
separate page, dedicated to callbacks to Python XXXXXXXXXX.

TreeSplitConstructor and Derived Classes
========================================

Split construction is almost as exciting as waiting for a delayed flight.
Boring, that is. Split constructors are programmed as spaghetti code
that juggles with contingency matrices, with separate cases for discrete
and continuous classes... Most split constructors work either for
discrete or for continuous attributes. The suggested practice is
to use a :obj:`TreeSplitConstructor_Combined` that can handle
both by simply delegating attributes to specialized split constructors.

Note: split constructors that cannot handle attributes of particular
type (discrete, continuous) do not report an error or a warning but
simply skip the attribute. It is your responsibility to use a correct
split constructor for your dataset. (May we again suggest
using :obj:`TreeSplitConstructor_Combined`?)

The same components can be used either for inducing classification and
regression trees. The only component that needs to be chosen accordingly
is the 'measure' attribute for the :obj:`TreeSplitConstructor_Measure`
class (and derived classes).

TreeSplitConstructor
====================

.. class:: TreeSplitConstructor_Measure

    An abstract base class for split constructors that employ 
    a :obj:`orange.MeasureAttribute` to assess a quality of a split. At present,
    all split constructors except for :obj:`TreeSplitConstructor_Combined`
    are derived from this class.

    .. attribute:: measure

        A component of type :obj:`orange.MeasureAttribute` used for
        evaluation of a split. Note that you must select the subclass 
        :obj:`MeasureAttribute` capable of handling your class type 
        - you cannot use :obj:`orange.MeasureAttribute_gainRatio`
        for building regression trees or :obj:`orange.MeasureAttribute_MSE`
        for classification trees.

    .. attribute:: worstAcceptable

        The lowest required split quality for a split to be acceptable.
        Note that this value make sense only in connection with a
        :obj:`measure` component. Default is 0.0.

.. class:: TreeSplitConstructor_Attribute

    Attempts to use a discrete attribute as a split; each value of the 
    attribute corresponds to a branch in the tree. Attributes are
    evaluated with the :obj:`measure` and the one with the
    highest score is used for a split. If there is more than one
    attribute with the highest score, one of them is selected by random.

    The constructed :obj:`branchSelector` is an instance of 
    :obj:`orange.ClassifierFromVarFD` that returns a value of the 
    selected attribute. If the attribute is :obj:`orange.EnumVariable`,
    :obj:`branchDescription`'s are the attribute's values. The 
    attribute is marked as spent, so that it cannot reappear in the 
    node's subtrees.

.. class:: TreeSplitConstructor_ExhaustiveBinary

    Works on discrete attributes. For each attribute, it determines
    which binarization of the attribute gives the split with the
    highest score. If more than one split has the highest score,
    one of them is selected by random. After trying all the attributes,
    it returns one of those with the highest score.

    The constructed :obj:`branchSelector` is again an instance
    :obj:`orange.ClassifierFromVarFD` that returns a value of the
    selected attribute. This time, however, its :obj:`transformer`
    contains an instance of :obj:`MapIntValue` that maps the values
    of the attribute into a binary attribute. Branch descriptions are
    of form "[<val1>, <val2>, ...<valn>]" for branches corresponding to
    more than one value of the attribute. Branches that correspond to
    a single value of the attribute are described with this value. If 
    the attribute was originally binary, it is spent and cannot be 
    used in the node's subtrees. Otherwise, it can reappear in the 
    subtrees.


.. class:: TreeSplitConstructor_Threshold

    This is currently the only constructor for splits with continuous 
    attributes. It divides the range of attributes values with a threshold 
    that maximizes the split's quality. As always, if there is more than 
    one split with the highest score, a random threshold is selected. 
    The attribute that yields the highest binary split is returned.

    The constructed :obj:`branchSelector` is again an instance of 
    :obj:`orange.ClassifierFromVarFD` with an attached 
    :obj:`transformer`. This time, :obj:`transformer` is of type 
    :obj:`orange.ThresholdDiscretizer`. The branch descriptions are 
    "<threshold" and ">=threshold". The attribute is not spent.

.. class:: TreeSplitConstructor_Combined

    This constructor delegates the task of finding the optimal split 
    to separate split constructors for discrete and for continuous
    attributes. Each split constructor is called, given only attributes
    of appropriate types as candidates. Both construct a candidate for
    a split; the better of them is selected.

    (Note that there is a problem when more candidates have the same
    score. Let there be are nine discrete attributes with the highest
    score; the split constructor for discrete attributes will select
    one of them. Now, let us suppose that there is a single continuous
    attribute with the same score. :obj:`TreeSplitConstructor_Combined`
    would randomly select between the proposed discrete attribute and 
    the continuous attribute, not aware of the fact that the discrete
    has already competed with eight other discrete attributes. So, 
    he probability for selecting (each) discrete attribute would be 1/18
    instead of 1/10. Although not really correct, we doubt that this
    would affect the tree's performance; many other machine learning
    systems simply choose the first attribute with the highest score 
    anyway.)

    The :obj:`branchSelector`, :obj:`branchDescriptions` and whether 
    the attribute is spent is decided by the winning split constructor.

    .. attribute: discreteSplitConstructor

        Split constructor for discrete attributes; can be, for instance,
        :obj:`TreeSplitConstructor_Attribute` or 
        :obj:`TreeSplitConstructor_ExhaustiveBinary`.

    .. attribute: continuousSplitConstructor

        Split constructor for continuous attributes; at the moment, it 
        can be either :obj:`TreeSplitConstructor_Threshold` or a 
        split constructor you programmed in Python.

    .. attribute: continuousSplitConstructor
    
        Split constructor for continuous attributes; at the moment, it 
        can be either :obj:`TreeSplitConstructor_Threshold` or a split
        constructor you programmed in Python.

TreeStopCriteria and TreeStopCriteria_common
============================================

obj:`TreeStopCriteria` determines when to stop the induction of subtrees, as described in detail in description of the learning process. XXXXXXXXXX

.. class:: TreeStopCriteria

    As opposed to :obj:`TreeSplitConstructor` and similar basic classes,
    :obj:`TreeStopCriteria` is not an abstract but a fully functional
    class that provides the basic stopping criteria. That is, the tree
    induction stops when there is at most one example left; in this case,
    it is not the weighted but the actual number of examples that counts.
    Besides that, the induction stops when all examples are in the same
    class (for discrete problems) or have the same value of the outcome
    (for regression problems).

    .. method:: __call__(examples[, weightID, domain contingencies])

        Decides whether to stop (true) or continue (false) the induction.
        If contingencies are given, they are used for checking whether
        the examples are in the same class (but not for counting the
        examples). Derived classes should use the contingencies whenever
        possible. If contingencies are not given, :obj:`TreeStopCriteria`
        will work without them. Derived classes should also use them if
        they are available, but otherwise compute them only when they
        really need them.

.. class:: TreeStopCriteria_common

    :obj:`TreeStopCriteria` contains additional criteria for pre-pruning:
    it checks the proportion of majority class and the number of weighted
    examples.

    .. attribute:: maxMajor

        Maximal proportion of majority class. When this is exceeded, 
        induction stops.

    .. attribute:: minExamples

        Minimal number of examples in internal leaves. Subsets with less
        than :obj:`minExamples` examples are not split any further.
        Example count is weighed.


TreeExampleSplitter and derived classes
=======================================

:obj:`TreeExampleSplitter` is the third crucial component of
:obj:`TreeLearner`. Its function is described in 
description of the learning process. XXXXXXXXX

.. class:: TreeExampleSplitter

    An abstract base class for objects that split sets of examples into 
    subsets. The derived classes differ in treatment of examples which
    cannot be unambiguously placed into a single branch (usually due
    to unknown value of the crucial attribute).

    .. method:: __call__(node, examples[, weightID])
        
        Use the information in :obj:`node` (particularly the 
        :obj:`branchSelector`) to split the given set of examples into subsets. 
        Return a tuple with a list of example generators and a list of weights. 
        The list of weights is either an ordinary python list of integers or 
        a None when no splitting of examples occurs and thus no weights are 
        needed.

.. class:: TreeExampleSplitter_IgnoreUnknowns

    Simply ignores the examples for which no single branch can be determined.

.. class:: TreeExampleSplitter_UnknownsToCommon

    Places all such examples to a branch with the highest number of
    examples. If there is more than one such branch, one is selected at
    random and then used for all examples.

.. class:: TreeExampleSplitter_UnknownsToAll

    Places examples with unknown value of the attribute into all branches.

.. class:: TreeExampleSplitter_UnknownsToRandom

    Selects a random branch for such examples.

.. class:: TreeExampleSplitter_UnknownsToBranch

    Constructs an additional branch to contain all such examples. 
    The branch's description is "unknown".

.. class:: TreeExampleSplitter_UnknownsAsBranchSizes

    Splits examples with unknown value of the attribute according to 
    proportions of examples in each branch.

.. class:: TreeExampleSplitter_UnknownsAsSelector

    Splits examples with unknown value of the attribute according to 
    distribution proposed by selector (which is in most cases the same 
    as proportions of examples in branches).

TreeDescender and derived classes
=================================

This is a classifier's counterpart for :obj:`TreeExampleSplitter`. It 
decides the destiny of examples that need to be classified and cannot
be unambiguously put in a branch. The detailed function of this class
is given in description of classification with trees. XXXXXX

.. class:: TreeDescender

    An abstract base object for tree descenders.

    .. method:: __call__(node, example)

        Descends down the tree until it reaches a leaf or a node in 
        which a vote of subtrees is required. In both cases, a tuple 
        of two elements is returned; in the former, the tuple contains 
        the reached node and <CODE>None</CODE>, in the latter in 
        contains a node and weights of votes for subtrees (a list of floats).

        :obj:`TreeDescender`'s that never split examples always descend
        to a leaf, but they differ in the treatment of examples with
        unknown values (or, in general, examples for which a branch
        cannot be determined at some node(s) the tree).
        :obj:`TreeDescender`'s that do split examples differ in returned
        vote weights.

.. class:: TreeDescender_UnknownsToNode

    When example cannot be classified into a single branch, the
    current node is returned. Thus, the node's :obj:`NodeClassifier`
    will be used to make a decision. It is your responsibility to see
    that even the internal nodes have their :obj:`NodeClassifier`
    (i.e., don't disable creating node classifier or manually remove
    them after the induction, that's all)

.. class:: TreeDescender_UnknownsToCommon

    Classifies examples with unknown value to a special branch. This
    makes sense only if the tree itself was constructed with
    :obj:`TreeExampleSplitter_UnknownsToBranch`.

.. class:: TreeDescender_UnknownsToCommonBranch

    Classifies examples with unknown values to the branch with the
    highest number of examples. If there is more than one such branch,
    random branch is chosen for each example that is to be classified.

.. class:: TreeDescender_UnknownsToCommonSelector

    Classifies examples with unknown values to the branch which received 
    the highest recommendation by the selector.

.. class:: TreeDescender_MergeAsBranchSizes

    Makes the subtrees vote for the example's class; the vote is
    weighted according to the sizes of the branches.

.. class:: TreeDescender_MergeAsSelector

    Makes the subtrees vote for the example's class; the vote is 
    weighted according to the selectors proposal.

TreePruner and derived classes
==============================


.. index::
    pair: classification trees; pruning
.. index:: pruning classification trees

    Classes derived from :obj:`TreePruner` prune the trees as a
    described in the section pruning XXXXXXXX - make sure you read it 
    to understand what the pruners will do to your trees.

.. class:: TreePruner

    This is an abstract base class which defines nothing useful, only 
    a pure virtual call operator.

    .. method:: __call__(tree)

        Prunes a tree. The argument can be either a tree classifier or 
        a tree node; the result is of the same type as the argument.

.. class:: TreePruner_SameMajority

    In Orange, a tree can have a non-trivial subtrees (i.e. subtrees 
    with more than one leaf) in which all the leaves have the same majority 
    class. (This is allowed because those leaves can still have different
    distributions of classes and thus predict different probabilities.) 
    However, this can be undesired when we're only interested in the 
    class prediction or a simple tree interpretation. The 
    :obj:`TreePruner_SameMajority` prunes the tree so that there is no
    subtree in which all the nodes would have the same majority class.

    This pruner will only prune the nodes in which the node classifier 
    is of class :obj:`orange.DefaultClassifier` (or from a derived class).

    Note that the leaves with more than one majority class require some 
    special handling. The pruning goes backwards, from leaves to the root. 
    When siblings are compared, the algorithm checks whether they 
    have (at least one) common majority class. If so, they can be pruned.

.. class:: TreePruner_m

    Prunes a tree by comparing m-estimates of static and dynamic 
    error as defined in (Bratko, 2002).

    .. attributes:: m

        Parameter m for m-estimation.

========
Examples
========

This page does not provide examples for programming your own components, 
such as, for instance, a :obj:`TreeSplitConstructor`. Those examples
can be found on a page dedicated to callbacks to Python XXXXXXXX.

Tree Structure
==============

To have something to work on, we'll take the data from lenses dataset 
and build a tree using the default components (part of `treestructure.py`_, uses `lenses.tab`_):

.. literalinclude:: code/treestructure.py
   :lines: 7-10

How big is our tree (part of `treestructure.py`_, uses `lenses.tab`_)?

.. _lenses.tab: code/lenses.tab
.. _treestructure.py: code/treestructure.py

.. literalinclude:: code/treestructure.py
   :lines: 12-21

If node is None, we have a null-node; null nodes don't count, 
so we return 0. Otherwise, the size is 1 (this node) plus the
sizes of all subtrees. The node is an internal node if it has a 
:obj:`branchSelector`; it there's no selector, it's a leaf. Don't
attempt to skip the if statement: leaves don't have an empty list 
of branches, they don't have a list of branches at all.

    >>> treeSize(treeClassifier.tree)
    10

Don't forget that this was only an excercise - :obj:`TreeNode` has a 
built-in method :obj:`TreeNode.treeSize` that does exactly the same.

Let us now write a simple script that prints out a tree. The recursive
part of the function will get a node and its level (part of `treestructure.py`_, uses `lenses.tab`_).

.. literalinclude:: code/treestructure.py
   :lines: 26-41

Don't waste time on studying formatting tricks (\n's etc.), this is just
for nicer output. What matters is everything but the print statements.
As first, we check whether the node is a null-node (a node to which no
learning examples were classified). If this is so, we just print out
"<null node>" and return.

After handling null nodes, remaining nodes are internal nodes and leaves.
For internal nodes, we print a node description consisting of the
attribute's name and distribution of classes. :obj:`TreeNode`'s branch
description is, for all currently defined splits, an instance of a
class derived from :obj:`orange.Classifier` (in fact, it is a
:obj:`orange.ClassifierFromVarFD`, but a :obj:`orange.Classifier` would 
suffice), and its :obj:`classVar` XXXXX points to the attribute we seek. 
So we print its name. We will also assume that storing class distributions 
has not been disabled and print them as well. A more able function for 
printing trees (as one defined in orngTree XXXXXXXXXX) has an alternative 
means to get the distribution, when this fails. Then we iterate 
through branches; for each we print a branch description and iteratively 
call the :obj:`printTree0` with a level increased by 1 (to increase 
the indent).

Finally, if the node is a leaf, we print out the distribution of 
learning examples in the node and the class to which the examples in 
the node would be classified. We again assume that the :obj:`nodeClassifier` 
is the default one - a :obj:`DefaultClassifier`. A better print 
function should be aware of possible alternatives.

Now, we just need to write a simple function to call our printTree0. 
We could write something like...

    def printTree(x):
        printTree0(x.tree, 0)

... but we won't. Let us learn how to handle arguments of different
types. Let's write a function that will accept either a :obj:`TreeClassifier`
or a :obj:`TreeNode`; just like TreePruners XXXXXX, remember? Part of `treestructure.py`_, uses `lenses.tab`_.

.. literalinclude:: code/treestructure.py
   :lines: 43-49

It's fairly straightforward: if :obj:`x` is of type derived from 
:obj:`orange.TreeClassifier`, we print :obj:`x.tree`; if it's 
:obj:`TreeNode` we just call :obj:`printTree0` with :obj:`x`. If it's 
of some other type, we don't know how to handle it and thus raise 
an exception. (Note that we could also use 
::
    if type(x) == orange.TreeClassifier:

but this would only work if :obj:`x` would be of type 
:obj:`orange.TreeClassifier` and not of any derived types. The latter, 
however, do not exist yet...)

    >>> printTree(treeClassifier)
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

For a final exercise, let us write a simple pruning procedure. It will 
be written entirely in Python, unrelated to any :obj:`TreePruner`. Our
procedure will limit the tree depth - the maximal depth (here defined
as the number of internal nodes on any path down the tree) shall be
given as an argument. For example, to get a two-level tree, we would
call cutTree(root, 2). The function will be recursive, with the second 
argument (level) decreasing at each call; when zero, the current node 
will be made a leaf (part of `treestructure.py`_, uses `lenses.tab`_):

.. literalinclude:: code/treestructure.py
   :lines: 54-62

There's nothing to prune at null-nodes or leaves, so we act only when 
:obj:`node` and :obj:`node.branchSelector` are defined. If level is 
not zero, we call the function for each branch. Otherwise, we clear 
the selector, branches and branch descriptions.

    >>> cutTree(tree.tree, 2)
    >>> printTree(tree)
    tear_rate (<15.000, 5.000, 4.000>)
    : reduced --> none (<12.000, 0.000, 0.000>)
    : normal
       astigmatic (<3.000, 5.000, 4.000>)
       : no --> soft (<1.000, 5.000, 0.000>)
       : yes --> hard (<2.000, 0.000, 4.000>)

Learning
========

You've already seen a simple example of using a :obj:`TreeLearner`. You can just call it and let it fill the empty slots with the default components. This section will teach you three things: what are the missing components (and how to set the same components yourself), how to use alternative components to get a different tree and, finally, how to write a skeleton for tree induction in Python.</p>

<H4>Default components for TreeLearner</H4>

<p>Let us construct a :obj:`TreeLearner` to play with.</p>

<p class="header"><a href="treelearner.py">treelearner.py</a>
(uses <a href="lenses.tab">lenses.tab</a>)</p>
<xmp class="code">>>> learner = orange.TreeLearner()
</xmp>

<p>There are three crucial components in learning: the split and stop criteria, and the <code>exampleSplitter</code> (there are some others, which become important during classification; we'll talk about them later). They are not defined; if you use the learner, the slots are filled temporarily but later cleared again.</code>

<xmp class="code">>>> print learner.split
None
>>> learner(data)
<TreeClassifier instance at 0x01F08760>
>>> print learner.split
None
</xmp>

<H4>Stopping criteria</H4>
<p>The stop is trivial. The default is set by</p>

<xmp class="code">>>> learner.stop = orange.TreeStopCriteria_common()
</xmp>

<p>Well, this is actually done in C++ and it uses a global component that is constructed once for all, but apart from that we did effectively the same thing.</p>

<p>We can now examine the default stopping parameters.</p>

<xmp class="code">>>> print learner.stop.maxMajority, learner.stop.minExamples
1.0 0.0
</xmp>

<p>Not very restrictive. This keeps splitting the examples until there's nothing left to split or all the examples are in the same class. Let us set the minimal subset that we allow to be split to five examples and see what comes out.</p>

<p class="header">part of <a href="treelearner.py">treelearner.py</a>
(uses <a href="lenses.tab">lenses.tab</a>)</p>
<xmp class="code">>>> learner.stop.minExamples = 5.0
>>> tree = learner(data)
>>> printTree(tree)
tear_rate (<15.000, 5.000, 4.000>)
: reduced --> none (<12.000, 0.000, 0.000>)
: normal
   astigmatic (<3.000, 5.000, 4.000>)
   : no
      age (<1.000, 5.000, 0.000>)
      : young --> soft (<0.000, 2.000, 0.000>)
      : pre-presbyopic --> soft (<0.000, 2.000, 0.000>)
      : presbyopic --> soft (<1.000, 1.000, 0.000>)
   : yes
      prescription (<2.000, 0.000, 4.000>)
      : myope --> hard (<0.000, 0.000, 3.000>)
      : hypermetrope --> none (<2.000, 0.000, 1.000>)
</xmp>

<p>OK, that's better. If we want an even smaller tree, we can also limit the maximal proportion of majority class.</p>

<p class="header">part of <a href="treelearner.py">treelearner.py</a>
(uses <a href="lenses.tab">lenses.tab</a>)</p>
<xmp class="code">>>> learner.stop.maxMajority = 0.5
>>> tree = learner(tree)
>>> printTree(tree)
--> none (<15.000, 5.000, 4.000>)
</xmp>

<p>Well, this might have been an overkill...</p>


<H4>Splitting criteria</H4>

<H4>Example splitter</H4>

<H4>Flags and similar</H4>

... also mention nodeLearner and descender

<H4>Programming your own tree learner skeleton</H4>

<H3>Classification</H3>

<H4>Descender</H4>

<H4>Node classifier</H4>

<H3>Pruning</H3>

<H4>Same majority pruning</H4>

<H4>Post pruning</H4>

... show a series of trees

<H3>Defining your own components</H3>

<H3>Various tricks</H3>

<H4>Storing testing examples</H4>

... show how to remember misclassifications etc.<P>

<H4>Replacing node classifiers</H4>
... replacing with something else, but based on learning examples<P>

<hr>

<H3><U>References</U></H3>

Bratko, I. (2002). <EM>Prolog Programming for Artificial Intelligence</EM>, Addison Wesley, 2002.<P>
"""

from Orange.core import \
     TreeLearner, \
         TreeClassifier, \
         C45Learner, \
         C45Classifier, \
         C45TreeNode, \
         C45TreeNodeList, \
         TreeDescender, \
              TreeDescender_UnknownMergeAsBranchSizes, \
              TreeDescender_UnknownMergeAsSelector, \
              TreeDescender_UnknownToBranch, \
              TreeDescender_UnknownToCommonBranch, \
              TreeDescender_UnknownToCommonSelector, \
         TreeExampleSplitter, \
              TreeExampleSplitter_IgnoreUnknowns, \
              TreeExampleSplitter_UnknownsAsBranchSizes, \
              TreeExampleSplitter_UnknownsAsSelector, \
              TreeExampleSplitter_UnknownsToAll, \
              TreeExampleSplitter_UnknownsToBranch, \
              TreeExampleSplitter_UnknownsToCommon, \
              TreeExampleSplitter_UnknownsToRandom, \
         TreeNode, \
         TreeNodeList, \
         TreePruner, \
              TreePruner_SameMajority, \
              TreePruner_m, \
         TreeSplitConstructor, \
              TreeSplitConstructor_Combined, \
              TreeSplitConstructor_Measure, \
                   TreeSplitConstructor_Attribute, \
                   TreeSplitConstructor_ExhaustiveBinary, \
                   TreeSplitConstructor_OneAgainstOthers, \
                   TreeSplitConstructor_Threshold, \
         TreeStopCriteria, \
              TreeStopCriteria_Python, \
              TreeStopCriteria_common


