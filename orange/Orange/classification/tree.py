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

    .. method:: treeSize():
        
        Return the number of nodes in the subtrees (including the node, excluding null-nodes).

<A name="classification"></A>
<H3>Classification</H3>

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

    # Descender reaches a leaf. This happens when nothing went wrong (there are no unknown or out-of-range values in the example) or when things went wrong, but the descender smoothed them by selecting a single branch and continued the descend. In this case, the descender returns the reached :obj:`TreeNode`.

    # :obj:`branchSelector` returned a distribution and the :obj:`TreeDescender` decided to stop the descend at this (internal) node. Again, descender returns the current <code>TreeNode</code> and nothing else.</LI>

    # :obj:`branchSelector` returned a distribution and the <code>TreeNode</code> wants to split the example (i.e., to decide the class by voting). It returns a <code>TreeNode</code> and the vote-weights for the branches. The weights can correspond to the distribution returned by
<code>branchSelector</code>, to the number of learning examples that were assigned to each branch, or to something else.</LI>
</UL>

<p><code>TreeClassifier</code> uses the descender to descend from the root. If it returns only a <code>TreeNode</code> and no distribution, the descend should stop; it does not matter whether it's a leaf (the first case above) or an internal node (the second case). The node's <code>nodeClassifier</code> is used to decide the class. If the descender returns a <code>TreeNode</code> and a distribution, the <code>TreeClassifier</code> recursively calls itself for each of the subtrees and the predictions are weighted as requested by the descender.</p>

<p>When voting, subtrees do not predict the class but probabilities of classes. The predictions are multiplied by weights, summed and the most probable class is returned.</p>

<p><B>The rest of this section is only for those interested in the C++ code.</B></p>

<p>If you'd like to understand how the classification works in C++, start reading at <code>TTreeClassifier::vote</code>. It gets a <code>TreeNode</code>, an <code>Example</code> and a distribution of vote weights. For each node, it calls the <code>TTreeClassifier::classDistribution</code> and then multiplies and sums the distribution. <code>vote</code> returns a normalized distribution of predictions.</p>

<p>A new overload of <code>TTreeClassifier::classDistribution</code> gets an additional parameter, a <code>TreeNode</code>. This is done for the sake of recursion. The normal version of <code>classDistribution</code> simply calls the overloaded with a tree root as an additional parameter. <code>classDistribution</code> uses <code>descender</code>. If descender reaches a leaf, it calls <code>nodeClassifier</code>, otherwise it calls <CODE>vote</CODE>.</P>

<p>Thus, the <code>TreeClassifier</code>'s <code>vote</code> and <code>classDistribution</code> are written in a form of double recursion. The recursive calls do not happen at each node of the tree but only at nodes where a vote is needed (that is, at nodes where the descender halts).</p>

<p>For predicting a class, <code>operator()</code>, calls the descender. If it reaches a leaf, the class is predicted by the leaf's <code>nodeClassifier</code>. Otherwise, it calls <code>vote</code>. From now on, <code>vote</code> and <code>classDistribution</code> interweave down the tree and return a distribution of predictions. <code>operator()</code> then simply chooses the most probable class.</p>


<A name="learning"></A>
<H3>Learning</H3>

<p>The main learning object is <CODE><INDEX name="classes/TreeLearner">TreeLearner</CODE>. It is basically a skeleton into which the user must plug the components for particular functions. For easier use, defaults are provided.</p>

<p>Components that govern the structure of the tree are <code>split</code> (of type <code>TreeSplitConstructor</code>), <code>stop</code> (of type <code>TreeStopCriteria</code>) and <code>exampleSplitter</code> (of type <code>TreeExampleSplitter</code>).</p>

<H4>TreeSplitConstructor</H4>

<p>The job of <code><INDEX name="classes/TreeSplitConstructor">TreeSplitConstructor</code> is to find a suitable criteria for dividing the learning (and later testing) examples coming to the node. The data it gets is a set of examples (and, optionally, an ID of weight meta-attribute), a domain contingency computed from examples, apriori class probabilities, a list of candidate attributes it should consider and a node classifier (if it was constructed, that is, if <CODE>storeNodeClassifier</CODE> is left <CODE>true</CODE>).</p>

<p>The <code>TreeSplitConstructor</code> should use the domain contingency when possible. The reasons are two-fold; one is that it's faster and the other is that the contingency matrices are not necessarily constructed by simply counting the examples. Why and how is explained later. There are, however, cases, when domain contingency does not suffice, for examples, when ReliefF is used as a measure of quality of attributes. In this case, there's no other way but to use the examples and ignore the precomputed contingencies.</p>

<p>The split constructor should consider only the attributes in the candidate list (the list is a vector of booleans, one for each attribute).</p>

<p><code>TreeSplitConstructor</code> returns most of the data we talked about when describing the <code>TreeNode</code>. It returns a classifier to be used as <code>TreeNode</code>'s <code>branchSelector</code>, a list of branch descriptions and a list with the number of examples that go into each branch. Just what we need for the <code>TreeNode</code>. It can return an empty list for the number of examples in branches; in this case, the <code>TreeLearner</code> will find the number itself after splitting the example set into subsets. However, if a split constructors can provide the numbers at no extra computational cost, it should do so.</P>

<p>In addition, it returns a quality of the split; a number without any fixed meaning except that higher numbers mean better splits.</p>

<p>If the constructed splitting criterion uses an attribute in such a way that the attribute is 'completely spent' and should not be considered as a split criterion in any of the subtrees (the typical case of this are discrete attributes that are used as-they-are, that is, without any binarization or subsetting), then it should report the index of this attribute. Some splits do not spend any attribute; this is indicated by returning a negative index.</p>

<p>A <code>TreeSplitConstructor</code> can veto the further tree induction by returning no classifier. This can happen for many reasons. A general one is related to number of examples in the branches. <code>TreeSplitConstructor</code> has a field <code>minSubset</code>, which sets the minimal number of examples in a branch; null nodes, however, are allowed. If there is no split where this condition is met, <code>TreeSplitConstructor</code> stops the induction.</p>


<H4>TreeStopCriteria</H4>

<p><code><INDEX name="classes/TreeStopCriteria">TreeStopCriteria</code> is a much simpler component that, given a set of examples, weight ID and contingency matrices, decides whether to continue the induction or not. The basic criterion checks whether there are any examples and whether they belong to at least two different classes (if the class is discrete). Derived components check things like the number of examples and the proportion of majority classes.</p>


<H4>TreeExampleSplitter</H4>

<p><code><INDEX name="classes/TreeExampleSplitter">TreeExampleSplitter</code> is analogous to the <code>TreeDescender</code> described about a while ago. Just like the <code>TreeDescender</code> decides the branch for an example during classification, the <code>TreeExampleSplitter</code> sorts the learning examples into branches.</p>

<p><code>TreeExampleSplitter</code> is given a <code>TreeNode</code> (from which it can use different stuff, but most of splitters only use the <code>branchSelector</code>), a set of examples to be divided, and the weight ID. The result is a list of subsets of examples and, optionally, a list of new weight ID's.</p>

<p>Subsets are usually stored as <code>ExamplePointerTable</code>'s. Most of <code>TreeExampleSplitters</code> simply call the node's <code>branchSelector</code> and assign examples to corresponding branches. When the value is unknown they choose a particular branch or simply skip the example.</p>

<p>Some enhanced splitters can split examples. An example (actually, a pointer to it) is copied to more than one subset. To facilitate real splitting, weights are needed. Each branch is assigned a weight ID (each would usually have its own ID) and all examples that are in that branch (either completely or partially) should have this meta attribute. If an example hasn't been split, it has only one additional attribute - with weight ID corresponding to the subset to which it went. Example that is split between, say, three subsets, has three new meta attributes, one for each subset. ID's of weight meta attributes are returned by the <code>TreeExampleSplitter</code> to be used at induction of the corresponding subtrees.</p>

<p>Note that weights are used only when needed. When no splitting occured - because the splitter is not able to do it or because there was no need for splitting - no weight ID's are returned.</p>

<H4>TreeLearner</H4>

<p>TreeLearner has a number of components.</p>

<P class=section>
<DL class=attributes>
<DT>split</DT>
<DD>Object of type <code>TreeSplitConstructor</code>. Default value, provided by <code>TreeLearner</code>, is <code>SplitConstructor_Combined</code> with separate constructors for discrete and continuous attributes. Discrete attributes are used as are, while continuous attributes are binarized. Gain ratio is used to select attributes. A minimum of two examples in a leaf is required for discrete and five examples in a leaf for continuous attributes.</DD>

<DT>stop</DT>
<DD>Object of type <code>TreeStopCriteria</code>. The default stopping criterion stops induction when all examples in a node belong to the same class.</DD>

<DT>splitter</DT>
<DD>Object of type <code>TreeExampleSplitter</code>. The default splitter is <code>TreeExampleSplitter_UnknownsAsSelector</code> that splits the learning examples according to distributions given by the selector.</DD>

<DT>contingencyComputer</DT>
<DD>By default, this slot is left empty and ordinary contingency matrices are computed for examples at each node. If need arises, one can change the way the matrices are computed. This can be used to change the way that unknown values are treated when assessing qualities of attributes. As mentioned earlier, the computed matrices can be used by split constructor and by stopping criteria. On the other hand, they can be (and are) ignored by some splitting constructors.</DD>

<DT>nodeLearner</DT>
<DD>Induces a classifier from examples belonging to a node. The same learner is used for internal nodes and for leaves. The default <code>nodeLearner</code> is <code>MajorityLearner</code>.</DD>

<DT>descender</DT>
<DD>Descending component that the induces <code>TreeClassifier</code> will use. Default descender is <code>TreeDescender_UnknownMergeAsSelector</code> which votes using the <code>branchSelector</code>'s distribution for vote weights.</DD>

<DT>maxDepth</DT>
<DD>Gives maximal tree depth; 0 means that only root is generated. The default is 100 to prevent any infinite tree induction due to missettings in stop criteria. If you are sure you need larger trees, increase it. If you, on the other hand, want to lower this hard limit, you can do so as well.</DD>

<DT>storeDistributions, storeContingencies, storeExamples, storeNodeClassifier</DT>
<DD>Decides whether to store class distributions, contingencies and examples in <code>TreeNodes</code>, and whether the <code>nodeClassifier</code> should be build for internal nodes. By default, distributions and node classifiers are stored, while contingencies and examples are not. You won't save any memory by not storing distributions but storing contingencies, since distributions actually points to the same distribution that is stored in <code>contingency.classes</code>.</DD>
</DL>

<p>The <code>TreeLearner</code> first sets the defaults for missing components. Although stored in the actual <code>TreeLearner</code>'s fields, they are removed when the induction is finished.</p>

<p>Then it ensures that examples are stored in a table. This is needed because the algorithm juggles with pointers to examples. If examples are in a file or are fed through a filter, they are copied to a table. Even if they are already in a table, they are copied if <code>storeExamples</code> is set. This is to assure that pointers remain pointing to examples even if the user later changes the example table. If they are in the table and the <code>storeExamples</code> flag is clear, we just use them as they are. This will obviously crash in a multi-threaded system if one changes the table during the tree induction. Well... don't do it.</p>

<p>Apriori class probabilities are computed. At this point we check the sum of example weights; if it's zero, there are no examples and we cannot proceed. A list of candidate attributes is set; in the beginning, all attributes are candidates for the split criterion.</p>

<p>Now comes the recursive part of the <code>TreeLearner</code>. Its arguments are a set of examples, a weight meta-attribute ID (a tricky thing, it can be always the same as the original or can change to accomodate splitting of examples among branches), apriori class distribution and a list of candidates (represented as a vector of Boolean values).</p>

<p><code>Contingency matrix</code> is computed next. This happens even if the flag <code>storeContingencies</code> is <CODE>false</CODE>. If the <code>contingencyComputer</code> is given we use it, otherwise we construct just an ordinary contingency matrix.</p>

<p>A <code>stop</code> is called to see whether it's worth to continue. If not, a <code>nodeClassifier</code> is built and the <code>TreeNode</code> is returned. Otherwise, a <code>nodeClassifier</code> is only built if <code>forceNodeClassifier</code> flag is set.</p>

<p>To get a <code>TreeNode</code>'s <code>nodeClassifier</code>, the <code>nodeLearner</code>'s <code>smartLearn</code> function is called with the given examples, weight ID and the just computed matrix. If the learner can use the matrix (and the default, <code>MajorityLearner</code>, can), it won't touch the examples. Thus, a choice of <code>contingencyComputer</code> will, in many cases, affect the <code>nodeClassifier</code>. The <code>nodeLearner</code> can return no classifier; if so and if the classifier would be needed for classification, the <code>TreeClassifier</code>'s function returns DK or an empty distribution. If you're writing your own tree classifier - pay attention.</p>

<p>If the induction is to continue, a <code>split</code> component is called. If it fails to return a branch selector, induction stops and the <code>TreeNode</code> is returned.</p>

<p><code>TreeLearner</code> than uses <code>ExampleSplitter</code> to divide the examples as described above.</p>

<p>The contingency gets removed at this point if it is not to be stored. Thus, the <code>split</code>, <code>stop</code> and <code>exampleSplitter</code> can use the contingency matrices if they will.</p>

<p>The <code>TreeLearner</code> then recursively calls itself for each of the non-empty subsets. If the splitter returnes a list of weights, a corresponding weight is used for each branch. Besides, the attribute spent by the splitter (if any) is removed from the list of candidates for the subtree.</p>

<p>A subset of examples is stored in its corresponding tree node, if so requested. If not, the new weight attributes are removed (if any were created).</p>

<A name="pruning"></A>
<H3>Pruning</H3>

<p>Tree pruners derived from <code><INDEX name="classes/TreePrune">TreePrune</code> can be given either a <code>TreeNode</code> (presumably, but not necessarily a root) or a <code>TreeClassifier</code>. The result is a new, pruned <code>TreeNode</code> or a new <code>TreeClassifier</code> with a pruned tree. The original tree remains intact.</p>

<p>Note however that pruners construct only a shallow copy of a tree. The pruned tree's <code>TreeNode</code>s contain references to the same contingency matrices, node classifiers, branch selectors, ... as the original tree. Thus, you may modify a pruned tree structure (manually cut it, add new nodes, replace components) but modifying, for instance, some node's <code>nodeClassifier</code> (a <code>nodeClassifier</code> itself, not a reference to it!) would modify the node's <code>nodeClassifier</code> in the corresponding node of the original tree.</p>

<p>Talking about node classifiers - pruners cannot construct a <code>nodeClassifier</code> nor merge <code>nodeClassifiers</code> of the pruned subtrees into classifiers for new leaves. Thus, if you want to build a prunable tree, internal nodes must have their <code>nodeClassifiers</code> defined. Fortunately, all you need to do is nothing; if you leave the <code>TreeLearner</code>'s flags as they are by default, the <code>nodeClassifiers</code> are created.</p>

<hr>

<A name="classes"></A>
<H2>Classes</H2>

<p>Several classes described above are already functional and can (and mostly will) be used as they are. Those classes are <code>TreeNode</code>, <code>TreeLearner</code> and <code>TreeClassifier</code>. This section describe the other classes.</p>

<p>Classes <code>TreeSplitConstructor</code>, <code>TreeStopCriteria</code>, <code>TreeExampleSplitter</code>, <code>TreeDescender</code>, <code>Learner</code> and <code>Classifier</code> are among the Orange classes that can be subtyped in Python and have the call operator overloadedd in such a way that it is callbacked from C++ code. You can thus program your own components for <code>TreeLearners</code> and <code>TreeClassifiers</code>. The detailed information on how this is done and what can go wrong, is given in a separate page, dedicated to <A href="callbacks.htm">callbacks to Python</A>.</p>


<H3>TreeSplitConstructor and Derived Classes</H3>

<p>Split construction is almost as exciting as waiting for a delayed flight. Boring, that is. Split constructors are programmed as spaghetti code that juggles with contingency matrices, with separate cases for discrete and continuous classes... Most split constructors work either for discrete or for continuous attributes. The suggested practice is to use a <code>TreeSplitConstructor_Combined</code> that can handle both by simply delegating attributes to specialized split constructors.</p>

<p><B>Note:</B> split constructors that cannot handle attributes of particular type (discrete, continuous) do not report an error or a warning but simply skip the attribute. It is your responsibility to use a correct split constructor for your dataset. (May we again suggest using <code>TreeSplitConstructor_Combined</code>?)</p>

<p>The same components can be used either for inducing classification and regression trees. The only component that needs to be chosen accordingly is the 'measure' attribute for the <code>TreeSplitConstructor_Measure</code> class (and derived classes).</p>

<H4>TreeSplitConstructor</H4>

<p>The <code>TreeSplitConstructor</code>'s function has been described in details in <a href="#learning">description of the learning process</A>.</p>

<p class=section>Attributes</P>
<DL class=attributes>
<DT>minSubset</DT>
<DD>Sets the minimal number of examples in non-null leaves. As always in Orange (where not specified otherwise), "number of examples" refers to the weighted number of examples.</DD>
</DL>

<p class=section>Methods</P>
<DL class=attributes>
<DT>__call__(examples[, weightID, apriori_distribution, candidates])</DT>
<DD>Constructs a split. Examples can be given in any acceptable form (an <code>ExampleGenerator</code>, such as <code>ExampleTable</code>, or a list of examples). <code>WeightID</code> is optional; the default of 0 means that all examples have a weight of 1.0. Apriori-distribution should be of type <code>orange.Distribution</code> and candidates should be a Python list of objects which are interpreted as booleans.
The function returns a tuple (<code>branchSelector</code>, <code>branchDescriptions</code>, <code>subsetSizes</code>, <code>quality</code>, <code>spentAttribute</code>). <code>SpentAttribute</code> is -1 if no attribute is completely spent by the split criterion. If no split is constructed, the <code>selector</code>, <code>branchDescriptions</code> and <code>subsetSizes</code> are <code>None</code>, while <CODE>quality</CODE> is 0.0 and <code>spentAttribute</code> is -1.</DD>
</DL>


<H4>TreeSplitConstructor_Measure</H4>
<index name="classes/TreeSplitConstructor_Measure">

<p>An abstract base class for split constructors that employ a <code>MeasureAttribute</code> to assess a quality of a split. At present, all split constructors except for <code>TreeSplitConstructor_Combined</code> are derived from this class.</p>

<p class=section>Attributes</p>
<DL class=attributes>
<DT>measure</DT>
<DD>A component of type <code>MeasureAttribute</code> used for evaluation of a split. Note that you must select the subclass <code>MeasureAttribute</code> capable of handling your class type - you cannot use <code>MeasureAttribute_gainRatio</code> for building regression trees or <code>MeasureAttribute_MSE</code> for classification trees.</DD>

<DT>worstAcceptable</DT>
<DD>The lowest required split quality for a split to be acceptable. Note that this value make sense only in connection with a <code>measure</code> component. Default is 0.0.</DD>
</DL>

<H4>TreeSplitConstructor_Attribute</H4>
<index name="classes/TreeSplitConstructor_Attribute">

<p><code>TreeSplitConstructor_Attribute</code> attempts to use a discrete attribute as a split; each value of the attribute corresponds to a branch in the tree. Attributes are evaluated with the <code>measure</code> and the one with the highest score is used for a split. If there is more than one attribute with the highest score, one of them is selected by random.</P>

<p>The constructed <code>branchSelector</code> is an instance of <code>ClassifierFromVarFD</code> that returns a value of the selected attribute. If the attribute is <code>EnumVariable</code>, <code>branchDescription</code>'s are the attribute's values. The attribute is marked as spent, so that it cannot reappear in the node's subtrees.</p>

<H4>TreeSplitConstructor_ExhaustiveBinary</H4>
<index name="classes/TreeSplitConstructor_ExhaustiveBinary">
<index name="binarization (in trees)"

<p><code>TreeSplitConstructor_ExhaustiveBinary</code> works on discrete attributes. For each attribute, it determines which binarization of the attribute gives the split with the highest score. If more than one split has the highest score, one of them is selected by random. After trying all the attributes, it returns one of those with the highest score.</p>

<p>The constructed <CODE>branchSelector</CODE> is again an instance <code>ClassifierFromVarFD</code> that returns a value of the selected attribute. This time, however, its <code>transformer</code> contains an instance of <code>MapIntValue</code> that maps the values of the attribute into a binary attribute. Branch descriptions are of form "[&lt;val1&gt;, &lt;val2&gt;, ...&lt;valn&gt;]" for branches corresponding to more than one value of the attribute. Branches that correspond to a single value of the attribute are described with this value. If the attribute was originally binary, it is spent and cannot be used in the node's subtrees. Otherwise, it can reappear in the subtrees.</p>

<H4>TreeSplitConstructor_Threshold</H4>
<index name="classes/TreeSplitConstructor_Threshold">

<p>This is currently the only constructor for splits with continuous attributes. It divides the range of attributes values with a threshold that maximizes the split's quality. As always, if there is more than one split with the highest score, a random threshold is selected. The attribute that yields the highest binary split is returned.</p>

<p>The constructed <code>branchSelector</code> is again an instance of <code>ClassifierFromVarFD</code> with an attached <code>transformer</code>. This time, <code>transformer</code> is of type <code>ThresholdDiscretizer</code>. The branch descriptions are "&lt;threshold" and "&gt;=threshold". The attribute is not spent.</p>


<H4>TreeSplitConstructor_Combined</H4>
<index name="classes/TreeSplitConstructor_Combined">

<p>This constructor delegates the task of finding the optimal split to separate split constructors for discrete and for continuous attributes. Each split constructor is called, given only attributes of appropriate types as candidates. Both construct a candidate for a split; the better of them is selected.</p>

<p>(Note that there is a problem when more candidates have the same score. Let there be are nine discrete attributes with the highest score; the split constructor for discrete attributes will select one of them. Now, let us suppose that there is a single continuous attribute with the same score. <code>TreeSplitConstructor_Combined</code> would randomly select between the proposed discrete attribute and the continuous attribute, not aware of the fact that the discrete has already competed with eight other discrete attributes. So, the probability for selecting (each) discrete attribute would be 1/18 instead of 1/10. Although not really correct, we doubt that this would affect the tree's performance; many other machine learning systems simply choose the first attribute with the highest score anyway.)</p>

<p>The <code>branchSelector</code>, <code>branchDescriptions</code> and whether the attribute is spent is decided by the winning split constructor.</p>

<p class=section>Attributes</p>
<DL class=attributes>
<DT>discreteSplitConstructor</DT>
<DD>Split constructor for discrete attributes; can be, for instance, <code>TreeSplitConstructor_Attribute</code> or <code>TreeSplitConstructor_ExhaustiveBinary</code></DD>

<DT>continuousSplitConstructor</DT>
<DD>Split constructor for continuous attributes; at the moment, it can be either <code>TreeSplitConstructor_Threshold</code> or a split constructor you programmed in Python.</DD>
</DL>


<H3>TreeStopCriteria and TreeStopCriteria_common</H3>

<p><code>TreeStopCriteria</code> determines when to stop the induction of subtrees, as described in detail in <a href="#learning">description of the learning process</A>.</p>

<H4>TreeStopCriteria</H4>

<p>As opposed to <code>TreeSplitConstructor</code> and similar basic classes, <code>TreeStopCriteria</code> is not an abstract but a fully functional class that provides the basic stopping criteria. That is, the tree induction stops when there is at most one example left; in this case, it is not the weighted but the actual number of examples that counts. Besides that, the induction stops when all examples are in the same class (for discrete problems) or have the same value of the outcome (for regression problems).</p>

<p class=section>Methods</p>
<DL class=attributes>
<DT>__call__(examples[, weightID, domain contingencies])</DT>
<DD>
Decides whether to stop (<CODE>true</CODE>) or continue (<CODE>false</CODE>) the induction. If contingencies are given, they are used for checking whether the examples are in the same class (but not for counting the examples). Derived classes should use the contingencies whenever possible. If contingencies are not given, <code>TreeStopCriteria</code> will work without them. Derived classes should also use them if they are available, but otherwise compute them only when they really need them.
</DD>
</DL>

<H4>TreeStopCriteria_common</H4>
<index name="classes/TreeSplitConstructor_common">

<p><code>TreeStopCriteria_common</code> contains additional criteria for pre-pruning: it checks the proportion of majority class and the number of weighted examples.</p>

<p class=section>Attributes</p>
<DL class=attributes>
<DT>maxMajor</DT>
<DD>Maximal proportion of majority class. When this is exceeded, induction stops.</DD>

<DT>minExamples</DT>
<DD>Minimal number of examples in internal leaves. Subsets with less than <code>minExamples</code> examples are not split any further. Example count is weighed.</DD>
</DL>

<H3>TreeExampleSplitter and derived classes</H3>

<p><code>TreeExampleSplitter</code> is the third crucial component of <code>TreeLearner</code>. Its function is described in <a href="#learning">description of the learning process</A>.</p>

<H4>TreeExampleSplitter</H4>

<p>An abstract base class for objects that split sets of examples into subsets. The derived classes differ in treatment of examples which cannot be unambiguously placed into a single branch (usually due to unknown value of the crucial attribute).</p>

<P class=section>Methods</p>
<DL class=attributes>
<DT><B>__call__(node, examples[, weightID])</B></DT>
<DD>
Uses the information in <code>node</code> (particularly the <code>branchSelector</code>) to split the given set of examples into subsets. Function returns a tuple with a list of example generators and a list of weights. The list of weights is either an ordinary python list of integers or a None when no splitting of examples occurs and thus no weights are needed.
</DD>
</DL>

<p class=section>Derived classes</p>
<DL class=attributes>
<DT><INDEX name="classes/TreeExampleSplitter_IgnoreUnknowns">TreeExampleSplitter_IgnoreUnknowns</DT>
<DD>Simply ignores the examples for which no single branch can be determined.</DD>

<DT><INDEX name="classes/TreeExampleSplitter_UnknownsToCommon">TreeExampleSplitter_UnknownsToCommon</DT>
<DD>Places all such examples to a branch with the highest number of examples. If there is more than one such branch, one is selected at random and then used for all examples.</DD>

<DT><INDEX name="classes/TreeExampleSplitter_UnknownsToAll">TreeExampleSplitter_UnknownsToAll</DT>
<DD>Places examples with unknown value of the attribute into all branches.</DD>

<DT><INDEX name="classes/TreeExampleSplitter_UnknownsToRandom">TreeExampleSplitter_UnknownsToRandom</DT>
<DD>Selects a random branch for such examples.</DD>

<DT><INDEX name="classes/TreeExampleSplitter_UnknownsToBranch">TreeExampleSplitter_UnknownsToBranch</DT>
<DD>Constructs an additional branch to contain all such examples. The branch's description is "unknown".</DD>

<DT><INDEX name="classes/TreeExampleSplitter_UnknownsAsBranchSizes">TreeExampleSplitter_UnknownsAsBranchSizes</DT>
<DD>Splits examples with unknown value of the attribute according to proportions of examples in each branch.</DD>

<DT><INDEX name="classes/TreeExampleSplitter_UnknownsAsSelector">TreeExampleSplitter_UnknownsAsSelector</DT>
<DD>Splits examples with unknown value of the attribute according to distribution proposed by selector (which is in most cases the same as proportions of examples in branches).</DD>
</DL>



<H3>TreeDescender and derived classes</H3>

<p>This is a classifier's counterpart for <code>TreeExampleSplitter</code>. It decides the destiny of examples that need to be classified and cannot be unambiguously put in a branch. The detailed function of this class is given in <a href="#classification"> description of classification with trees</A>.</p>

<H4>TreeDescender</H4>

<p>An abstract base object for tree descenders.</p>

<p class=section>Methods</p>
<DL class=attributes>
<DT>__call__(node, example)</DT>
<DD>Descends down the tree until it reaches a leaf or a node in which a vote of subtrees is required. In both cases, a tuple of two elements is returned; in the former, the tuple contains the reached node and <CODE>None</CODE>, in the latter in contains a node and weights of votes for subtrees (a list of floats).

<code>TreeDescender</code>'s that never split examples always descend to a leaf, but they differ in the treatment of examples with unknown values (or, in general, examples for which a branch cannot be determined at some node(s) the tree). <code>TreeDescender</code>'s that do split examples differ in returned vote weights.</DD>
</DL>

<p class=section>Derived classes</p>
<DL class=attributes>
<DT><INDEX name="classes/TreeDescender_UnknownsToNode">TreeDescender_UnknownsToNode</DT>
<DD>When example cannot be classified into a single branch, the current node is returned. Thus, the node's <CODE>nodeClassifier</CODE> will be used to make a decision. It is your responsibility to see that even the internal nodes have their <CODE>nodeClassifiers</CODE> (i.e., don't disable creating node classifier or manually remove them after the induction, that's all)</DD>

<DT><INDEX name="classes/TreeDescender_UnknownsToCommon">TreeDescender_UnknownsToCommon</DT>
<DD>Classifies examples with unknown value to a special branch. This makes sense only if the tree itself was constructed with <CODE>TreeExampleSplitter_UnknownsToBranch</CODE>.</DD>

<DT><INDEX name="classes/TreeDescender_UnknownsToCommonBranch">TreeDescender_UnknownsToCommonBranch</DT>
<DD>Classifies examples with unknown values to the branch with the highest number of examples. If there is more than one such branch, random branch is chosen for each example that is to be classified.</DD>

<DT><INDEX name="classes/TreeDescender_UnknownsToCommonSelector">TreeDescender_UnknownsToCommonSelector</DT>
<DD>Classifies examples with unknown values to the branch which received the highest recommendation by the selector.</DD>


<DT><INDEX name="classes/TreeDescender_MergeAsBranchSizes">TreeDescender_MergeAsBranchSizes</DT>
<DD>Makes the subtrees vote for the example's class; the vote is weighted according to the sizes of the branches.</DD>

<DT><INDEX name="classes/TreeDescender_MergeAsSelector">TreeDescender_MergeAsSelector</DT>
<DD>Makes the subtrees vote for the example's class; the vote is weighted according to the selectors proposal.</DD>
</DL>


<H3>TreePruner and derived classes</H3>
<index name="classification trees/pruning">
<index name="pruning classification trees">

<p>Classes derived from TreePruner prune the trees as described in the section <A href="#pruning">pruning</A> - make sure you read it to understand what the pruners will do to your trees.</p>

<H4>TreePruner</H4>

<p>This is an abstract base class which defines nothing useful, only a pure virtual call operator.</p>

<P class=section>Methods</P>
<DL class=attributes>
<DT>__call__(tree)</DT>
<DD>Prunes a tree. The argument can be either a tree classifier or a tree node; the result is of the same type as the argument.</DD>
</DL>


<H4>TreePruner_SameMajority</H4>
<index name="classes/TreePruner_SameMajority">

<p>In Orange, a tree can have a non-trivial subtrees (i.e. subtrees with more than one leaf) in which all the leaves have the same majority class. (The reason why this is allowed is that those leaves can still have different distributions of classes and thus predict different probabilities.) However, this can be undesired when we're only interested in the class prediction or a simple tree interpretation. The <code>TreePruner_SameMajority</code> prunes the tree so that there is no subtree in which all the nodes would have the same majority class.</p>

<p>This pruner will only prune the nodes in which the node classifier is of class <code>DefaultClassifier</code> (or from a derived class).</p>

<p>Note that the leaves with more than one majority class require some special handling. The pruning goes backwards, from leaves to the root. When siblings are compared, the algorithm checks whether they have (at least one) common majority class. If so, they can be pruned.</p>

<H4>TreePruner_m</H4>
<index name="classes/TreePruner_m">

<p>Prunes a tree by comparing m-estimates of static and dynamic error as defined in (Bratko, 2002).</p>

<p class=section>Attributes</p>
<DL class=attributes>
<DT>m</DT>
<DD>Parameter m for m-estimation.</DD>
</DL>

<hr>

<A name="examples"></A>
<H2>Examples</H2>

<p>This page does not provide examples for programming your own components, such as, for instance, a <code>TreeSplitConstructor</code>. Those examples can be found on a page dedicated to <A href="callbacks.htm">callbacks to Python</A>.</p>

<H3>Tree Structure</H3>

<p>To have something to work on, we'll take the data from lenses dataset and build a tree using the default components.</p>

<p class="header">part of <a href="treestructure.py">treestructure.py</a>
(uses <a href="lenses.tab">lenses.tab</a>)</p>
<xmp class="code">>>> data = orange.ExampleTable("lenses")
>>> treeClassifier = orange.TreeLearner(data)
</xmp>

<p>How big is our tree?</p>

<p class="header">part of <a href="treestructure.py">treestructure.py</a>
(uses <a href="lenses.tab">lenses.tab</a>)</p>
<xmp class="code">def treeSize(node):
    if not node:
        return 0

    size = 1
    if node.branchSelector:
        for branch in node.branches:
            size += treeSize(branch)

    return size
</xmp>

<p>If node is <CODE>None</CODE>, we have a null-node; null nodes don't count, so we return 0. Otherwise, the size is 1 (this node) plus the sizes of all subtrees. The node is an internal node if it has a <CODE>node.branchSelector</CODE>; it there's no selector, it's a leaf. Don't attempt to skip the <CODE>if</CODE> statement: leaves don't have an empty list of branches, they don't have a list of branches at all.</p>

<xmp>>>> treeSize(treeClassifier.tree)
10
</xmp>

<p>Don't forget that this was only an excercise - <code>TreeNode</code> has a built-in method <code>treesize()</code> that does exactly the same.</p>

<p>Let us now write a simple script that prints out a tree. The recursive part of the function will get a node and its level.</p>

<p class="header">part of <a href="treestructure.py">treestructure.py</a>
(uses <a href="lenses.tab">lenses.tab</a>)</p>
<xmp class="code">def printTree0(node, level):
    if not node:
        print " "*level + "<null node>"
        return

    if node.branchSelector:
        nodeDesc = node.branchSelector.classVar.name
        nodeCont = node.distribution
        print "\n" + "   "*level + "%s (%s)" % (nodeDesc, nodeCont),
        for i in range(len(node.branches)):
            print "\n" + "   "*level + ": %s" % node.branchDescriptions[i],
            printTree0(node.branches[i], level+1)
    else:
        nodeCont = node.distribution
        majorClass = node.nodeClassifier.defaultValue
        print "--> %s (%s) " % (majorClass, nodeCont),
</xmp>

<p>Don't waste time on studying formatting tricks (\n's etc.), this is just for nicer output. What matters is everything but the print statements. As first, we check whether the node is a null-node (a node to which no learning examples were classified). If this is so, we just print out "&lt;null node&gt;" and return.</p>

<p>After handling null nodes, remaining nodes are internal nodes and leaves. For internal nodes, we print a node description consisting of the attribute's name and distribution of classes. <code>TreeNode</code>'s branch description is, for all currently defined splits, an instance of a class derived from <code>Classifier</code> (in fact, it is a <code>ClassifierFromVarFD</code>, but a <code>Classifier</code> would suffice), and its <code>classVar</code> points to the attribute we seek. So we print its name. We will also assume that storing class distributions has not been disabled and print them as well. A more able function for printing trees (as one defined in <code>orngTree</code>) has an alternative means to get the distribution, when this fails. Then we iterate through branches; for each we print a branch description and iteratively call the <code>printTree0</code> with a level increased by 1 (to increase the indent).</p>

<p>Finally, if the node is a leaf, we print out the distribution of learning examples in the node and the class to which the examples in the node would be classified. We again assume that the <code>nodeClassifier</code> is the default one - a <code>DefaultClassifier</code>. A better print function should be aware of possible alternatives.</p>

<p>Now, we just need to write a simple function to call our printTree0. We could write something like...</p>
<xmp class="code">def printTree(x):
    printTree0(x.tree, 0)
</xmp>
<p>... but we won't. Let us learn how to handle arguments of different types. Let's write a function that will accept either a <code>TreeClassifier</code> or a <code>TreeNode</code> (just like <code>TreePruners</code>, remember?)</p>

<p class="header">part of <a href="treestructure.py">treestructure.py</a>
(uses <a href="lenses.tab">lenses.tab</a>)</p>
<xmp class="code">def printTree(x):
    if isinstance(x, orange.TreeClassifier):
        printTree0(x.tree, 0)
    elif isinstance(x, orange.TreeNode):
        printTree0(x, 0)
    else:
        raise TypeError, "TreeClassifier or TreeNode expected"
</xmp>

<p>It's fairly straightforward: if <code>x</code> is of type derived from <code>orange.TreeClassifier</code>, we print <code>x.tree</code>; if it's <code>TreeNode</code> we just call <code>printTree0</code> with <code>x</code>. If it's of some other type, we don't know how to handle it and thus raise an exception. (Note that we could also use <CODE>if type(x) == orange.TreeClassifier:</CODE>, but this would only work if <CODE>x</CODE> would be of type <CODE>orange.TreeClassifier</CODE> and not of any derived types. The latter, however, do not exist yet...)</p>

<p class="header">part of <a href="treestructure.py">treestructure.py</a>
(uses <a href="lenses.tab">lenses.tab</a>)</p>
<xmp class="code">>>> printTree(treeClassifier)

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
</xmp>

<p>For a final exercise, let us write a simple pruning procedure. It will be written entirely in Python, unrelated to any <code>TreePruner</code>s. Our procedure will limit the tree depth - the maximal depth (here defined as the number of internal nodes on any path down the tree) shall be given as an argument. For example, to get a two-level tree, we would call "<code>cutTree(root, 2)</code>". The function will be recursive, with the second argument (level) decreasing at each call; when zero, the current node will be made a leaf.</code>

<p class="header">part of <a href="treestructure.py">treestructure.py</a>
(uses <a href="lenses.tab">lenses.tab</a>)</p>
<xmp class="code">def cutTree(node, level):
    if node and node.branchSelector:
        if level:
            for branch in node.branches:
                cutTree(branch, level-1)
        else:
            node.branchSelector = None
            node.branches = None
            node.branchDescriptions = None
</xmp>

<p>There's nothing to prune at null-nodes or leaves, so we act only when <CODE>node</CODE> and <CODE>node.branchSelector</CODE> are defined. If level is not zero, we call the function for each branch. Otherwise, we clear the selector, branches and branch descriptions.</p>

<p class="header">part of <a href="treestructure.py">treestructure.py</a>
(uses <a href="lenses.tab">lenses.tab</a>)</p>
<xmp class="code">>>> cutTree(tree.tree, 2)
>>> printTree(tree)

tear_rate (<15.000, 5.000, 4.000>)
: reduced --> none (<12.000, 0.000, 0.000>)
: normal
   astigmatic (<3.000, 5.000, 4.000>)
   : no --> soft (<1.000, 5.000, 0.000>)
   : yes --> hard (<2.000, 0.000, 4.000>)
</xmp>


<H3>Learning</H3>

<p>You've already seen a simple example of using a <code>TreeLearner</code>. You can just call it and let it fill the empty slots with the default components. This section will teach you three things: what are the missing components (and how to set the same components yourself), how to use alternative components to get a different tree and, finally, how to write a skeleton for tree induction in Python.</p>

<H4>Default components for TreeLearner</H4>

<p>Let us construct a <code>TreeLearner</code> to play with.</p>

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


