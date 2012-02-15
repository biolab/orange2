.. automodule:: Orange.data.sample

=================================
Random sampling data (``sample``)
=================================

Random sampling is done by constructing a vector of subset indices
(e.g. a table of 0's and 1's), one corresponding to each instance, and
then passing the vector to the table's :obj:`Orange.data.Table.select`
method.
 
Orange provides several methods for construction of such indices:
:obj:`SubsetIndices2` for splitting into two sets (or extracting a
random subset), :obj:`SubsetIndicesN` for splitting into multiple
sets and :obj:`SubsetIndicesCV` for cross validation. All classes are
derived from the abstract class :obj:`SubsetIndices`.

The typical usage pattern is as follows. ::

    lenses = Orange.data.Table("lenses")
    indices2 = Orange.data.sample.SubsetIndices2(p0=0.25)
    ind = indices2(lenses)
    lenses0 = lenses.select(ind, 0)
    lenses1 = lenses.select(ind, 1)

Subset indices are deterministic in the sense that unless the caller
explicitly modifies random seeds, the same setup will always return
the same indices. Details are shown in the section about
:obj:`SubsetIndices2`.
 
.. class:: SubsetIndices

    .. attribute:: stratified

        Defines whether the samples should be stratified, that is,
        whether all subset should have approximatelly equal class
        distributions. Possible values are

	.. data:: Stratified

            Division is stratified; exceptions is raised if this is
            not possible, for instance if the data is numeric.

	.. data:: NotStratified

            Division is not stratified.

	.. data:: StratifiedIfPossible

            Division is stratified if possible and unstratified
            otherwise (default).

    .. attribute:: randseed
    
    .. attribute:: random_generator

        If :obj:`random_generator` (of type :obj:`Orange.misc.Random`)
        is set, it is used for generation of random numbers. In this
        case, :obj:`SubsetIndices` will return a different set of
        indices each time it is called.

        The same generator can be shared between different objects;
        this can be useful when constructing an experiment that
        depends on a single random seed.

        If :obj:`random_generator` is not given, but :attr:`randseed`
        is set (that is, positive), the value is used to initiate a
        new, temporary local random generator. This way, the indices
        generator will always give same indices for the same data.

        If none of the two is defined, a new random generator is
        constructed each time the object is called and initialized
        with a seed of 0. Note that this is different from some other
        classes, such as :obj:`~Orange.data.feature.Descriptor`,
        :obj:`~Orange.statistics.distribution.Distribution` and
        :obj:`~Orange.data.Table`, that store such generators for
        future use: the generator constructed by :obj:`SubsetIndices`
        is disposed after use) and initialized with random seed
        0.

        Examples are shown in documentation for :obj:`SubsetIndices2`.

    .. method:: __call__(data)

        Return a list of indices for the given data table. If data has
        a discrete class, sampling can be stratified.

    .. method:: __call__(n)

        Return a list of ``n`` indices. Sampling cannot be stratified.

.. class:: SubsetIndices2

    Prepares a list of 0's and 1's in the given proportions.
 
    .. attribute:: p0

        The proportion or a number of 0's. If :obj:`p0` is less than
        1, the number gives a proportion; for instance, if :obj:`p0`
        is 0.2, 20% of indices will be 0's and 80% will be 1's. If
        :obj:`p0` is 1 or more, it gives the number of 0's; with
        :obj:`p0=10`, the list will have 10 0's and the rest of the
        list will be 1's.
 
    The following examples splits the data on lenses to two datasets,
    the first containing only 6 data instances and the other
    containing the rest (from :download:`randomindices2.py
    <code/randomindices2.py>`):

    .. literalinclude:: code/randomindices2.py
	:lines: 11-17

    Output::

	<0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1>
	6 18

    Repeating this gives the same set of indices.

    .. literalinclude:: code/randomindices2.py
	:lines: 19-21

    Output::

	Indices without playing with random generator
	<0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1>
	<0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1>
	<0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1>
	<0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1>
	<0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1>

    With a random generator, it gives different indices every time.

    .. literalinclude:: code/randomindices2.py
	:lines: 23-26

    Output::

	Indices with random generator
	<1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1>
	<1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1>
	<1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1>
	<1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0>
	<1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1>

    Running this same script again however gives the same indices
    since the same random generator is constructed and used.

    The next example sets the random seed and removes the random
    generator (otherwise the seed would have no effect as the
    generator has the priority). At each call, it constructs a private
    random generator and initializes it with the given seed, and
    therefore always returns the same indices.

    .. literalinclude:: code/randomindices2.py
	:lines: 28-32

    Output::

	Indices with randseed
	<1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1>
	<1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1>
	<1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1>
	<1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1>
	<1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1>

    There are 24 instances in the dataset. Setting
    :obj:`SubsetIndices2.p0` to 0.25 instead of 6 gives the same result.

    .. literalinclude:: code/randomindices2.py
	:lines: 35-37

    Output::

	Indices with p0 set as probability (not 'a number of')
	<1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1>

    The class can also be called with a number of data instances
    instead of the data. In this case, stratification is not possible.

    .. literalinclude:: code/randomindices2.py
	:lines: 64-66

    Output::

	... stratified 'if possible'
	<1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1>
 
.. class:: SubsetIndicesN

    A generalization of :obj:`RandomIndices2` to multiple subsets.

    .. attribute:: p

        A list of proportions of data that go to each fold. If
        :obj:`p` has a length of 3, the returned list will have four
        different indices, the first three will have probabilities as
        defined in :obj:`p` while the last will have a probability of
        (1 - sum of elements of :obj:`p`).

    :obj:`SubsetIndicesN` does not support stratification; setting
    :obj:`stratified` to :obj:`Stratified` will yield an error.

    The following constructs a division in which one half of data is
    in the first set and one quarter in the second and in the third
    :download:`randomindicesn.py <code/randomindicesn.py>`).

    .. literalinclude:: code/randomindicesn.py
        :lines: 9-14

    Output::

        <1, 0, 0, 2, 0, 1, 1, 0, 2, 0, 2, 2, 1, 0, 0, 0, 2, 0, 0, 0, 1, 2, 1, 0>


.. class:: SubsetIndicesCV
 
    Computes indices for cross-validation by constructing a list of
    indices between 0 and :obj:`folds`-1 (inclusive), with an equal
    number of each (if the number of instances is not divisible by
    :obj:`folds`, the last folds will have one element less).

    .. attribute:: folds

        Number of folds. Default is 10.
 
    This prepares indices for ten-fold cross validation and indices
    for 10 data instances for 5-fold cross validation without giving
    the actual data in the latter case (:download:`randomindicescv.py
    <code/randomindicescv.py>`).

    .. literalinclude:: code/randomindicescv.py
	:lines: 7-12

    Output::
	Indices for ordinary 10-fold CV
	<1, 1, 3, 8, 8, 3, 2, 7, 5, 0, 1, 5, 2, 9, 4, 7, 4, 9, 3, 6, 0, 2, 0, 6>
	Indices for 5 folds on 10 instances
	<3, 0, 1, 0, 3, 2, 4, 4, 1, 2>

    Since instances do not divide evenly into ten folds, the first
    four folds have one element more - there are three 0's, 1's, 2's
    and 3's, but only two 4's, 5's..
