##############################
Preprocessing (``preprocess``)
##############################

.. automodule:: Orange.data.preprocess

.. autoclass:: DiscretizeEntropy(method=Orange.feature.discretization.Entropy())

.. autoclass:: RemoveContinuous

.. autoclass:: Continuize

.. autoclass:: RemoveDiscrete

.. autoclass:: Impute

.. autoclass:: FeatureSelection(measure=Orange.feature.scoring.Relief(), filter=None, limit=10)

.. autofunction:: bestP

.. autofunction:: bestN

.. autofunction:: selectNRandom

.. autofunction:: selectPRandom

.. autoclass:: RFE

.. autoclass:: Sample

.. autoclass:: PreprocessorList

.. class:: RemoveUnusedValues(variable, data, remove_one_valued=False)

    Removes unused values and reduces the variable, if a variable
    declares values that do not appear in the data.

    :param variable: :obj:`~Orange.feature.Descriptor`
    :param data: :obj:`~Orange.data.Table`
    :param remove_one_valued: Decides whether to remove or to retain
        the attributes with only one value defined (default: False).

    Example:

    .. literalinclude:: code/unusedValues.py

    There are four possible outcomes:

    1. The variable does not have any used values in the data - value
    of this variable is undefined for all examples. The variable is
    thus useless and the class returns None.

    2. The variable has only one used value (or, possibly, only one
    value at all). Such a variable is in fact useless, and can
    probably be removed without harm. Nevertheless, its fate is
    decided by the flag remove_one_valued which is False by default,
    so such variables are retained unless explicitly specified
    otherwise.

    3. All variable's values occur in the data (and the variable has more
    than one value; otherwise the above case applies). The original variable
    is returned.

    4. There are some unused values. A new variable is constructed and the
    unused values are omitted. The value of the new variable is computed
    automatically from the value of the original variable
    :class:`~Orange.classification.lookup.ClassifierByLookupTable` is used
    for mapping.

    Results of example:

    .. literalinclude:: code/unusedValues.res

    Variables a and y are OK and are left alone. In b, value 1 is not used
    and is removed (not in the original variable, of course; a new variable
    is created). c is useless and is removed altogether. d is retained since
    remove_one_valued was left at False; if we set it to True, this variable
    would be removed as well.


.. automodule:: Orange.data.preprocess.scaling
