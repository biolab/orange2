.. py:currentmodule:: Orange.core

Value transformers are objects that take care of simple transformations
of values. Discretization, for instance, creates a transformer that
converts continuous values into discrete, while continuizers do the
opposite. Classification trees use transformers for binarization where
values of discrete attributes are converted into binary.

These objects are most often constructed by other classes and only seldom
manually. See information on :file:`Orange.data.discretization` and
:file:`Orange.data.continuization`.

.. class TransformValue

    The abstract root of the hierarchy of transformers provides the call
    operator and chaining of transformers.

    .. attribute subtransformer

        The transformation that takes place prior to this.
        This way, transformations can be chained.


.. class Ordinal2Continuous

    Converts ordinal values to continuous. For example, variable values
    values `small`, `medium`, `large`, `extra large` (if given in
    that order) would be, by default, converted to 0.0, 1.0, 2.0 and 3.0.
    It is possible to add a factor by which the values are multiplied. If
    the factor for the above case were 0.3333, the value would be
    converted to 0, 0.3333, 0.6666 and 0.9999.

    .. attribute factor

        The factor by which the values are multiplied.

    .. literalinclude:: transformvalue-o2c.py
        :lines: 7-23

    The values of attribute `age` (`young`, `pre-presbyopic` and
    `presbyopic`) are transformed to 0.0, 1.0 and 2.0 in `age_c` and to
    0, 0.5 and 1 in `age_cn`.


.. class Discrete2Continuous

    Converts a discrete value to a continuous so that some chosen
    value is converted to 1.0 and all others to 0.0 or -1.0, depending on
    the settings.

    .. attribute value

        The value that in converted to 1.0; others are converted to 0.0
        or -1.0. Value needs to be specified by an integer index.

    .. attribute zero_based

        Decides whether the other values will be transformed to 0.0
        (``True``, default) or -1.0 (``False``).
        When ``False`` undefined values are transformed to 0.0;
        otherwise, undefined values yield an error.

    .. attribute invert

        If ``True`` (default is ``False``), the transformations are
        reversed - the selected ``value<`` becomes 0.0 (or -1.0)
        and others 1.0.


    The following script load the Monks 1 data set and constructs a new
    attribute `e1` that will indicate whether `e` is 1 or not.

    .. literalinclude transformvalue-d2c.py


.. class NormalizeContinuous

    Takes a continuous values and subtracts the ``average`` and
    divides the difference by half of the ``span``.

    .. attribute average

        The value that is subtracted from the original.</DD>

    .. span

        The divisor

    The following script "normalizes" all attribute in the Iris dataset by
    subtracting the average value and dividing by the half of
    deviation.

    .. literalinclude transformvalue-nc.py
        :lines: 1-17

.. class MapIntValue

    A discrete-to-discrete transformer that changes values according to the
    given mapping. MapIntValue is used for binarization in decision trees.

    .. attribute mapping

        A mapping that determines the new value: ``v = mapping[v]``.
        Undefined values remain undefined. Elements of the mapping
        are  contains integer indices of values.

    The following script transforms the value of `age` in dataset lenses
    from 'young' to 'young', and from 'pre-presbyopic' and 'presbyopic' to
    'old'.

    .. literalinclude transformvalue-miv.py
        :lines: 1-12

    The mapping tells that the 0th value of `age` maps to the 0th of
    `age_b`, and the 1st and 2nd value go to the 1st value of `age_b`.
