.. py:currentmodule:: Orange.core

###################################
Continuization (``continuization``)
###################################

Continuization refers to transformation of discrete (binary or
multinominal) variables to continuous. The class described below
operates on the entire domain; documentation on
:file:`Orange.core.transformvalue.rst` explains how to treat each
variable separately.

.. class DomainContinuizer

    Returns a new domain containing only continuous attributes given a
    domain or data table. Some options are available only if the data is
    provided.

    The attributes are treated according to their type:

    * continuous variables can be normalized or left unchanged

    * discrete attribute with less than two possible values are removed;

    * binary variables are transformed into 0.0/1.0 or -1.0/1.0
      indicator variables

    * multinomial variables are treated according to the flag
      ``multinomial_treatment``.

    .. attribute zero_based

        Determines the value used as the "low" value of the variable. When
        binary variables are transformed into continuous or when multivalued
        variable is transformed into multiple variables, the transformed
        variable can either have values 0.0 and 1.0 (default, ``zero_based``
        is ``True``) or -1.0 and 1.0 (``zero_based`` is ``False``). The
        following text assumes the default case.

    .. attribute multinomial_treatment

       Decides the treatment of multinomial variables. Let N be the
       number of the variables's values.

       DomainContinuizer.NValues

           The variable is replaced by N indicator variables, each
           corresponding to one value of the original variable. In other
           words, for each value of the original attribute, only the
           corresponding new attribute will have a value of 1 and others
           will be zero.

           Note that these variables are not independent, so they cannot be
           used (directly) in, for instance, linear or logistic regression.

       DomainContinuizer.LowestIsBase
           Similar to the above except that it creates only N-1
           variables. The missing indicator belongs to the lowest value:
           when the original variable has the lowest value all indicators
           are 0.

	   If the variable descriptor has the ``base_value`` defined, the
           specified value is used as base instead of the lowest one.

       DomainContinuizer.FrequentIsBase

           Like above, except that the most frequent value is used as the
           base (this can again be overidden by setting the descriptor's
           ``base_value``). If there are multiple most frequent values, the
           one with the lowest index is used. The frequency of values is
           extracted from data, so this option cannot be used if constructor
           is given only a domain.
           
       DomainContinuizer.Ignore
           Multivalued variables are omitted.

       DomainContinuizer.ReportError 
           Raise an error if there are any multinominal variables in the data.

       DomainContinuizer.AsOrdinal
           Multivalued variables are treated as ordinal and replaced by a
           continuous variables with the values' index, e.g. 0, 1, 2, 3...

       DomainContinuizer.AsNormalizedOrdinal 
           As above, except that the resulting continuous value will be from
           range 0 to 1, e.g. 0, 0.25, 0.5, 0.75, 1 for a five-valued
           variable.

    .. attribute normalize_continuous

        If ``False`` (default), continues variables are left unchanged. If
        ``True``, they are replaced with normalized values by subtracting
        the average value and dividing by the deviation. Statistics are
        computed from the data, so constructor must be given data, not just
        domain.

    .. attribute class_treatment

        Determines the treatment of discrete class attribute. Continuous
        class attributes are always left unchanged.

        DomainContinuizer.Ignore
           Class attribute is copied as is. Note that this is different
           from the meaning of this value at multinomial_treatment where
           it denotes omitting the attribute.

        DomainContinuizer.AsOrdinal, DomainContinuizer.AsNormalizedOrdinal
           If class is multinomial, it is treated as ordinal, in the
           same manner as described above. Binary classes are
           transformed to 0.0/1.0 attributes.
