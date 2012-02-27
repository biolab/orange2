"""\
=======================
Base regression learner
=======================

.. index:: regression

.. autoclass:: BaseRegressionLearner
    :members:

"""

import Orange

class BaseRegressionLearner(Orange.core.Learner):
    """Fitting regressors typically requires data that has only
    continuous-valued features and no missing values. This class
    provides methods for appropriate transformation of the data and
    serves as a base class for most regressor classes.
    """

    def __new__(cls, table=None, weight_id=None, **kwds):
        learner = Orange.core.Learner.__new__(cls, **kwds)
        if table is not None:
            learner.__init__(**kwds)
            return learner(table, weight_id)
        else:
            return learner

    def __init__(self, imputer=None, continuizer=None):
        self.imputer = None
        self.continuizer = None
        
        self.set_imputer(imputer)
        self.set_continuizer(continuizer)
        

    def set_imputer(self, imputer=None):
        """ Set the imputer for missing data.

        :param imputer: function which constructs the imputer for the
            missing values, if ``None``, the default imputer replaces
            missing continuous data with the average of the
            corresponding variable and missing discrete data with the
            most frequent value.
        :type imputer: None or Orange.feature.imputation.ModelConstructor
        """
        if imputer is not None:
            self.imputer = imputer
        else: # default imputer
            self.imputer = Orange.feature.imputation.ModelConstructor()
            self.imputer.learner_continuous = Orange.regression.mean.MeanLearner()
            self.imputer.learner_discrete = Orange.classification.majority.MajorityLearner()

    def set_continuizer(self, continuizer=None):
        """Set the continuizer of the discrete variables

        :param continuizer: function which replaces the categorical
            (dicrete) variables with numerical variables. If ``None``,
            the default continuizer is used
        :type continuizer: None or Orange.data.continuization.DomainContinuizer
        """
        if continuizer is not None:
            self.continuizer = continuizer
        else: # default continuizer
            self.continuizer = Orange.data.continuization.DomainContinuizer()
            self.continuizer.multinomial_treatment = self.continuizer.FrequentIsBase
            self.continuizer.zero_based = True

    def impute_table(self, table):
        """Impute missing values and return a new
        :class:`Orange.data.Table` object

        :param table: data instances.
        :type table: :class:`Orange.data.Table`
        """
        if table.has_missing_values():
            imputer = self.imputer(table)
            table = imputer(table)
        return table

    def continuize_table(self, table):
        """Replace discrete variables with continuous and return a new
        instance of :class:`Orange.data.Table`.

        :param table: data instances.
        :type table: :class:`Orange.data.Table` 
        """
        if table.domain.has_discrete_attributes():
            new_domain = self.continuizer(table)
            table = table.translate(new_domain)
        return table
