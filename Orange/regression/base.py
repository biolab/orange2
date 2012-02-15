"""\
====================================
Basic regression learner (``basic``)
====================================

.. index:: regression

.. autoclass:: BaseRegressionLearner
    :members:

"""

import Orange

class BaseRegressionLearner(Orange.core.Learner):
    """ Base Regression Learner "learns" how to treat the discrete
        variables and missing data.
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
        """ Sets the imputer for missing values.

        :param imputer: function which imputes the missing values,
            if None, the default imputer: mean for the continuous variables
            and most frequent value (majority) for discrete variables
        :type imputer: None or Orange.feature.imputation.ModelConstructor
        """
        if imputer is not None:
            self.imputer = imputer
        else: # default imputer
            self.imputer = Orange.feature.imputation.ModelConstructor()
            self.imputer.learner_continuous = Orange.regression.mean.MeanLearner()
            self.imputer.learner_discrete = Orange.classification.majority.MajorityLearner()

    def set_continuizer(self, continuizer=None):
        """ Sets the continuizer of the discrete variables

        :param continuizer: function which replaces the categorical (dicrete)
            variables with numerical variables. If None, the default continuizer
            is used
        :type continuizer: None or Orange.data.continuization.DomainContinuizer
        """
        if continuizer is not None:
            self.continuizer = continuizer
        else: # default continuizer
            self.continuizer = Orange.data.continuization.DomainContinuizer()
            self.continuizer.multinomial_treatment = self.continuizer.FrequentIsBase
            self.continuizer.zero_based = True

    def impute_table(self, table):
        """ Imputes missing values.
        Returns a new :class:`Orange.data.Table` object

        :param table: data instances.
        :type table: :class:`Orange.data.Table`
        """
        if table.has_missing_values():
            imputer = self.imputer(table)
            table = imputer(table)
        return table

    def continuize_table(self, table):
        """ Continuizes the discrete variables.
        Returns a new :class:`Orange.data.Table` object

        :param table: data instances.
        :type table: :class:`Orange.data.Table` 
        """
        if table.domain.has_discrete_attributes():
            new_domain = self.continuizer(table)
            table = table.translate(new_domain)
        return table
