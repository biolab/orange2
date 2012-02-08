import Orange.feature.imputation

class ImputeTable(object):
    """Imputes missing values in the data table.

    :param data: data to impute.
    :type data: :class:`Orange.data.Table`

    :param method: feature imputation method.
    :type method: :class:`Imputation`

    """
    def __new__(cls, data=None, method=Orange.feature.imputation.AverageConstructor()):
        if data is None:
            self = object.__new__(cls)
            return self
        else:
            self = cls(method=method)
            return self(data)

    def __init__(self, method=Orange.feature.imputation.AverageConstructor()):
        self.method = method

    def __call__(self, data):
        return self.method(data)(data)
