import Orange

from Orange.core import Preprocessor_discretize

class DiscretizeTable(object):
    """Discretizes all continuous features of the data table.

    :param data: Data to discretize.
    :type data: :class:`Orange.data.Table`

    :param features: Data features to discretize. `None` (default) to
        discretize all features.
    :type features: list of :class:`Orange.feature.Descriptor`

    :param method: Feature discretization method.
    :type method: :class:`Orange.feature.discretization.Discretization`

    :param clean: Clean the data domain after discretization. If `True`,
        features discretized to a constant will be removed. Useful only
        for discretizers which infer number of discretization intervals
        from data, like :class:`Orange.feature.discretize.Entropy`
        (default: `True`).
    :type clean: bool

    """
    def __new__(cls, data=None, features=None, discretize_class=False,
                method=Orange.feature.discretization.EqualFreq(n=3), clean=True):
        if data is None:
            self = object.__new__(cls)
            return self
        else:
            self = cls(features=features, discretize_class=discretize_class,
                       method=method, clean=clean)
            return self(data)

    def __init__(self, features=None, discretize_class=False,
                 method=Orange.feature.discretization.EqualFreq(n=3), clean=True):
        self.features = features
        self.discretize_class = discretize_class
        self.method = method
        self.clean = clean

    def __call__(self, data):
        pp = Preprocessor_discretize(attributes=self.features,
                                     discretize_class=self.discretize_class)
        pp.method = self.method
        ddata = pp(data)

        if self.clean:
            features = [x for x in ddata.domain.features if len(x.values) > 1]
            domain = Orange.data.Domain(features, ddata.domain.class_var,
                                        class_vars=ddata.domain.class_vars)
            return Orange.data.Table(domain, ddata)
        else:
            return ddata
