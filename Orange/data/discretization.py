import Orange

from Orange.core import\
    EquiNDiscretization as EqualFreq,\
    BiModalDiscretization as BiModal,\
    Preprocessor_discretize

class DiscretizeTable(object):
    """Discretizes all continuous features of the data table.

    :param data: data to discretize.
    :type data: :class:`Orange.data.Table`

    :param features: data features to discretize. None (default) to discretize all features.
    :type features: list of :class:`Orange.data.variable.Variable`

    :param method: feature discretization method.
    :type method: :class:`Discretization`

    :param clean: clean the data domain after discretization. If True, features discretized to a constant will be
      removed. Useful only for discretizers which infer number of discretization intervals from data,
      like :class:`Orange.feature.discretize.Entropy` (default: True).
    :type clean: boolean

    """
    def __new__(cls, data=None, features=None, discretize_class=False, method=EqualFreq(n=3), clean=True):
        if data is None:
            self = object.__new__(cls)
            return self
        else:
            self = cls(features=features, discretize_class=discretize_class, method=method, clean=clean)
            return self(data)

    def __init__(self, features=None, discretize_class=False, method=EqualFreq(n=3), clean=True):
        self.features = features
        self.discretize_class = discretize_class
        self.method = method
        self.clean = clean

    def __call__(self, data):
        pp = Preprocessor_discretize(attributes=self.features, discretizeClass=self.discretize_class)
        pp.method = self.method
        ddata = pp(data)

        if self.clean:
            return ddata.select([x for x in ddata.domain.features if len(x.values)>1] + [ddata.domain.classVar])
        else:
            return ddata
