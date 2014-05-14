from Orange import core


class PyLearner(object):
    def __new__(cls, data=None, **kwds):
        learner = object.__new__(cls)
        if data is not None:
            learner.__init__(**kwds) # force init
            return learner(data)
        else:
            return learner  # invokes the __init__

    def __init__(self, name='learner'):
        self.name = name

    def __call__(self, data, weight=None):
        return None


class PyRegression:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __call__(self, example, result_type=core.GetValue):
        return self.regression(example, result_type)
