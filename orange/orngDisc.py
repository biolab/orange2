import orange

def entropyDiscretization(data):
  """discretize continuous attributes, removing those discretized to a constant"""
  orange.setrandseed(0)
  tablen=orange.Preprocessor_discretize(data, method=orange.EntropyDiscretization())

  attrlist=[a for a in tablen.domain.attributes if len(a.values)>1]
  attrlist.append(tablen.domain.classVar)
  return tablen.select(attrlist)

class EntropyDiscretization:
  def __call__(self, data):
    return entropyDiscretization(data)


def DiscretizedLearner(baseLearner, examples=None, weight=0, **kwds):
  learner = apply(DiscretizedLearner_Class, [baseLearner], kwds)
  if examples: return learner(examples, weight)
  else: return learner

class DiscretizedLearner_Class:
  def __init__(self, baseLearner, discretizer=EntropyDiscretization(), **kwds):
    self.baseLearner = baseLearner
    if hasattr(baseLearner, "name"):
      self.name = baseLearner.name
    self.discretizer = discretizer
    self.__dict__.update(kwds)
  def __call__(self, data, weight=None):
    # filter the data and then learn
    ddata = self.discretizer(data)
    if weight<>None:
      model = self.baseLearner(ddata, weight)
    else:
      model = self.baseLearner(ddata)
    dcl = DiscretizedClassifier(classifier = model)
    if hasattr(model, "domain"):
      dcl.domain = model.domain
    if hasattr(model, "name"):
      dcl.name = model.name
    return dcl

class DiscretizedClassifier:
  def __init__(self, **kwds):
    self.__dict__ = kwds
  def __call__(self, example, resultType = orange.GetValue):
    return self.classifier(example, resultType)
