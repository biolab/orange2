import orange

def entropyDiscretization(data):
  """
  Discretizes the continuous attributes using the entropy based discretization.
  It removes the attributes discretized to a single interval and prints their names.
  Arguments: data         an example table
  Results:   a table of examples with discretized atributes. attributes that are
             categorized to a single value (constant) are removed
  """
  orange.setrandseed(0)
  tablen=orange.Preprocessor_discretize(data, method=orange.EntropyDiscretization())

  # print "\nAttributes removed by discretization:\n",
  attrlist=[]
  nrem=0
  for i in tablen.domain.attributes:
    if (len(i.values)>1):
      attrlist.append(i)
    else:
      # if nrem: print ", %s" % i.name[2:],
      # else: print "%s" % i.name[2:],
      nrem=nrem+1
  # if not nrem: print "(None)",
  # print

  attrlist.append(tablen.domain.classVar)

  # print "Retained attributes:"
  # for i in attrlist: print i.name,
  # print
  
  return tablen.select(attrlist)

#

class EntropyDiscretization:
  def __call__(self, data):
    return entropyDiscretization(data)

#

def DiscretizedLearner(baseLearner, examples=None, weight=0, **kwds):
  learner = apply(DiscretizedLearner_Class, [baseLearner], kwds)
  if examples: return learner(examples, weight)
  else: return learner

class DiscretizedLearner_Class:
  def __init__(self, baseLearner, discretizer=EntropyDiscretization(), **kwds):
    self.baseLearner = baseLearner
    if hasattr(baseLearner, "name"):
      learner.name = baseLearner.name
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
