import Orange.core as orange

# from orngFSS
def bestNAtts(scores, N):
  """
  Returns the first N attributes from the list returned by function attMeasure.
  Arguments: scores   a list such as one returned by "attMeasure"
             N             the number of attributes
  Result: the first N attributes (without measures)
  """
  return map(lambda x:x[0], scores[:N])

def attsAbovethreshold(scores, threshold=0.0):
  """
  Returns attributes from the list returned by function attMeasure that
  have the score above or equal to a specified threshold
  Arguments: scores   a list such as one returned by "attMeasure"
             threshold      threshold, default is 0.0
  Result: the first N attributes (without measures)
  """
  pairs = filter(lambda x, t=threshold: x[1] > t, scores)
  return map(lambda x:x[0], pairs)

def selectBestNAtts(data, scores, N):
  """
  Constructs and returns a new set of examples that includes a
  class and only N best attributes from a list scores
  Arguments: data          an example table
             scores   a list such as one returned by "attMeasure"
             N             the number of attributes
  Result: data with the first N attributes (without measures)
  """
  return data.select(bestNAtts(scores, N)+[data.domain.classVar.name])


def selectAttsAboveThresh(data, scores, threshold=0.0):
  """
  Constructs and returns a new set of examples that includes a
  class and attributes from the list returned by function attMeasure that
  have the score above or equal to a specified threshold
  Arguments: data          an example table
             scores      a list such as one returned by "attMeasure"
             threshold      threshold, default is 0.0
  Result: the first N attributes (without measures)
  """
  return data.select(attsAbovethreshold(scores, threshold)+[data.domain.classVar.name])

def filterRelieff(data, measure = orange.MeasureAttribute_relief(k=20, m=50), margin=0):
  """
  Takes the data set and an attribute measure (Relief by default). Estimates
  attibute score by the measure, removes worst attribute if its measure
  is below the margin. Repeats, until no attribute has negative or zero score.
  Arguments: data          an example table
             measure       an attribute measure (derived from mlpy.MeasureAttribute)
             margin        if score is higher than margin, attribute is not removed
  """
  measl = attMeasure(data, measure)
  
  while len(data.domain.attributes)>0 and measl[-1][1]<margin:
    data = selectBestNAtts(data, measl, len(data.domain.attributes)-1)
#    print 'remaining ', len(data.domain.attributes)
    measl = attMeasure(data, measure)
  return data

##############################################################################
# wrappers

def FilterAttsAboveThresh(data=None, **kwds):
  filter = apply(FilterAttsAboveThresh_Class, (), kwds)
  if data: return filter(data)
  else: return filter
  
class FilterAttsAboveThresh_Class:
  def __init__(self, measure=orange.MeasureAttribute_relief(k=20, m=50), threshold=0.0):
    self.measure = measure
    self.threshold = threshold
  def __call__(self, data):
    ma = attMeasure(data, self.measure)
    return selectAttsAboveThresh(data, ma, self.threshold)

#

def FilterBestNAtts(data=None, **kwds):
  filter = apply(FilterBestNAtts_Class, (), kwds)
  if data: return filter(data)
  else: return filter
  
class FilterBestNAtts_Class:
  def __init__(self, measure=orange.MeasureAttribute_relief(k=20, m=50), n=5):
    self.measure = measure
    self.n = n
  def __call__(self, data):
    ma = attMeasure(data, self.measure)
    self.n = min(self.n, len(data.domain.attributes))
    return selectBestNAtts(data, ma, self.n)

#

def FilterRelief(data=None, **kwds):
  filter = apply(FilterRelief_Class, (), kwds)
  if data: return filter(data)
  else: return filter
  
class FilterRelief_Class:
  def __init__(self, measure=orange.MeasureAttribute_relief(k=20, m=50), margin=0):
    self.measure = measure
    self.margin = margin
  def __call__(self, data):
    return filterRelieff(data, self.measure, self.margin)

##############################################################################
# wrapped learner

def FilteredLearner(baseLearner, examples = None, weight = None, **kwds):
  learner = apply(FilteredLearner_Class, [baseLearner], kwds)
  if examples: return learner(examples, weight)
  else: return learner

class FilteredLearner_Class:
  def __init__(self, baseLearner, filter=FilterAttsAboveThresh(), name='filtered'):
    self.baseLearner = baseLearner
    self.filter = filter
    self.name = name
  def __call__(self, data, weight=0):
    # filter the data and then learn
    fdata = self.filter(data)
    model = self.baseLearner(fdata, weight)
    return FilteredClassifier(classifier = model, domain = model.domain)

class FilteredClassifier:
  def __init__(self, **kwds):
    self.__dict__.update(kwds)
  def __call__(self, example, resultType = orange.GetValue):
    return self.classifier(example, resultType)
  def atts(self):
    return self.domain.attributes  
