### Janez 03-02-14: The two classes for removal of redundant values
###                 were merged and moved to orngCI
### Inform Blaz and remove this comment


import orange

##############################################################################
# utility functions

def attMeasure(data, measure = orange.MeasureAttribute_relief(k=20, m=50)):
  """
  Assesses the quality of attributes using the given measure, outputs the results and
  returns a sorted list of tuples (attribute name, measure)
  Arguments: data           example table
             measure        an attribute measure (derived from orange.MeasureAttribute)
  Result:    a sorted list of tuples (attribute name, measure)
  """
  measl=[]
  for i in data.domain.attributes:
    measl.append((i.name, measure(i, data)))
  measl.sort(lambda x,y:cmp(y[1], x[1]))
   
#  for i in measl:
#    print "%25s, %6.3f" % (i[0], i[1])
  return measl

def bestNAtts(relevancies, N):
  """
  Returns the first N attributes from the list returned by function attMeasure.
  Arguments: relevancies   a list such as one returned by "attMeasure"
             N             the number of attributes
  Result: the first N attributes (without measures)
  """
  return map(lambda x:x[0], relevancies[:N])

def attsAbovethreshold(relevancies, threshold=0.0):
  """
  Returns attributes from the list returned by function attMeasure that
  have the relevancy above or equal to a specified threshold
  Arguments: relevancies   a list such as one returned by "attMeasure"
             threshold      threshold, default is 0.0
  Result: the first N attributes (without measures)
  """
  pairs = filter(lambda x, t=threshold: x[1] >= t, relevancies)
  return map(lambda x:x[0], pairs)

def selectBestNAtts(data, relevancies, N):
  """
  Constructs and returns a new set of examples that includes a
  class and only N best attributes from a list relevancies
  Arguments: data          an example table
             relevancies   a list such as one returned by "attMeasure"
             N             the number of attributes
  Result: data with the first N attributes (without measures)
  """
  return data.select(bestNAtts(relevancies, N)+[data.domain.classVar.name])


def selectAttsAboveThresh(data, relevancies, threshold=0.0):
  """
  Constructs and returns a new set of examples that includes a
  class and attributes from the list returned by function attMeasure that
  have the relevancy above or equal to a specified threshold
  Arguments: data          an example table
             relevancies      a list such as one returned by "attMeasure"
             threshold      threshold, default is 0.0
  Result: the first N attributes (without measures)
  """
  return data.select(attsAbovethreshold(relevancies, threshold)+[data.domain.classVar.name])


def filterRelieff(data, measure = orange.MeasureAttribute_relief(k=20, m=50), margin=0):
  """
  Takes the data set and an attribute measure (Relief by default). Estimates
  attibute relevancy by the measure, removes worst attribute if its measure
  is below the margin. Repeats, until no attribute has negative or zero relevancy.
  Arguments: data          an example table
             measure       an attribute measure (derived from mlpy.MeasureAttribute)
             margin        if relevance is higher than margin, attribute is not removed
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

# TODO: wrapper around recursive Relief, try it out  

##############################################################################
# wrapped learner

def FilteredLearner(baseLearner, examples=None, weight = 0, **kwds):
  learner = apply(FilteredLearner_Class, [baseLearner], kwds)
  if examples: return learner(examples, weight)
  else: return learner

class FilteredLearner_Class:
  def __init__(self, baseLearner, filter=FilterAttsAboveThresh(), name='filtered'):
    self.baseLearner = baseLearner
    self.filter = filter
    self.name = name
  def __call__(self, data, weight):
    # filter the data and then learn
    fdata = self.filter(data)
    model = self.baseLearner(fdata, weight)
    return FilteredClassifier(classifier = model, domain = model.domain)

class FilteredClassifier:
  def __init__(self, **kwds):
    self.__dict__ = kwds
  def __call__(self, example, resultType = orange.GetValue):
    return self.classifier(example, resultType)
  def atts(self):
    return self.domain.attributes
  
