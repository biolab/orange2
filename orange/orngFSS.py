import orange

##############################################################################
# utility functions

def attMeasure(data, measure = orange.MeasureAttribute_relief(k=20, m=50)):
  """
  Assesses the quality of attributes using the given measure, outputs the results and
  returns a sorted list of tuples (attribute name, measure)
  Arguments: data       example table
             measure    an attribute scoring function (derived from orange.MeasureAttribute)
  Result:    a sorted list of tuples (attribute name, measure)
  """
  measl=[]
  for i in data.domain.attributes:
    measl.append((i.name, measure(i, data)))
  measl.sort(lambda x,y:cmp(y[1], x[1]))
   
#  for i in measl:
#    print "%25s, %6.3f" % (i[0], i[1])
  return measl

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

class StepwiseLearner_Class:
  def __init__(self, **kwds):
    import orngStat
    self.removeThreshold = 0.3
    self.addThreshold = 0.2
    self.stat, self.statsign = orngStat.CA, 1
    self.__dict__.update(kwds)

  def __call__(self, examples, weightID = 0, **kwds):
    import orngTest, orngStat, statc
    
    self.__dict__.update(kwds)

    if self.removeThreshold < self.addThreshold:
        raise "'removeThreshold' should be larger or equal to 'addThreshold'"

    classVar = examples.domain.classVar
    
    indices = orange.MakeRandomIndicesCV(examples, folds = getattr(self, "folds", 10))
    domain = orange.Domain([], classVar)

    res = orngTest.testWithIndices([self.learner], orange.ExampleTable(domain, examples), indices)
    
    oldStat = self.stat(res)[0]
    oldStats = [self.stat(x)[0] for x in orngStat.splitByIterations(res)]
    print ".", oldStat, domain
    stop = False
    while not stop:
        stop = True
        if len(domain.attributes)>=2:
            bestStat = None
            for attr in domain.attributes:
                newdomain = orange.Domain(filter(lambda x: x!=attr, domain.attributes), classVar)
                res = orngTest.testWithIndices([self.learner], (orange.ExampleTable(newdomain, examples), weightID), indices)
                
                newStat = self.stat(res)[0]
                newStats = [self.stat(x)[0] for x in orngStat.splitByIterations(res)] 
                print "-", newStat, newdomain
                ## If stat has increased (ie newStat is better than bestStat)
                if not bestStat or cmp(newStat, bestStat) == self.statsign:
                    if cmp(newStat, oldStat) == self.statsign:
                        bestStat, bestStats, bestAttr = newStat, newStats, attr
                    elif statc.wilcoxont(oldStats, newStats)[1] > self.removeThreshold:
                            bestStat, bestAttr, bestStats = newStat, newStats, attr
            if bestStat:
                domain = orange.Domain(filter(lambda x: x!=bestAttr, domain.attributes), classVar)
                oldStat, oldStats = bestStat, bestStats
                stop = False
                print "removed", bestAttr.name

        bestStat, bestAttr = oldStat, None
        for attr in examples.domain.attributes:
            if not attr in domain.attributes:
                newdomain = orange.Domain(domain.attributes + [attr], classVar)
                res = orngTest.testWithIndices([self.learner], (orange.ExampleTable(newdomain, examples), weightID), indices)
                
                newStat = self.stat(res)[0]
                newStats = [self.stat(x)[0] for x in orngStat.splitByIterations(res)] 
                print "+", newStat, newdomain

                ## If stat has increased (ie newStat is better than bestStat)
                if cmp(newStat, bestStat) == self.statsign and statc.wilcoxont(oldStats, newStats)[1] < self.addThreshold:
                    bestStat, bestStats, bestAttr = newStat, newStats, attr
        if bestAttr:
            domain = orange.Domain(domain.attributes + [bestAttr], classVar)
            oldStat, oldStats = bestStat, bestStats
            stop = False
            print "added", bestAttr.name

    return self.learner(orange.ExampleTable(domain, examples), weightID)

def StepwiseLearner(examples = None, weightID = None, **argkw):
    sl = apply(StepwiseLearner_Class, (), argkw)
    if examples:
        return sl(examples, weightID)
    else:
        return sl
