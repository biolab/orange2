import orange

def attMeasure(table, meas):
  """
  Assesses the quality of attributes using the given measure, outputs the results and
  returns a sorted list of tuples (attribute name, measure)
  Arguments: table          example table
             measure        an attribute measure (derived from mlpy.MeasureAttribute)
  Result:    a sorted list of tuples (attribute name, measure)
  """
  measl=[]
  for i in table.domain.attributes:
    measl.append((i.name, meas(i, table)[0]))
  measl.sort(lambda x,y:cmp(y[1], x[1]))
   
#  for i in measl:
#    print "%25s, %6.3f" % (i[0], i[1])
  return measl

def bestNAtts(ls, N):
  """
  Returns the first N attributes from the list returned by function attMeasure.
  Arguments: ls            a list such as one returned by "measure"
             N             the number of attributes
  Result: the first N attributes (without measures)
  """
  return map(lambda x:x[0], ls[:N])

def attsAboveTreshold(ls, treshold=0.0):
  """
  Returns attributes from the list returned by function attMeasure that
  have the relevancy above or equal to a specified treshold
  Arguments: ls            a list such as one returned by "measure"
             treshold      treshold, default is 0.0
  Result: the first N attributes (without measures)
  """
  pairs = filter(lambda x, t=treshold: x[1] >= t, ls)
  return map(lambda x:x[0], pairs)

def selectBestNAtts(data, ls, N):
  """
  Constructs and returns a new set of examples that includes a
  class and only N best attributes from a list ls
  Arguments: data          an example table
             ls            a list such as one returned by "measure"
             N             the number of attributes
  Result: the first N attributes (without measures)
  """
  return data.select(bestNAtts(ls, N)+[data.domain.classVar.name])


def selectAttsAboveTresh(data, ls, treshold=0.0):
  """
  Constructs and returns a new set of examples that includes a
  class and ottributes from the list returned by function attMeasure that
  have the relevancy above or equal to a specified treshold
  Arguments: data          an example table
             ls            a list such as one returned by "measure"
             treshold      treshold, default is 0.0
  Result: the first N attributes (without measures)
  """
  return data.select(attsAboveTreshold(ls, treshold)+[data.domain.classVar.name])


def filterRelieff(data, measure = orange.MeasureAttribute_relief(k=20, m=50)):
  """
  Takes the data set and an attribute measure (Relief by default). Estimates
  attibute relevancy by the measure, removes worst attribute if its measure
  is 0 or lower. Repeats, until no attribute has negative or zero relevancy.
  Arguments: data          an example table
             measure       an attribute measure (derived from mlpy.MeasureAttribute)
  """
  measl = attMeasure(data, measure)
  while len(data.domain.attributes)>0 and measl[-1][1]<0:
    data = selectBestNAtts(data, measl, len(data.domain.attributes)-1)
#    print 'remaining ', len(data.domain.attributes)
    measl = attMeasure(data, measure)
  return data
