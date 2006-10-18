import orange
import stats

class OutlierDetection:
  
  def __init__(self):
    self.clear()
    self.setKNN()
  
  def clear(self):
    #distmatrix not calculated yet
    self.distmatrixC = 0
    
    #using distance measurment
    self.distance = None
    
    self.examples = None
    self.distmatrix = None
        
  def setExamples(self, examples, distance = None):
    self.clear()
    self.examples = examples
    if (distance == None):
      distance = orange.ExamplesDistanceConstructor_Manhattan(self.examples)
    self.distance = distance

  def setDistanceMatrix(self, distances):
    self.clear()
    self.distmatrix = distances
    self.distmatrixC = 1
  
  def setKNN(self, knn=0):
    self.knn = knn
  
  def calcDistanceMatrix(self):
    #other distance measures
    self.distmatrix = orange.SymMatrix(len(self.examples))
    for i in range(len(self.examples)):
       for j in range(i+1):
         self.distmatrix[i, j] = self.distance(self.examples[i], self.examples[j])
    self.distmatrixC = 1
  
  def distanceMatrix(self):
    if (self.distmatrixC == 0): 
      self.calcDistanceMatrix()
    return self.distmatrix
   
  def averageMeans(self):
    means = []
    dm = self.distanceMatrix()
    for i,dist in enumerate(dm):
      nearest = self.findNearestLimited(i, dist, self.knn)
      means.append(self.average(nearest))
    return means
  
  def average(self, list):
    av = 0.0
    for el in list:
      av = av + el
    return av/len(list)
    
  def findNearestLimited(self, i, dist, knn):
    copy = []
    for el in dist:
      copy.append(el)
    #remove distance to same element
    copy[i:i+1] = []
    if (knn == 0):
      return copy
    else:
      takelimit = min(len(dist)-1, knn)
      copy.sort()
      return copy[:takelimit]
              
  def zValues(self):
    #stats.lzs crashes if all values are 
    list = self.averageMeans()
    if (self.allsame(list)):
      return self.listzero(len(list))
    else:
      return stats.lzs(list)
   
  def allsame(self, list):
    firstel = None
    for el in list:
      if firstel == None:
        firstel = el
      else:
        if (firstel <> el):
          return 0
    return 1
 
  def listzero(self, length):
    ret = []
    for el in range(length):
      ret.append(0.0)
    return ret
    
 