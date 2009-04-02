import random
import orange, orngMisc, orngLookup


def __learnConstructor(cls, examples, bound, weightID, argkw):
  fm = apply(cls, (), argkw)
  if examples:
    if not bound:
      raise TypeError, "bound set not given"
    fm = fm(examples, bound, weightID)
  else:
    if bound:
      raise TypeError, "invalid example set"
  return fm

  
######################################################
# Minimal complexity decomposition


def FeatureByMinComplexity(examples=None, bound=None, weightID=0, **argkw):
  return __learnConstructor(FeatureByMinComplexityClass, examples, bound, weightID, argkw)

class FeatureByMinComplexityClass:
  NoCompletion = orange.FeatureByMinComplexity.NoCompletion
  CompletionByDefault = orange.FeatureByMinComplexity.CompletionByDefault
  CompletionByBayes = orange.FeatureByMinComplexity.CompletionByBayes

  def __init__(self, **keyw):
    """(colorIG=, complete=)"""
    self.__dict__ = keyw
    self.instance = None

  def __setattr__(self, name, value):
    if name in ["colorIG", "complete"]:
        self.instance = None
    self.__dict__[name] = value

  def createInstance(self):
    self.instance = orange.FeatureByMinComplexity()
    if hasattr(self, "colorIG"):
      self.instance.colorIG = self.colorIG
    if hasattr(self, "complete"):
      self.instance.complete = self.complete
    return self.instance


  def __call__(self, table, bound, weightID=0):
    if not self.instance:
      self.createInstance()
    return self.instance(table, bound, "", weightID)


######################################################
# Minimal error decomposition (and other IM-based methods)

def FeatureByIM(examples=None, bound=None, weightID=0, **argkw):
  return __learnConstructor(FeatureByIMClass, examples, bound, weightID, argkw)


class FeatureByIMClass:
  NoCompletion = orange.FeatureByIM.NoCompletion
  CompletionByDefault = orange.FeatureByIM.CompletionByDefault
  CompletionByBayes = orange.FeatureByIM.CompletionByBayes

  def __init__(self, **keyw):
    self.__dict__ = keyw
    self.instance = None

  def __setattr__(self, name, value):
    if name in ["IMconstructor", "completion", "clustersFromIM", "stopCriterion", "columnAssessor", "measure", "m"]:
        self.instance = None
    self.__dict__[name] = value

  def createInstance(self):    
    self.instance = fim = orange.FeatureByIM()
       
    fim.IMconstructor = getattr(self, "IMconstructor", orange.IMBySorting())
    cfi = fim.clustersFromIM = getattr(self, "clustersFromIM", orange.ClustersFromIMByAssessor())

    if hasattr(self, "columnAssessor"):
      cfi.columnAssessor = self.columnAssessor
    elif not hasattr(self, "clustersFromIM"):
      if (hasattr(self, "measure")):
        cfi.columnAssessor = getattr(self, "columnAssessor", orange.ColumnAssessor_Measure())
      else:
        cfi.columnAssessor = getattr(self, "columnAssessor", orange.ColumnAssessor_m())

    if hasattr(self, "measure"):
      if not hasattr(cfi.columnAssessor, "measure"):
        raise AttributeError, "invalid combination of columnAssessor arguments (cannot set 'measure')"
      cfi.columnAssessor.measure = self.measure
    elif hasattr(self, "m"):
      if not hasattr(cfi.columnAssessor, "m"):
        raise AttributeError, "invalid combination of columnAssessor arguments (cannot set 'm')"
      cfi.columnAssessor.m = self.m

    if hasattr(self, "stopCriterion"):
      cfi.stopCriterion = self.stopCriterion
    elif not hasattr(self, "clustersFromIM"):
      if hasattr(self, "stopCriterion"):
        cfi.stopCriterion = self.stopCriterion
      else:
        if hasattr(self, "n"):
          cfi.stopCriterion = orange.StopIMClusteringByAssessor_n()
        # the second term means "if it's clustersFromIM has columnAssessor attribute and it is (derived from) orange.ColumnAssessor_Kramer
        elif getattr(self, "binary", 0) or isinstance(getattr(cfi, "columnAssessor", None), orange.ColumnAssessor_Kramer):
          cfi.stopCriterion = orange.StopIMClusteringByAssessor_binary()
        else:
          cfi.stopCriterion = getattr(self, "stopCriterion", orange.StopIMClusteringByAssessor_noProfit())
          
    if hasattr(self, "minProfitProportion"):
      if not hasattr(cfi.stopCriterion, "minProfitProportion"):
        raise AttributeError, "invalid combination of stopping criteria (cannot set 'minProfitProportion')"
      cfi.stopCriterion.minProfitProportion = self.minProfitProportion
    elif hasattr(self, "n"):
      if not hasattr(cfi.stopCriterion, "n"):
        raise AttributeError, "invalid combination of stopping criteria (cannot set 'n')"
      cfi.stopCriterion.n = self.n

    if hasattr(self, "completion"):
      fim.completion = self.completion
        
    return fim

  
  def __call__(self, table, bound, weightID=0):
    if not self.instance:
      self.createInstance()
    return self.instance(table, bound, "", weightID)


FeatureByMinError = FeatureByIM

######################################################
# Kramer's algorithm (and similar distribution-based methods)


def FeatureByKramer(examples=None, bound=None, weightID=0, **argkw):
  return __learnConstructor(FeatureByKramerClass, examples, bound, weightID, argkw)


class FeatureByKramerClass:
  NoCompletion = orange.FeatureByIM.NoCompletion
  CompletionByDefault = orange.FeatureByIM.CompletionByDefault
  CompletionByBayes = orange.FeatureByIM.CompletionByBayes

  def __init__(self, **keyw):
    self.__dict__ = keyw
    self.instance = None

  def __setattr__(self, name, value):
    if name in ["clustersFromDistributions", "stopCriterion", "distributionAssessor", "measure", "m", "minProfitProportion"]:
        self.instance = None
    self.__dict__[name] = value

  def createInstance(self):    
    self.instance = fim = orange.FeatureByDistributions()
       
    cfd = fim.clustersFromDistributions = getattr(self, "clustersFromDistributions", orange.ClustersFromDistributionsByAssessor())

    if hasattr(self, "distributionAssessor"):
      cfd.distributionAssessor= self.distributionAssessor
    elif not hasattr(self, "classifierFromDistributions") and not hasattr(self, "clustersFromDistributions"):
      if (hasattr(self, "measure")):
        cfd.distributionAssessor = getattr(self, "distributionAssessor", orange.DistributionAssessor_Measure())
      elif (hasattr(self, "m")):
        cfd.distributionAssessor = getattr(self, "distributionAssessor", orange.DistributionAssessor_m())
      else:
        cfd.distributionAssessor = getattr(self, "distributionAssessor", orange.DistributionAssessor_Kramer())

    if hasattr(self, "measure"):
      if not hasattr(fim.clustersFromDistributions.distributionAssessor, "measure"):
        raise AttributeError, "invalid combination of distributionAssessor arguments (cannot set 'measure')"
      cfd.distributionAssessor.measure = self.measure
    elif hasattr(self, "m"):
      if not hasattr(fim.clustersFromDistributions.distributionAssessor, "m"):
        raise AttributeError, "invalid combination of distributionAssessor arguments (cannot set 'm')"
      cfd.distributionAssessor.m = self.m

    if hasattr(self, "stopCriterion"):
      cfd.stopCriterion=self.stopCriterion
    elif getattr(self, "binary", 0):
      if hasattr(self, "n") or hasattr(self, "minProfitProportion"):
        raise AttributeError, "invalid combination of stopping criteria"
      else:
        cfd.stopCriterion = orange.StopDistributionClustering_binary()
    elif hasattr(self, "n"):
      if hasattr(self, "minProfitProportion"):
        raise AttributeError, "invalid combination of stopping criteria"
      else:
        cfd.stopCriterion = orange.StopDistributionClustering_n()
    elif hasattr(self, "minProfitProportion") or not isinstance(getattr(cfd, "distributionAssessor", None), orange.DistributionAssessor_Kramer):
      cfd.stopCriterion = orange.StopDistributionClustering_noProfit()
    else:
      cfd.stopCriterion = orange.StopDistributionClustering_binary()
        
    if hasattr(self, "minProfitProportion"):
      if not hasattr(fim.clustersFromDistributions.stopCriterion, "minProfitProportion"):
        raise AttributeError, "invalid combination of stopping criteria (cannot set 'minProfitProportion')"
      cfd.stopCriterion.minProfitProportion = self.minProfitProportion
    elif hasattr(self, "n"):
      if not hasattr(fim.clustersFromDistributions.stopCriterion, "n"):
        raise AttributeError, "invalid combination of stopping criteria (cannot set 'n')"
      cfd.stopCriterion.n = self.n

    return fim


  def __call__(self, table, bound, weightID=0):
    if not self.instance:
            self.createInstance()
    return self.instance(table, bound, "", weightID)


######################################################
# Constructive induction by random merge


def FeatureByRandom(examples=None, bound=None, weightID=0, **argkw):
  return __learnConstructor(FeatureByRandomClass, examples, bound, weightID, argkw)


class FeatureByRandomClass:
  def __init__(self, **keyw):
    self.__dict__=keyw
    if not hasattr(self, "n"):
      self.n = 2
    self.instance = self

  def createInstance(self):
    return self
  
  def __call__(self, table, bound, weight=0):
    bound = [table.domain[a] for a in bound]
    newattr = orange.EnumVariable(reduce(lambda x,y:x+"-"+y, [a.name for a in bound]), values = ["r%i" % i for i in range(self.n)])
    if not len(bound):
      raise AttributeError, "no bound attributes"

    newattr.getValueFrom = orngLookup.lookupFromBound(newattr, [table.domain[x] for x in bound])
    lookupTable = newattr.getValueFrom.lookupTable = [random.randint(0, self.n-1) for i in newattr.getValueFrom.lookupTable]

    return newattr, random.randint(0, 100)


######################################################
# Constructive induction by random merge

def FeatureByCartesianProduct(examples=None, bound=None, weightID=0, **argkw):
  return __learnConstructor(FeatureByCartesianProductClass, examples, bound, weightID, argkw)

class FeatureByCartesianProductClass:
  def __init__(self, **keyw):
    self.__dict__= keyw
    if not hasattr(self, "measure"):
      self.measure = None
    self.instance = self

  def createInstance(self):
    return self

  def __call__(self, table, bound, weightID=0):
    if not len(bound):
      raise AttributeError, "no bound attributes"

    bound = [table.domain[a] for a in bound]
    newVar = orange.EnumVariable("-".join([a.name for a in bound]))

    if (len(bound)==1):
      newVar.values = list(bound[0].values)
      clsfr = orange.ClassifierByLookupTable(newVar, bound[0])
    else:
      import orngMisc
      for vs in orngMisc.LimitedCounter([len(a.values) for a in bound]):
        newVar.values.append("-".join([bound[i].values[v] for i, v in enumerate(vs)]))
      clsfr = orange.ClassifierByLookupTable(newVar, bound)
      
##    elif (len(bound)==2):
##      for v1 in bound[0].values:
##        for v2 in bound[1].values:
##          newVar.values.append(v1+"-"+v2)
##      clsfr = orange.ClassifierByLookupTable2(newVar, bound[0], bound[1])
##    elif (len(bound)==3):
##      for v1 in bound[0].values:
##        for v2 in bound[1].values:
##          for v3 in bound[2].values:
##            newVar.values.append(v1+"-"+v2+"-"+v3)
##      clsfr = orange.ClassifierByLookupTable3(newVar, bound[0], bound[1], bound[2])
##    else:
##      raise AttributeError, "cannot deal with more than 3 bound attributes"

    for i in range(len(newVar.values)):
      clsfr.lookupTable[i] = orange.Value(newVar, i)

    newVar.getValueFrom = clsfr

    if self.measure:
      meas = self.measure(newVar, table)
    else:
      meas = 0
    return newVar, meas

  def getLookupTableIndex(self, valArray):
    if len(valArray) == 1:
      return valArray[0]
    elif len(valArray) == 2:
      return clsfr.noOfValues1 * valArray[0] + valArray[1]
    elif len(valArray) == 3:
      return ((clsfr.noOfValues1 * valArray[0]) + valArray[1]) * clsfr.noOfValues2 + valArray[2]
    elif len(valArray) > 3:
      tmp = 0
      for i in range(len(clsfr.noOfValues)-1):
        tmp = (tmp + valArray[i]) * clsfr.noOfValues[i]
      return tmp + valArray[-1]

######################################################
# Feature construction for removal of redundant values

class AttributeRedundanciesRemover:
  def __init__(self, **keyw):
    self.__dict__ = keyw

  def __call__(self, data, weight):
    if hasattr(self, "inducer"):
      inducer = self.inducer
    else:
      if hasattr(self, "m"):
        inducer = FeatureByMinError()
      else:
        inducer = FeatureByMinComplexity()

    if hasattr(self, "m"):
      if not hasattr(inducer, m):
        raise TypeError, "invalid combination of arguments ('m' is given, but 'inducer' does not need it)"
      inducer.m = self.m

    import orngEvalAttr
    measure = getattr(self, "measure", orange.MeasureAttribute_relief(m=5, k=10))
    ordered = orngEvalAttr.OrderAttributesByMeasure(measure)(data, weight)

    for attr in ordered:
      newattr = inducer(data, [attr], weight)[0]
      if len(newattr.values) < len(attr.values):
        newset = filter(lambda x: x!=attr, data.domain.attributes)
        if len(newattr.values)>1:
          newset.append(newattr)
          newattr.name = attr.name + "'"
        data = data.select(newset + [data.domain.classVar])

    return data

######################################################
# Feature generators and structure inducers

def FeatureGenerator(examples = None, weightID = 0, **argkw):
  fm = apply(FeatureGeneratorClass, (), argkw)
  if examples:
      fm = fm(examples, weightID)
  return fm

class FeatureGeneratorClass:
  def __init__(self, **keyw):
    self.__dict__=keyw

  def __call__(self, data, weightID=0):
    if not hasattr(self, "featureInducer"):
      raise AttributeError, "'featureInducer' not set"
    
    ssgen = getattr(self, "subsetsGenerator")
    if not ssgen:
      ssgen = orange.SubsetsGenerator_constSize(2)
    if not ssgen.reset(data.domain.attributes):
      return []

    return [self.featureInducer(data, bound, weightID) for bound in ssgen]


def StructureInducer(examples=None, weightID=0, **argkw):
  fm = apply(StructureInducerClass, (), argkw)
  if examples:
      fm = fm(examples, weightID)
  return fm

  
class StructureInducerClass:
  def __init__(self, **keyw):
    self.__dict__=keyw

    for i in ["redundancyRemover", "alternativeMeasure", "learnerForUnknown", "subsetsGenerator"]:
      if not hasattr(self, i):
        setattr(self, i, None)

  def __call__(self, data, weight=0):
    import orngLookup
    
    if self.alternativeMeasure:
      raise SystemError, "alternativeMeasure not implemented yet"

    keepDuplicates = getattr(self, "keepDuplicates", 0)

    data = orange.ExampleTable(data)
    if not weight:
      # This is here for backward compatibility
      if hasattr(self, "weight"):
        weight = self.weight
      else:
        weight = orange.newmetaid()
        data.addMetaAttribute(weight)

    if self.redundancyRemover:
      data = self.redundancyRemover(data, weight)
    if not keepDuplicates:
      data.removeDuplicates(weight)

    induced = 0
    featureGenerator = FeatureGenerator(featureInducer=self.featureInducer, subsetsGenerator = self.subsetsGenerator)
    
    while(1):
      newFeatures = featureGenerator(data, weight)
      if not newFeatures or not len(newFeatures):
        break

      best = orngMisc.selectBest(newFeatures, orngMisc.compare2_lastBigger)[0]
      if len(best.getValueFrom.boundset()) == len(data.domain.attributes):
        break
      
      induced += 1
      best.name = "c%d" % induced
      
      data = replaceWithInduced(best, data)
      if not keepDuplicates:
        data.removeDuplicates(weight)

    if self.learnerForUnknown:
      learnerForUnknown = self.learnerForUnknown
    else:
      learnerForUnknown = orange.BayesLearner()

    return orngLookup.lookupFromExamples(data, weight, learnerForUnknown)


######################################################
# HINT: both, the original algorithms for inducing a structure

def HINT(examples=None, weightID=0, **argkw):
  fm = apply(HINTClass, (), argkw)
  if examples:
      fm = fm(examples, weightID)
  return fm

class HINTClass:
  def __init__(self, **keyw):
    self.__dict__=keyw

  def __call__(self, data, weight=0):
    import orngWrap
    
    type=getattr(self, "type", "auto")

    if hasattr(self, "boundsize"):
      if type(self)==int:
        subgen=orange.SubsetsGenerator_constSize(B = self.boundsize)
      else:
        subgen=orange.SubsetsGenerator_minMaxSize(min = self.boundsize[0], max = self.boundsize[1])
    else:
        subgen=orange.SubsetsGenerator_constSize(B = 2)
        

    if type=="auto":
      im=orange.IMBySorting(data, [])
      if im.fuzzy():
        type="error"
      else:
        type="complexity"

    inducer=StructureInducer(removeDuplicates = 1,
                             redundancyRemover = AttributeRedundanciesRemover(),
                             learnerForUnknown = orange.MajorityLearner()
                           )

    if type=="complexity":
      inducer.featureInducer = FeatureByMinComplexity()
      return inducer(data, weight)

    elif type=="error":
      ms=getattr(self, "m", orange.frange(0.1)+orange.frange(1.2, 3.0, 0.2)+orange.frange(4.0, 10.0, 1.0))
    
      inducer.redundancyRemover.inducer=inducer.featureInducer = FeatureByMinError()

      # it's the same object for redundancy remover and the real inducer, so we can tune just one
      return orngWrap.Tune1Parameter(
          parameter = "featureInducer.m",
          values = ms,
          object = inducer,
          returnWhat = orngWrap.Tune1Parameter.returnClassifier
      )(data, weight)
      
      print inducer.featureInducer.m, inducer.redundancyRemover.inducer.m
      return inducer(data, weight)



def replaceWithInduced(attr, table):
  return table.select(filter(lambda x, b=list(attr.getValueFrom.boundset()): not b.count(x), table.domain.attributes)+[attr, table.domain.classVar])


def addAnAttribute(attr, table):
  return table.select(table.domain.attributes+[attr, table.domain.classVar])


###########################

def printHierarchy(cblt):
    if isinstance(cblt, orange.Variable):
      printHierarchy1(0, cblt)
    else:
      # cblt does not necessarily equal cblt.classVar.getValueFrom,
      # hence a special case
      me = cblt.classVar
      print "%s/%i %s" % (me.name, len(me.values), me.values)
      try:
        bound = cblt.boundset()
        for i in bound:
          printHierarchy1(1, i)
      except:
        pass

def printHierarchy1(dep, me):
    print '  '*dep*3 + ("%s/%i %s" % (me.name, len(me.values), me.values))
    try:
        bound=me.getValueFrom.boundset()
    except:
        return
    for i in bound:
        printHierarchy1(dep+1, i)


def dotHierarchy(file, cblt):
    fopened=0
    if type(file)==str:
        file=open(file, "wt")
        fopened=1

    file.write('digraph G {\n')

    if isinstance(cblt, orange.Variable):
      dotHierarchy1(file, "a", cblt)
    else:
      # cblt does not necessarily equal cblt.classVar.getValueFrom,
      # hence a special case
      myname = "a"
      me = cblt.classVar
      bound = cblt.boundset()
      file.write('  '*len(myname) + ' %s [label="%s/%d", shape=plaintext]\n' % (myname, me.name, len(me.values)))
      for i in range(len(bound)):
          subname="%s%i" % (myname, i)
          file.write('   %s -> %s \n' % (myname, subname))
          dotHierarchy1(file, subname, bound[i])

    file.write("}\n")
    
    if fopened:
        file.close()

def dotHierarchy1(file, myname, me):
    file.write('  '*len(myname) + ' %s [label="%s/%d", shape=plaintext]\n' % (myname, me.name, len(me.values)))
    try:
        bound=me.getValueFrom.boundset()
    except:
        return
    for i in range(len(bound)):
        subname="%s%i" % (myname, i)
        file.write('  '*len(myname) + (' %s -> %s \n' % (myname, subname)))
        dotHierarchy1(file, subname, bound[i])
