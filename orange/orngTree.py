import orange, types, string

# Z-multiplier table for confidence intervals
Z = {
  0.75:1.15,
  0.80:1.28,
  0.85:1.44,
  0.90:1.64,
  0.95:1.96,
  0.99:2.58
  }

def TreeLearner(examples = None, weightID = 0, **argkw):
  tree = apply(TreeLearnerClass, (), argkw)
  if examples:
    tree = tree(examples, weightID)
  return tree

class TreeLearnerClass:
  def __init__(self, **kw):
    self.learner = None
    self.__dict__.update(kw)
    
  def __setattr__(self, name, value):
    if name in ["split", "binarization", "measure", "worstAcceptable", "minSubset",
          "stop", "maxMajority", "minExamples", "nodeLearner", "maxDepth"]:
      self.learner = None
    self.__dict__[name] = value

  def __call__(self, examples, weight=0):
    if not self.learner:
      self.learner = self.instance()
    if not hasattr(self, "split") and not hasattr(self, "measure"):
      if examples.domain.classVar.varType == orange.VarTypes.Discrete:
        measure = orange.MeasureAttribute_gainRatio()
      else:
        measure = orange.MeasureAttribute_MSE()
      self.learner.split.continuousSplitConstructor.measure = measure
      self.learner.split.discreteSplitConstructor.measure = measure
        
    tree = self.learner(examples, weight)
    if getattr(self, "sameMajorityPruning", 0):
      tree = orange.TreePruner_SameMajority(tree)
    if getattr(self, "mForPruning", 0):
      tree = orange.TreePruner_m(tree, m = self.mForPruning)
    return tree


  def instance(self):
    learner = orange.TreeLearner()

    if hasattr(self, "split"):
      learner.split = self.split
    else:
      learner.split = orange.TreeSplitConstructor_Combined()
      learner.split.continuousSplitConstructor = orange.TreeSplitConstructor_Threshold()
      if getattr(self, "binarization", 0):
        learner.split.discreteSplitConstructor = orange.TreeSplitConstructor_ExhaustiveBinary()
      else:
        learner.split.discreteSplitConstructor = orange.TreeSplitConstructor_Attribute()

      measures = {"infoGain": orange.MeasureAttribute_info,
          "gainRatio": orange.MeasureAttribute_gainRatio,
          "gini": orange.MeasureAttribute_gini,
          "relief": orange.MeasureAttribute_relief,
          "retis": orange.MeasureAttribute_MSE
          }

      measure = getattr(self, "measure", None)
      if type(measure) == str:
        measure = measures[measure]()

      learner.split.continuousSplitConstructor.measure = measure
      learner.split.discreteSplitConstructor.measure = measure

      wa = getattr(self, "worstAcceptable", 0)
      if wa:
        learner.split.continuousSplitConstructor.worstAcceptable = wa
        learner.split.discreteSplitConstructor.worstAcceptable = wa

      ms = getattr(self, "minSubset", 0)
      if ms:
        learner.split.continuousSplitConstructor.minSubset = ms
        learner.split.discreteSplitConstructor.minSubset = ms

    if hasattr(self, "stop"):
      learner.stop = self.stop
    else:
      learner.stop = orange.TreeStopCriteria_common()
      mm = getattr(self, "maxMajority", 0)
      if mm:
        learner.stop.maxMajority = self.maxMajority
      me = getattr(self, "minExamples", 0)
      if me:
        learner.stop.minExamples = self.minExamples

    for a in ["storeDistributions", "storeContingencies", "storeExamples", "storeNodeClassifier", "nodeLearner", "maxDepth"]:
      if hasattr(self, a):
        setattr(learner, a, getattr(self, a))

    return learner


def __countNodes(node):
  if node:
    count = 1
    if node.branches: # internal node
      for node in node.branches:
        count = count + __countNodes(node)
    return count
  else: # null-node
    return 0

def countNodes(tree):
  if type(tree) == orange.TreeClassifier:
    tree = tree.tree
  if tree:
    return __countNodes(tree)

def __countLeaves(node):
  if node:
    if node.branches: # internal node
      count = 0
      for node in node.branches:
        count = count + __countLeaves(node)
      return count
    else:
      return 1
  else:
    return 0

def countLeaves(tree):
  if type(tree) == orange.TreeClassifier:
    tree = tree.tree
  if tree:
    return __countLeaves(tree)

def __printNode(outputFormat, node, lev, name="0", continuous=0, cont="", major="", examples = 0, el=-1, depth=10000, bi=-1,\
        bsn="", bstr="", lshp="plaintext",inshp="box", iNF=[], lF=[], dP=3, z=0.95):
  s=""
  if node:
    if outputFormat=='DOT': lev+=1
    if node.branches:
      if iNF==[]:
        iNFString=""
      else:
        if continuous:
          e = string.find(cont,"[")
          if "average" in iNF:
            if "confidenceInterval" in iNF:
              iNFString = " ("+cont+")"
            else:
              iNFString = " ("+cont[:e-1]+")"
          else:
            if "confidenceInterval" in iNF:
              iNFString = " ("+cont[e:]+")"
          if "examples" in iNF:
            iNFString += " (%i)" % examples
        else:
          if outputFormat=='TXT':
            iNFString = " ("+reduce(lambda x,y: x+"; "+y, [i[1] for i in [("major",major),("distribution",cont),("baseValue",bstr)] if i[0] in iNF])+")"
          else:
            iNFString = ""
            if "major" in iNF:
              iNFString+=major
            if "distribution" in iNF:
              iNFString+=cont
            if "baseValue" in iNF:
              iNFString+=bstr
      if outputFormat=='DOT':
        s = s + "\tn%s [ shape=%s, label = \"%s\\n%s\"]\n" % (str(name), inshp, node.branchSelector.classVar.name, iNFString)
      for i in range(len(node.branches)):
        if node.branches[i] and lev<depth and node.branches[i].distribution.abs >= el:
          new_name = "%s_%s" % (name, str(i))
          if outputFormat=='DOT':
            s = s + "\tn%s -> n%s [ label = \"%s\"] \n" % (str(name), str(new_name), node.branchDescriptions[i])
          majorString = ""
          if continuous:
            avg = node.branches[i].distribution.average()
            try:
              err = node.branches[i].distribution.error()
            except:
              err=1
            formatstring = "%"+"%d.%df" % (dP+2, dP)
            formatstring = formatstring+" ["+formatstring+", "+formatstring+"]"
            cont = formatstring % (avg, avg - Z[z]*err, avg + Z[z]*err)
          else:
            if bi!=-1 and bsn!="":
              bstr = "%s: %s%s" % (bsn, str(node.branches[i].distribution[bi]*100 / node.branches[i].distribution.abs)[0:5], '%')
              if outputFormat=='DOT': bstr+="\\n"
            if cont!="":
              cont = "%s" % __formatDistribution(node.branches[i].distribution)
              if outputFormat=='DOT': cont+="\\n"
            if major!="":
              l = __maxx(node.branches[i].distribution)
              majorString = "%s%s" % (str(l[0][0]*100 / node.branches[i].distribution.abs)[0:5], '%')
              if outputFormat=='DOT': majorString+="\\n"
          if outputFormat=='TXT':
            s+= "\n"+"|   "*lev + "%s%s %s: " % (node.branchSelector.classVar.name, iNFString, node.branchDescriptions[i])
            s+= __printNode(outputFormat,node.branches[i], lev+1, new_name,continuous, cont, majorString, node.branches[i].distribution.abs, el,\
                  depth, bi, bsn, bstr,lshp,inshp,iNF,lF,dP,z)
          else:
            s+= __printNode(outputFormat,node.branches[i], lev, new_name,continuous, cont, majorString, node.branches[i].distribution.abs, el,\
                  depth, bi, bsn, bstr,lshp,inshp,iNF,lF,dP,z)
    else: # print a leaf
      if outputFormat=='TXT' and lev>=depth:
        return ""
      lFString=""
      if continuous:
        e = string.find(cont,"[")
        if "average" in lF:
          if "confidenceInterval" in lF:
            lFString = cont
          else:
            lFString = cont[:e-1]
        else:
          if "confidenceInterval" in lF:
            lFString = cont[e:]
        if "examples" in lF:
          lFString += " (%i)" % examples
        if outputFormat=='DOT':
          s+="\tn%s [shape=%s, label = \"%s\"]\n" % (str(name), lshp, lFString)
        else:
          s += lFString
      else:
        if outputFormat=='TXT':
          lFString = " ("+reduce(lambda x,y: x+"; "+y, [i[1] for i in [("major",major),("distribution",cont),("baseValue",bstr),("examples",examples)] if i[0] in lF])+")"
          if lFString == " ()":
            lFString=""
          s+= "%s%s" % (node.nodeClassifier.defaultValue, lFString)
        else:
          if "major" in lF:
            lFString+=major
          if "distribution" in lF:
            lFString+=cont
          if "baseValue" in lF:
            lFString+=bstr
          s+="\tn%s [shape=%s, label = \"%s\\n%s\"]\n" % (str(name), lshp, node.nodeClassifier.defaultValue, lFString)
  else:
    s+= "null node\n"
  return s

def printTxt(tree, fileName="", examplesLimit=-1, depthLimit=10000, baseValueIndex=-1, \
        internalNodeFields=[], leafFields=["major","average"], decimalPlaces=3, confidenceLevel=0.95):

  out = None
  if type(tree) == orange.TreeClassifier:
    tree = tree.tree
  if confidenceLevel not in Z.keys():
    confidenceLevel=0.95
  if not tree.distribution:
    raise "Class distributions haven't been not stored in the tree"
  if fileName!="":
    out = open(fileName, 'w')
  if depthLimit < 1 or tree.distribution and tree.distribution.abs < examplesLimit:
    if out:
      out.close()
    return

  if type(leafFields) == types.StringType:
    leafFields=[leafFields]
  if type(internalNodeFields) == types.StringType:
    internalNodeFields=[internalNodeFields]

  baseValueString = ""
  baseName = ""
  distributionString =""
  majorString = ""
  continuous = 0

  if tree.distribution.supportsContinuous:
    # Continuous class
    continuous = 1
    avg = tree.distribution.average()
    err = tree.distribution.error()
    formatstring = "%"+"%d.%df" % (decimalPlaces+2, decimalPlaces)
    formatstring = formatstring+" ["+formatstring+", "+formatstring+"]"
    distributionString = formatstring % (avg, avg-Z[confidenceLevel]*err, avg+Z[confidenceLevel]*err)
  else:
    # Discrete class
    if baseValueIndex != -1:
      baseName = tree.distribution.variable.values[baseValueIndex]
      baseValueString = "%s: %s%s" % (baseName,str(tree.distribution[baseValueIndex]*100/tree.distribution.abs)[0:5],'%')

    distributionString = "%s" % __formatDistribution(tree.distribution)
    l = __maxx(tree.distribution)
    majorString = "%s%s" % (str(l[0][0]*100 / tree.distribution.abs)[0:5], '%')

  s = __printNode('TXT',tree, 0, continuous=continuous, cont=distributionString, major=majorString, examples=tree.distribution.abs, \
          el=examplesLimit, depth=depthLimit, bi=baseValueIndex, bsn=baseName, bstr=baseValueString, \
          lshp="", inshp="", iNF=internalNodeFields, lF=leafFields, dP=decimalPlaces,z=confidenceLevel)
  if out:
    out.write(s)
    out.close()
  else:
    print s

# tree ... the tree to be printed out
# fileName ... the name of the output file in dot format
# examplesLimit ... recursively write out the tree while there's more (or equal) than examplesLimit examples in the node
# depthLimit ... recursively write out the tree until depthLimit is reached
# distribution ... 1=output class distribution for each node; 0=don't
# pctOfMajor ... 1=output the percentage of majority class for each node; 0=no pct. of majority class output
# baseValueIndex ... index of the base class; no baseValue pct. output if -1
# leafShape ... a shape of the leaf node in dot format
# internalNodeShape ... a shape of the internal node in dot format
def printDot(tree, fileName="out.dot", examplesLimit=-1, depthLimit=10000, baseValueIndex=-1, \
      leafShape="plaintext", internalNodeShape="box", internalNodeFields=[], \
      leafFields=["major","average"], decimalPlaces=3, confidenceLevel=0.95):

  if type(tree) == orange.TreeClassifier:
    tree = tree.tree
  if confidenceLevel not in Z.keys():
    confidenceLevel=0.95
  if not tree.distribution:
    raise "Class distributions haven't been not stored in the tree"
  out = open(fileName, 'w')
  if depthLimit < 1 or tree.distribution and tree.distribution.abs < examplesLimit:
    out.close()
    return
  out.write("digraph G {\n")

  if type(leafFields) == types.StringType:
    leafFields=[leafFields]
  if type(internalNodeFields) == types.StringType:
    internalNodeFields=[internalNodeFields]

  baseValueString = ""
  baseName = ""
  distributionString =""
  majorString = ""
  continuous = 0

  if tree.distribution.supportsContinuous:
    # Continuous class
    continuous = 1
    avg = tree.distribution.average()
    err = tree.distribution.error()
    formatstring = "%"+"%d.%df" % (decimalPlaces+2, decimalPlaces)
    formatstring = formatstring+" ["+formatstring+", "+formatstring+"]"
    distributionString = formatstring % (avg, avg - Z[confidenceLevel]*err, avg + Z[confidenceLevel]*err)
  else:
    # Discrete class
    if baseValueIndex != -1:
      baseName = tree.distribution.variable.values[baseValueIndex]
      baseValueString = "%s: %s%s\\n" % (baseName,str(tree.distribution[baseValueIndex]*100/tree.distribution.abs)[0:5],'%')

    distributionString = "%s\\n" % __formatDistribution(tree.distribution)
    l = __maxx(tree.distribution)
    majorString = "%s%s\\n" % (str(l[0][0]*100 / tree.distribution.abs)[0:5], '%')

  s = __printNode('DOT',tree, 0, continuous=continuous, cont=distributionString, major=majorString, \
          el=examplesLimit, depth=depthLimit, bi=baseValueIndex, bsn=baseName, bstr=baseValueString, \
          lshp=leafShape, inshp=internalNodeShape, iNF=internalNodeFields, lF=leafFields, dP=decimalPlaces,z=confidenceLevel)
  out.write(s+"}\n")
  out.close()

def __formatDistribution(c):
  if type(c)==orange.ContDistribution:
    return str(c)  
  l=[int(i) for i in c if int(i)==float(i)]
  if len(l)!=len(c):
    l=[]
    for i in c:
      if int(i)==float(i):
        l.append(str(int(i)))
      else:
        tmp=string.split(str(i), '.')
        temp=tmp[1]
        if len(temp)>3:
          temp=tmp[1][:3]
        l.append(tmp[0]+'.'+temp)
  return "<"+str(l)[1:-1]+">"

def __maxx(cont):
  if type(cont)==orange.ContDistribution:
    return 0
  m = -1
  l = []
  for i in range(len(cont)):
    if cont[i]>=m:
      if cont[i]>m:
        l=[]
      l.append((cont[i],i))
      m = cont[i]
  return l
