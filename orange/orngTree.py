import orange
from warnings import warn

class TreeLearner(orange.Learner):
    """TreeLearner(**kwargs)
    
    Keyword arguments:
    split -- Object of type TreeSplitConstructor. Default value, provided by TreeLearner, is SplitConstructor_Combined 
        with separate constructors for discrete and continuous attributes. Discrete attributes are used as are, while 
        continuous attributes are binarized. Gain ratio is used to select attributes. 
        A minimum of two examples in a leaf is required for discrete and five examples in a leaf for continuous attributes.
    binarization -- can be one of:
        1: orange.TreeSplitConstructor_ExhaustiveBinary()
        2: orange.TreeSplitConstructor_OneAgainstOthers()
        else: learner.split.discreteSplitConstructor = orange.TreeSplitConstructor_Attribute()
    measure -- can be either a measure or a string, in which case one of the following is used:
        "infoGain": orange.MeasureAttribute_info,
        "gainRatio": orange.MeasureAttribute_gainRatio,
        "gini": orange.MeasureAttribute_gini,
        "relief": orange.MeasureAttribute_relief,
        "retis": orange.MeasureAttribute_MSE
    worstAcceptable -- 
    minSubset --
    stop -- Object of type TreeStopCriteria. The default stopping criterion stops induction when all examples in a node belong to the same class.
    maxMajority --
    minExamples --
    nodeLearner --
    maxDepth -- Gives maximal tree depth; 
        0 means that only root is generated. The default is 100 to prevent any infinite tree induction due to missettings in stop criteria. 
        If you are sure you need larger trees, increase it. If you, on the other hand, want to lower this hard limit, you can do so as well.
    sameMajorityPruning --
    mForPruning -- 
    reliefM -- 
    reliefK --
    storeDistributions --
    storeContingencies --
    storeExamples --
    storeNodeClassifier --
    nodeLearner --
    """
    def __new__(cls, examples = None, weightID = 0, **argkw):
        self = orange.Learner.__new__(cls, **argkw)
        if examples:
            self.__init__(**argkw)
            return self.__call__(examples, weightID)
        else:
            return self
      
    def __init__(self, **kw):
        self.learner = None
        self.__dict__.update(kw)
      
    def __setattr__(self, name, value):
        if name in ["split", "binarization", "measure", "worstAcceptable", "minSubset",
              "stop", "maxMajority", "minExamples", "nodeLearner", "maxDepth", "reliefM", "reliefK"]:
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

        hasSplit = hasattr(self, "split")
        if hasSplit:
            learner.split = self.split
        else:
            learner.split = orange.TreeSplitConstructor_Combined()
            learner.split.continuousSplitConstructor = orange.TreeSplitConstructor_Threshold()
            binarization = getattr(self, "binarization", 0)
            if binarization == 1:
                learner.split.discreteSplitConstructor = orange.TreeSplitConstructor_ExhaustiveBinary()
            elif binarization == 2:
                learner.split.discreteSplitConstructor = orange.TreeSplitConstructor_OneAgainstOthers()
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
            if not hasSplit and not measure:
                measure = orange.MeasureAttribute_gainRatio()

            measureIsRelief = type(measure) == orange.MeasureAttribute_relief
            relM = getattr(self, "reliefM", None)
            if relM and measureIsRelief:
                measure.m = relM
            
            relK = getattr(self, "reliefK", None)
            if relK and measureIsRelief:
                measure.k = relK

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
            mm = getattr(self, "maxMajority", 1.0)
            if mm < 1.0:
                learner.stop.maxMajority = self.maxMajority
            me = getattr(self, "minExamples", 0)
            if me:
                learner.stop.minExamples = self.minExamples

        for a in ["storeDistributions", "storeContingencies", "storeExamples", "storeNodeClassifier", "nodeLearner", "maxDepth"]:
            if hasattr(self, a):
                setattr(learner, a, getattr(self, a))

        return learner


def __countNodes(node):
    count = 0
    if node:
        count += 1
        if node.branches:
            for node in node.branches:
                count += __countNodes(node)
    return count

def countNodes(tree):
    return __countNodes(type(tree) == orange.TreeClassifier and tree.tree or tree)


def __countLeaves(node):
    count = 0
    if node:
        if node.branches: # internal node
            for node in node.branches:
                count += __countLeaves(node)
        else:
            count += 1
    return count

def countLeaves(tree):
    return __countLeaves(type(tree) == orange.TreeClassifier and tree.tree or tree)



import re
fs = r"(?P<m100>\^?)(?P<fs>(\d*\.?\d*)?)"
by = r"(?P<by>(b(P|A)))?"
bysub = r"((?P<bysub>b|s)(?P<by>P|A))?"
opc = r"(?P<op>=|<|>|(<=)|(>=)|(!=))(?P<num>\d*\.?\d+)"
opd = r'(?P<op>=|(!=))"(?P<cls>[^"]*)"'
intrvl = r'((\((?P<intp>\d+)%?\))|(\(0?\.(?P<intv>\d+)\))|)'
fromto = r"(?P<out>!?)(?P<lowin>\(|\[)(?P<lower>\d*\.?\d+)\s*,\s*(?P<upper>\d*\.?\d+)(?P<upin>\]|\))"
re_V = re.compile("%V")
re_N = re.compile("%"+fs+"N"+by)
re_M = re.compile("%"+fs+"M"+by)
re_m = re.compile("%"+fs+"m"+by)
re_Ccont = re.compile("%"+fs+"C"+by+opc)
re_Cdisc = re.compile("%"+fs+"C"+by+opd)
re_ccont = re.compile("%"+fs+"c"+by+opc)
re_cdisc = re.compile("%"+fs+"c"+by+opd)
re_Cconti = re.compile("%"+fs+"C"+by+fromto)
re_cconti = re.compile("%"+fs+"c"+by+fromto)
re_D = re.compile("%"+fs+"D"+by)
re_d = re.compile("%"+fs+"d"+by)
re_AE = re.compile("%"+fs+"(?P<AorE>A|E)"+bysub)
re_I = re.compile("%"+fs+"I"+intrvl)

def insertDot(s, mo):
    return s[:mo.start()] + "." + s[mo.end():]

def insertStr(s, mo, sub):
    return s[:mo.start()] + sub + s[mo.end():]

def insertNum(s, mo, n):
    grps = mo.groupdict()
    m100 = grps.get("m100", None)
    if m100:
        n *= 100
    fs = grps.get("fs") or (m100 and ".0" or "5.3")
    return s[:mo.start()] + ("%%%sf" % fs % n) + s[mo.end():]

def byWhom(by, parent, tree):
        if by=="bP":
            return parent
        else:
            return tree.tree

def replaceV(strg, mo, node, parent, tree):
    return insertStr(strg, mo, str(node.nodeClassifier.defaultValue))

def replaceN(strg, mo, node, parent, tree):
    by = mo.group("by")
    N = node.distribution.abs
    if by:
        whom = byWhom(by, parent, tree)
        if whom and whom.distribution:
            if whom.distribution.abs > 1e-30:
                N /= whom.distribution.abs
        else:
            return insertDot(strg, mo)
    return insertNum(strg, mo, N)
        

def replaceM(strg, mo, node, parent, tree):
    by = mo.group("by")
    maj = int(node.nodeClassifier.defaultValue)
    N = node.distribution[maj]
    if by:
        whom = byWhom(by, parent, tree)
        if whom and whom.distribution:
            if whom.distribution[maj] > 1e-30:
                N /= whom.distribution[maj]
        else:
            return insertDot(strg, mo)
    return insertNum(strg, mo, N)
        

def replacem(strg, mo, node, parent, tree):
    by = mo.group("by")
    maj = int(node.nodeClassifier.defaultValue)
    if node.distribution.abs > 1e-30:
        N = node.distribution[maj] / node.distribution.abs
        if by:
            if whom and whom.distribution:
                byN = whom.distribution[maj] / whom.distribution.abs
                if byN > 1e-30:
                    N /= byN
            else:
                return insertDot(strg, mo)
    else:
        N = 0.
    return insertNum(strg, mo, N)


def replaceCdisc(strg, mo, node, parent, tree):
    if tree.classVar.varType != orange.VarTypes.Discrete:
        return insertDot(strg, mo)
    
    by, op, cls = mo.group("by", "op", "cls")
    N = node.distribution[cls]
    if op == "!=":
        N = node.distribution.abs - N
    if by:
        whom = byWhom(by, parent, tree)
        if whom and whom.distribution:
            if whom.distribution[cls] > 1e-30:
                N /= whom.distribution[cls]
        else:
            return insertDot(strg, mo)
    return insertNum(strg, mo, N)

    
def replacecdisc(strg, mo, node, parent, tree):
    if tree.classVar.varType != orange.VarTypes.Discrete:
        return insertDot(strg, mo)
    
    op, by, cls = mo.group("op", "by", "cls")
    N = node.distribution[cls]
    if node.distribution.abs > 1e-30:
        N /= node.distribution.abs
        if op == "!=":
            N = 1 - N
    if by:
        whom = byWhom(by, parent, tree)
        if whom and whom.distribution:
            if whom.distribution[cls] > 1e-30:
                N /= whom.distribution[cls] / whom.distribution.abs
        else:
            return insertDot(strg, mo)
    return insertNum(strg, mo, N)


import operator
__opdict = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "=": operator.eq, "!=": operator.ne}

def replaceCcont(strg, mo, node, parent, tree):
    if tree.classVar.varType != orange.VarTypes.Continuous:
        return insertDot(strg, mo)
    
    by, op, num = mo.group("by", "op", "num")
    op = __opdict[op]
    num = float(num)
    N = sum([x[1] for x in node.distribution.items() if op(x[0], num)], 0.)
    if by:
        whom = byWhom(by, parent, tree)
        if whom and whom.distribution:
            byN = sum([x[1] for x in whom.distribution.items() if op(x[0], num)], 0.)
            if byN > 1e-30:
                N /= byN
        else:
            return insertDot(strg, mo)

    return insertNum(strg, mo, N)
    
    
def replaceccont(strg, mo, node, parent, tree):
    if tree.classVar.varType != orange.VarTypes.Continuous:
        return insertDot(strg, mo)
    
    by, op, num = mo.group("by", "op", "num")
    op = __opdict[op]
    num = float(num)
    N = sum([x[1] for x in node.distribution.items() if op(x[0], num)], 0.)
    if node.distribution.abs > 1e-30:
        N /= node.distribution.abs
    if by:
        whom = byWhom(by, parent, tree)
        if whom and whom.distribution:
            byN = sum([x[1] for x in whom.distribution.items() if op(x[0], num)], 0.)
            if byN > 1e-30:
                N /= byN/whom.distribution.abs # abs > byN, so byN>1e-30 => abs>1e-30
        else:
            return insertDot(strg, mo)
    return insertNum(strg, mo, N)


def extractInterval(mo, dist):
    out, lowin, lower, upper, upin = mo.group("out", "lowin", "lower", "upper", "upin")
    lower, upper = float(lower), float(upper)
    if out:
        lop = lowin == "(" and operator.le or operator.lt
        hop = upin == ")" and operator.ge or operator.ge
        return filter(lambda x:lop(x[0], lower) or hop(x[0], upper), dist.items())
    else:
        lop = lowin == "(" and operator.gt or operator.ge
        hop = upin == ")" and operator.lt or operator.le
        return filter(lambda x:lop(x[0], lower) and hop(x[0], upper), dist.items())

    
def replaceCconti(strg, mo, node, parent, tree):
    if tree.classVar.varType != orange.VarTypes.Continuous:
        return insertDot(strg, mo)

    by = mo.group("by")
    N = sum([x[1] for x in extractInterval(mo, node.distribution)])
    if by:
        whom = byWhom(by, parent, tree)
        if whom and whom.distribution:
            byN = sum([x[1] for x in extractInterval(mo, whom.distribution)])
            if byN > 1e-30:
                N /= byN
        else:
            return insertDot(strg, mo)
        
    return insertNum(strg, mo, N)

            
def replacecconti(strg, mo, node, parent, tree):
    if tree.classVar.varType != orange.VarTypes.Continuous:
        return insertDot(strg, mo)

    N = sum([x[1] for x in extractInterval(mo, node.distribution)])
    ab = node.distribution.abs
    if ab > 1e-30:
        N /= ab

    by = mo.group("by")
    if by:
        whom = byWhom(by, parent, tree)
        if whom and whom.distribution:
            byN = sum([x[1] for x in extractInterval(mo, whom.distribution)])
            if byN > 1e-30:
                N /= byN/whom.distribution.abs
        else:
            return insertDot(strg, mo)
        
    return insertNum(strg, mo, N)

    
def replaceD(strg, mo, node, parent, tree):
    if tree.classVar.varType != orange.VarTypes.Discrete:
        return insertDot(strg, mo)

    fs, by, m100 = mo.group("fs", "by", "m100")
    dist = list(node.distribution)
    if by:
        whom = byWhom(by, parent, tree)
        if whom:
            for i, d in enumerate(whom.distribution):
                if d > 1e-30:
                    dist[i] /= d
        else:
            return insertDot(strg, mo)
    mul = m100 and 100 or 1
    fs = fs or (m100 and ".0" or "5.3")
    return insertStr(strg, mo, "["+", ".join(["%%%sf" % fs % (N*mul) for N in dist])+"]")


def replaced(strg, mo, node, parent, tree):
    if tree.classVar.varType != orange.VarTypes.Discrete:
        return insertDot(strg, mo)

    fs, by, m100 = mo.group("fs", "by", "m100")
    dist = list(node.distribution)
    ab = node.distribution.abs
    if ab > 1e-30:
        dist = [d/ab for d in dist]
    if by:
        whom = byWhom(by, parent, tree)
        if whom:
            for i, d in enumerate(whom.distribution):
                if d > 1e-30:
                    dist[i] /= d/whom.distribution.abs # abs > d => d>1e-30 => abs>1e-30
        else:
            return insertDot(strg, mo)
    mul = m100 and 100 or 1
    fs = fs or (m100 and ".0" or "5.3")
    return insertStr(strg, mo, "["+", ".join(["%%%sf" % fs % (N*mul) for N in dist])+"]")


def replaceAE(strg, mo, node, parent, tree):
    if tree.classVar.varType != orange.VarTypes.Continuous:
        return insertDot(strg, mo)

    AorE, bysub, by = mo.group("AorE", "bysub", "by")
    
    if AorE == "A":
        A = node.distribution.average()
    else:
        A = node.distribution.error()
    if by:
        whom = byWhom("b"+by, parent, tree)
        if whom:
            if AorE == "A":
                avg = whom.distribution.average()
            else:
                avg = whom.distribution.error()
            if bysub == "b":
                if avg > 1e-30:
                    A /= avg
            else:
                A -= avg
        else:
            return insertDot(strg, mo)
    return insertNum(strg, mo, A)


Z = { 0.75:1.15, 0.80:1.28, 0.85:1.44, 0.90:1.64, 0.95:1.96, 0.99:2.58 }

def replaceI(strg, mo, node, parent, tree):
    if tree.classVar.varType != orange.VarTypes.Continuous:
        return insertDot(strg, mo)

    fs = mo.group("fs") or "5.3"
    intrvl = float(mo.group("intp") or mo.group("intv") or "95")/100.
    mul = mo.group("m100") and 100 or 1

    if not Z.has_key(intrvl):
        raise SystemError, "Cannot compute %5.3f% confidence intervals" % intrvl

    av = node.distribution.average()    
    il = node.distribution.error() * Z[intrvl]
    return insertStr(strg, mo, "[%%%sf-%%%sf]" % (fs, fs) % ((av-il)*mul, (av+il)*mul))


# This class is more a collection of function, merged into a class so that they don't
# need to transfer too many arguments. It will be constructed, used and discarded,
# it is not meant to store any information.
class __TreeDumper:
    defaultStringFormats = [(re_V, replaceV), (re_N, replaceN), (re_M, replaceM), (re_m, replacem),
                              (re_Cdisc, replaceCdisc), (re_cdisc, replacecdisc),
                              (re_Ccont, replaceCcont), (re_ccont, replaceccont),
                              (re_Cconti, replaceCconti), (re_cconti, replacecconti),
                              (re_D, replaceD), (re_d, replaced),
                              (re_AE, replaceAE), (re_I, replaceI)
                             ]

    def __init__(self, leafStr, nodeStr, stringFormats, minExamples, maxDepth, simpleFirst, tree, **kw):
        self.stringFormats = stringFormats
        self.minExamples = minExamples
        self.maxDepth = maxDepth
        self.simpleFirst = simpleFirst
        self.tree = tree
        self.__dict__.update(kw)

        if leafStr:
            self.leafStr = leafStr
        else:
            if tree.classVar.varType == orange.VarTypes.Discrete:
                self.leafStr = "%V (%^.2m%)"
            else:
                self.leafStr = "%V"

        if nodeStr == ".":
            self.nodeStr = self.leafStr
        else:
            self.nodeStr = nodeStr
        

    def formatString(self, strg, node, parent):
        if hasattr(strg, "__call__"):
            return strg(node, parent, self.tree)
        
        if not node:
            return "<null node>"
        
        for rgx, replacer in self.stringFormats:
            if not node.distribution:
                strg = rgx.sub(".", strg)
            else:
                strt = 0
                while True:
                    mo = rgx.search(strg, strt)
                    if not mo:
                        break
                    strg = replacer(strg, mo, node, parent, self.tree)
                    strt = mo.start()+1
                        
        return strg
        

    def showBranch(self, node, parent, lev, i):
        bdes = node.branchDescriptions[i]
        bdes = node.branchSelector.classVar.name + (bdes[0] not in "<=>" and "=" or "") + bdes
        if node.branches[i]:
            nodedes = self.nodeStr and ": "+self.formatString(self.nodeStr, node.branches[i], node) or ""
        else:
            nodedes = "<null node>"
        return "|    "*lev + bdes + nodedes
        
        
    def dumpTree0(self, node, parent, lev):
        if node.branches:
            if node.distribution.abs < self.minExamples or lev > self.maxDepth:
                return "|    "*lev + ". . .\n"
            
            res = ""
            if self.leafStr and self.nodeStr and self.leafStr != self.nodeStr:
                leafsep = "\n"+("|    "*lev)+"    "
            else:
                leafsep = ""
            if self.simpleFirst:
                for i, branch in enumerate(node.branches):
                    if not branch or not branch.branches:
                        if self.leafStr == self.nodeStr:
                            res += "%s\n" % self.showBranch(node, parent, lev, i)
                        else:
                            res += "%s: %s\n" % (self.showBranch(node, parent, lev, i),
                                                 leafsep + self.formatString(self.leafStr, branch, node))
            for i, branch in enumerate(node.branches):
                if branch and branch.branches:
                    res += "%s\n%s" % (self.showBranch(node, parent, lev, i),
                                       self.dumpTree0(branch, node, lev+1))
                elif not self.simpleFirst:
                    if self.leafStr == self.nodeStr:
                        res += "%s\n" % self.showBranch(node, parent, lev, i)
                    else:
                        res += "%s: %s\n" % (self.showBranch(node, parent, lev, i),
                                             leafsep+self.formatString(self.leafStr, branch, node))
            return res
        else:
            return self.formatString(self.leafStr, node, parent)


    def dumpTree(self):
        if self.nodeStr:
            lev, res = 1, "root: %s\n" % self.formatString(self.nodeStr, self.tree.tree, None)
            self.maxDepth += 1
        else:
            lev, res = 0, ""
        return res + self.dumpTree0(self.tree.tree, None, lev)
        

    def dotTree0(self, node, parent, internalName):
        if node.branches:
            if node.distribution.abs < self.minExamples or len(internalName)-1 > self.maxDepth:
                self.fle.write('%s [ shape="plaintext" label="..." ]\n' % internalName)
                return
                
            label = node.branchSelector.classVar.name
            if self.nodeStr:
                label += "\\n" + self.formatString(self.nodeStr, node, parent)
            self.fle.write('%s [ shape=%s label="%s"]\n' % (internalName, self.nodeShape, label))
            
            for i, branch in enumerate(node.branches):
                if branch:
                    internalBranchName = internalName+chr(i+65)
                    self.fle.write('%s -> %s [ label="%s" ]\n' % (internalName, internalBranchName, node.branchDescriptions[i]))
                    self.dotTree0(branch, node, internalBranchName)
                    
        else:
            self.fle.write('%s [ shape=%s label="%s"]\n' % (internalName, self.leafShape, self.formatString(self.leafStr, node, parent)))


    def dotTree(self, internalName="n"):
        self.fle.write("digraph G {\n")
        self.dotTree0(self.tree.tree, None, internalName)
        self.fle.write("}\n")


def dumpTree(tree, leafStr = "", nodeStr = "", **argkw):
    return __TreeDumper(leafStr, nodeStr, argkw.get("userFormats", []) + __TreeDumper.defaultStringFormats,
                        argkw.get("minExamples", 0), argkw.get("maxDepth", 1e10), argkw.get("simpleFirst", True),
                        tree).dumpTree()


def printTree(*a, **aa):
    print dumpTree(*a, **aa)

printTxt = printTree


def dotTree(tree, fileName, leafStr = "", nodeStr = "", leafShape="plaintext", nodeShape="plaintext", **argkw):
    fle = type(fileName) == str and file(fileName, "wt") or fileName

    __TreeDumper(leafStr, nodeStr, argkw.get("userFormats", []) + __TreeDumper.defaultStringFormats,
                 argkw.get("minExamples", 0), argkw.get("maxDepth", 1e10), argkw.get("simpleFirst", True),
                 tree,
                 leafShape = leafShape, nodeShape = nodeShape, fle = fle).dotTree()
                        
printDot = dotTree
        
##import orange, orngTree, os
##os.chdir("c:\\d\\ai\\orange\\doc\\datasets")
##data = orange.ExampleTable("iris")
###data = orange.ExampleTable("housing")
##tree = orngTree.TreeLearner(data)
##printTxt(tree)
###print printTree(tree, '%V %4.2NbP %.3C!="Iris-virginica"')
###print printTree(tree, '%A %I(95) %C![20,22)bP', ".", maxDepth=3)
###dotTree("c:\\d\\ai\\orange\\x.dot", tree, '%A', maxDepth= 3)
