import orange, statc
import math, numpy, string, os

from orangeom import star, dist
from sets import Set

#pathQHULL = r"c:\qhull"
pathQHULL = r"c:\D\ai\Orange\test\squin\qhull"

class Cache:
    pass


def clearCache(cache):
    cache.points = cache.tri = cache.stars = cache.dts = cache.deltas = cache.findNearest = cache.attrStat = None


def makeEmptyCache(cache = None):
    if not cache:
        cache = Cache()
    cache.data = None
    cache.attributes = []
    cache.contAttributes = []
    cache.npoints = 0
    clearCache(cache)
    return cache


def makeBasicCache(data, cache = None):
    if not cache:
        cache = Cache()

    cache.data = data              
    cache.npoints = len(data)

    attributes = cache.data.domain.attributes            
    cache.contIndices = [i for i, attr in enumerate(attributes) if attr.varType == orange.VarTypes.Continuous]
    cache.contAttributes = [attributes[i] for i in cache.contIndices]
    cache.attributes = [(attr.name, attr.varType) for attr in cache.contAttributes]

    clearCache(cache)
    return cache



def triangulate(cache, points):
    num_points = len(points)
    pts1 = points
    f = file('input4qdelaunay.tab','w')
    f.write(reduce(lambda x, y: x+y, [str(len(cache.contAttributes))+"\n"+str(len(pts1))+"\n"]+ [string.join([str(x) for x in pts1[i][:-1]],'\t')+'\n' for i in xrange(num_points)] )) # [str(pts1[i][0])+"\t"+str(pts1[i][1])+"\n" for i in xrange(num_points)]
    f.close()
    os.system(pathQHULL + r"\qdelaunay s i Qt TO 'outputFromQdelaunay.tab' < input4qdelaunay.tab")
    f = file('outputFromQdelaunay.tab','r')
    vhod = f.read()
    f.close()
    k = string.find(vhod,'\n')
    num_of_triangles = int(vhod[0:k])
    vhod = vhod[k+1:]
    l = string.split(vhod,' \n')
    return [map(int, string.split(l[i],' ')) for i in xrange(num_of_triangles+1) if l[i]!='']



def simplex_with_xn(cache, xn,Star):
    for simplex in Star:
        bl = [numpy.linalg.det(a) for a in inside(cache, xn,simplex)]
        if reduce(lambda x,y: x and y, [i<0 for i in bl]) or reduce(lambda x,y: x and y, [i>0 for i in bl]):
            return simplex
    return None

def change(cache, i,j,n):
    if i==j:
        return n+[1]
    return cache.points[j][:-1]+[1]

def inside(cache, vertex,simplex):
    return [numpy.array([change(cache, i,j,vertex) for j in simplex]) for i in simplex]


def firstTriangle(cache, dimensions, progressCallback = None):
    if not cache.points:        
        cache.points = orange.ExampleTable(orange.Domain(cache.contAttributes, cache.data.domain.classVar), cache.data).native(0)
    points = cache.points
    npoints = len(points)

    if not cache.tri:
        cache.tri = triangulate(cache, points)
    tri = cache.tri
        
    if not cache.stars:
        cache.stars = [star(x, tri) for x in xrange(npoints)]
    S = cache.stars

    if not cache.dts:        
        cache.dts = [min([ min([ dist(points[x][:-1],points[v][:-1]) for v in simplex if v!=x]) for simplex in S[x]])*.1 for x in xrange(npoints)]


    if progressCallback:
        nPoints = 100.0/len(points)
        
    for x, (S, xp, dt, deltas) in enumerate(zip(cache.stars, points, cache.dts, cache.deltas)):
        for d in dimensions:

            xn = xp[:-1]
            O = numpy.array(xp[:-1])

            xn[d] += dt
            swx = simplex_with_xn(cache, xn, S)
            if swx:                
                obrni = 1
            else:
                xn[d] = xp[d]-dt
                swx = simplex_with_xn(cache, xn, S)
                if swx:
                    obrni = -1
                else:
                    deltas[d] = "?"
                    continue

            vecs = numpy.array([numpy.array(points[p][:-1])-O for p in swx if p!=x])
            vecs = vecs.transpose()
            XN = numpy.array(xn)-O
            coef = numpy.linalg.solve(vecs,XN)
            xnz = sum(coef*[numpy.array(points[p][-1]-xp[-1])for p in swx if p!=x])+xp[-1]

            deltas[d] = obrni * (xnz-xp[-1]) / dt

        if progressCallback:
            progressCallback(x*nPoints)


# calculates a linear regression on the star
def starRegression(cache, dimensions, progressCallback=None):
    if not cache.points:        
        cache.points = orange.ExampleTable(orange.Domain(cache.contAttributes, cache.data.domain.classVar), cache.data).native(0)
    points = cache.points
    npoints = len(points)

    if not cache.tri:
        cache.tri = triangulate(cache, points)
    tri = cache.tri
        
    if not cache.stars:
        cache.stars = [star(x, tri) for x in xrange(npoints)]
    S = cache.stars

    points = cache.points

    if progressCallback:
        nPoints = 100.0/len(points)
        
    for x,(S,p) in enumerate(zip(cache.stars,points)):
        if S==[]:
            cache.deltas[x] = ['?' for i in dimensions]
            continue
        st  =list(Set(reduce(lambda x,y: x+y, S)))
        A = [points[i][:-1] for i in st]
        b = [[points[i][-1]] for i in st]
        cache.deltas[x] = [i[0] for i in numpy.linalg.lstsq(A, b)[0]]

        if progressCallback:
            progressCallback(x*nPoints)

        
# calculates a univariate linear regression on the star
def starUnivariateRegression(cache, dimensions, progressCallback = None):
    if not cache.points:        
        cache.points = orange.ExampleTable(orange.Domain(cache.contAttributes, cache.data.domain.classVar), cache.data).native(0)
    points = cache.points
    npoints = len(points)

    if not cache.tri:
        cache.tri = triangulate(cache, points)
    tri = cache.tri
        
    if not cache.stars:
        cache.stars = [star(x, tri) for x in xrange(npoints)]
    S = cache.stars

    points = cache.points

    if progressCallback:
        nPoints = 100.0/len(points)
        
    for x,(S,p) in enumerate(zip(cache.stars,points)):
        if S==[]:
            cache.deltas[x] = ['?' for i in dimensions]
            continue
        st = list(Set(reduce(lambda x,y: x+y, S)))
        lenst = len(st)
        avgy = sum([points[i][-1] for i in st])/lenst
        for d in dimensions:
            avgx = sum([points[i][d] for i in st])/lenst
            sxx2 = sum([(points[i][d]-avgx)**2 for i in st])
            if sxx2:
                sxx = sum([(points[i][d]-avgx)*(points[i][-1]-avgy) for i in st])
                b = sxx/sxx2
                cache.deltas[x][d] = b
            else:
                cache.deltas[x][d] = '?'

        if progressCallback:
            progressCallback(x*nPoints)



# regression in a tube
def tubedRegression(cache, dimensions, progressCallback = None):
    if not cache.findNearest:
        cache.findNearest = orange.FindNearestConstructor_BruteForce(cache.data, distanceConstructor=orange.ExamplesDistanceConstructor_Euclidean(), includeSame=False)
        
    if not cache.attrStat:
        cache.attrStat = orange.DomainBasicAttrStat(cache.data)

    normalizers = cache.findNearest.distance.normalizers

    if progressCallback:
        nExamples = len(cache.data)
        nPoints = 100.0/nExamples/len(dimensions)
    
    for di, d in enumerate(dimensions):
        contIdx = cache.contIndices[d]

        minV, maxV = cache.attrStat[contIdx].min, cache.attrStat[contIdx].max
        if minV == maxV:
            continue
        
        oldNormalizer = normalizers[cache.contIndices[d]]
        normalizers[cache.contIndices[d]] = 0

        for exi, ref_example in enumerate(cache.data):
            if ref_example[contIdx].isSpecial():
                cache.deltas[exi][d] = "?"
                continue

            ref_x = float(ref_example[contIdx])

            Sx = Sy = Sxx = Syy = Sxy = n = 0.0

            nn = cache.findNearest(ref_example, cache.nNeighbours, True)
            mx = [abs(ex[contIdx] - ref_x) for ex in nn if not ex[contIdx].isSpecial()]
            # Tole ni prav - samo prevec enakih je...
            if not mx:
                cache.deltas[exi][d] = "?"
                continue
            
            kw = math.log(.001) / max(mx)**2
            for ex in nn:
                if ex[contIdx].isSpecial():
                    continue
                ex_x = float(ex[contIdx])
                ex_y = float(ex.getclass())
                w = math.exp(kw*(ex_x-ref_x)**2)
                Sx += w * ex_x
                Sy += w * ex_y
                Sxx += w * ex_x**2
                Syy += w * ex_y**2
                Sxy += w * ex_x * ex_y
                n += w

            div = n*Sxx-Sx**2
            if div and n>=3:# and i<40:
                b = (Sxy*n - Sx*Sy) / div
##                    a = (Sy - b*Sx)/n
##                    err = (n * a**2 + b**2 * Sxx + Syy + 2*a*b*Sx - 2*a*Sy - 2*b*Sxy)
##                    tot = Syy - Sy**2/n
##                    mod = tot - err
##                    merr = err/(n-2)
##                    F = mod/merr
##                    Fprob = statc.fprob(F, 1, int(n-2))
#                        print "%.4f" % Fprob,
                #print ("%.3f\t" + "%.0f\t"*6 + "%f\t%f") % (w, ref_x, ex_x, n, a, b, merr, F, Fprob)
                cache.deltas[exi][d] = b
            else:
                cache.deltas[exi][d] = "?"

            if progressCallback:
                progressCallback((nExamples*di+exi)*nPoints)

        normalizers[cache.contIndices[d]] = oldNormalizer


def createQTable(cache, data, dimensions, outputAttr = -1, threshold = 0, MQCNotation = False, derivativeAsMeta = False, differencesAsMeta = False, originalAsMeta = False):
    nDimensions = len(dimensions)
    
    needQ = outputAttr < 0 or derivativeAsMeta
    if needQ:
        import orngMisc
        if MQCNotation:
            qVar = orange.EnumVariable("Q", values = ["M%s(%s)" % ("".join(["+-"[x] for x in v if x<2]), ", ".join([cache.attributes[i][0] for i,x in zip(dimensions, v) if x<2])) for v in orngMisc.LimitedCounter([3]*nDimensions)])
        else:
            qVar = orange.EnumVariable("Q", values = ["M(%s)" % ", ".join(["+-"[x]+cache.attributes[i][0] for i, x in zip(dimensions, v) if x<2]) for v in orngMisc.LimitedCounter([3]*nDimensions)])
        
    if outputAttr >= 0:
        classVar = orange.FloatVariable("df/d"+cache.attributes[outputAttr][0])
    else:
        classVar = qVar
            
    dom = orange.Domain(data.domain.attributes, classVar)
    setattr(dom, "constraintAttributes", [cache.contAttributes[i] for i in dimensions])

    if derivativeAsMeta:
        derivativeID = orange.newmetaid()
        dom.addmeta(derivativeID, qVar)
    else:
        derivativeID = 0
            
    metaIDs = []        
    if differencesAsMeta:
        for dim in dimensions:
            metaVar = orange.FloatVariable("df/d"+cache.attributes[dim][0])
            metaID = orange.newmetaid()
            dom.addmeta(metaID, metaVar)
            metaIDs.append(metaID)

    if originalAsMeta:
        originalID = orange.newmetaid()
        dom.addmeta(originalID, data.domain.classVar)
    else:
        originalID = 0


    paded = orange.ExampleTable(dom, data)

    for pad, alldeltas in zip(paded, cache.deltas):
        deltas = [alldeltas[d] for d in dimensions]

        if needQ:
            qs = "".join([(delta > threshold and "0") or (delta < -threshold and "1") or (delta == "?" and "?") or "2" for delta in deltas])
            q = ("?" in qs and "?") or int(qs, 3)
                
        if outputAttr >= 0:
            pad.setclass(alldeltas[outputAttr])
        else:
            pad.setclass(q)

        if derivativeAsMeta:
            pad.setmeta(derivativeID, q)

        if differencesAsMeta:
            for a in zip(metaIDs, deltas):
                pad.setmeta(*a)

    return paded, derivativeID, metaIDs, originalID


def pade(data, attributes = None, method = tubedRegression, outputAttr = -1, threshold = 0, MQCNotation = False, derivativeAsMeta = False, differencesAsMeta = False, originalAsMeta = False):
    cache = makeBasicCache(data)
    cache.deltas = [[None] * len(cache.contAttributes) for x in xrange(len(data))]
    cache.nNeighbours = 30

    if not attributes:
        attributes = range(len(cache.contAttributes))
        
    dimensions = [data.domain.index(attr) for attr in attributes]

    if outputAttr != -1:
        outputAttr = data.domain.index(outputAttr)
        if outputAttr not in dimensions:
            dimensions.append(outputAttr)

    method(cache, dimensions)
    return createQTable(cache, data, dimensions, outputAttr, threshold, MQCNotation, derivativeAsMeta, differencesAsMeta, originalAsMeta)


### Quin-like measurement of quality
#
# We consider these measures completely inappropriate for various reasons.
# We nevertheless implemented them for the sake of comparison.

# puts examples from 'data' into the corresponding leaves of the subtree
def splitDataOntoNodes(node, data):
    if not node.branches:
        setattr(node, "leafExamples", data)
    else:
        goeswhere = [(ex, int(node.branchSelector(ex))) for ex in data]
        for bi, branch in enumerate(node.branches):
            splitDataOntoNodes(branch, [ex for ex, ii in goeswhere if ii == bi])


# checks whether 'ex1' and 'ex2' match the given constraint
# 'reversedAttributes' is a reversed list of the arguments of the QMC
# 'constraint' is the index of the class
# 'clsID' is the id of the meta attribute with the original class value
#
# Result: -2 classes or all attribute values are unknown or same for both
#            examples. This not considered a model's fault and doesn't count
#         -1 Ambiguity: one attribute value goes along the constraint and
#            another goes the opposite. This is ambiguity due to the model.
#          0 False prediction
#          1 Correct prediction
#
# Note: Quin does not distinguish between ambiguity due to the data and
#   due to the model. We understand the ambiguity in the data as
#   unavoidable and don't count the corresponding example pairs, while
#   ambiguity due to the model is model's fault and could even be
#   counted as misclassification

def checkMatch(ex1, ex2, reversedAttributes, constraint, clsID):
    cls1, cls2 = ex1[clsID], ex2[clsID]
    if cls1.isSpecial() or cls2.isSpecial():
        return -2  # unknown classes - useless example
    clsc = cmp(cls1, cls2)
    if not clsc:
        return -2  # ambiguity due to same class
    
    cs = 0
    for attr in reversedAttributes:
        if not constraint:
            break
        constraint, tc = constraint/3, constraint%3
        if tc==2:
            continue
        v1, v2 = ex1[attr], ex2[attr]
        if v1.isSpecial() or v2.isSpecial():
            return -2   # unknowns - useless example
        if tc:
            c = -cmp(v1, v2)
        else:
            c = cmp(v1, v2)
        if not cs:
            cs = c
        elif c != cs:
            return -1   # ambiguity due to example pair
    if not cs:
        return -2       # ambiguity due to same example values

    return cs == clsc


def computeAmbiguityAccuracyNode(node, reversedAttributes, clsID):
    samb = sacc = spairs = 0.0
    if node.branches:
        for branch in node.branches:
            if branch:
                amb, acc, pairs = computeAmbiguityAccuracyNode(branch, reversedAttributes, clsID)
                samb += amb
                sacc += acc
                spairs += pairs
    else:
        constraint = int(node.nodeClassifier.defaultVal)
        for i, ex1 in enumerate(node.leafExamples):
            for j in range(i):
                ma = checkMatch(ex1, node.leafExamples[j], reversedAttributes, constraint, clsID)
                if ma == -2:
                    continue
                if ma == -1:
                    samb += 1
                elif ma:
                    sacc += 1
                else:
                    ex2 = node.leafExamples[j]
#                    print "wrong: %s-%s  %s-%s  %s-%s %i" % (ex1[0], ex2[0], ex1[1], ex2[1], ex1[clsID], ex2[clsID], constraint)
                spairs += 1
    return samb, sacc, spairs


def computeAmbiguityAccuracy(tree, data, clsID):
    splitDataOntoNodes(tree.tree, data)
    l = tree.domain.constraintAttributes[:]
    l.reverse()
    amb, acc, pairs = computeAmbiguityAccuracyNode(tree.tree, l, clsID)
    return amb/pairs, acc/(pairs-amb)