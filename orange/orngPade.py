import orange, statc
import math, numpy, string, os

from orangeom import star, dist
from copy import deepcopy
from sets import Set

#pathQHULL = r"c:\qhull"
#pathQHULL = r"c:\D\ai\Orange\test\squin\qhull"
if __name__ != "__main__":
    pathQHULL = os.path.split(__file__)[0]

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
    import orangeom
    return orangeom.qhull(cache.data.toNumpy("a")[0][:, cache.contIndices]).tolist()
#    return [map(int, string.split(l[i],' ')) for i in xrange(num_of_triangles+1) if l[i]!='']

# 
def simplex_with_xnDBG(cache, xp, Star, d):
    zm=0.0001 # zm ... zelo malo
    r=[]
    xnd = [xp[i] for i in xrange(len(xp)-1) if i!=d] # vse koordinate, razen d-te
    xndz = xp[-1]
    for s in Star:
        for (p,j) in [(cache.points[i],i) for i in s]:
            pd = [p[i] for i in xrange(len(p)-1) if i!=d] # vse koordinate, razen d-te
            if reduce(lambda x,y: x+y,[math.fabs(i) for i in list(numpy.array(xnd)-numpy.array(pd))])<=zm and math.fabs(xp[d]-p[d])>zm:
                r += [(math.fabs(xp[d]-p[d]),j)]
    if len(r)>0:
        r.sort()
        xdt = cache.points[r[0][1]]
        dt = xdt[d]-xp[d]
        D = (xdt[-1]-xndz)/r[0][0]
        if dt<0:
            D = -D
        return D 
    return None

def simplex_with_xn(cache, xn, Star):
    for simplex in Star:
        bl = [numpy.linalg.det(a) for a in inside(cache, xn, simplex)]
        if reduce(lambda x,y: x and y, [i<0 for i in bl]) or reduce(lambda x,y: x and y, [i>0 for i in bl]):
            return simplex
    return None

def change(cache, i,j,n):
    if i==j:
        return n+[1]
    return cache.points[j][:-1]+[1]

def inside(cache, vertex,simplex):
    return [numpy.array([change(cache, i,j,vertex) for j in simplex]) for i in simplex]


def firstTriangle(cache, dimensions, progressCallback = None, **args):
    if len(cache.contAttributes) == 1:
        return triangles1D(cache, False)

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
        nPoints = 100.0/npoints
        
    for x, (S, xp, dt, deltas) in enumerate(zip(cache.stars, points, cache.dts, cache.deltas)):
        for d in dimensions:
            xn = xp[:-1]
            DBG=0
#            if xn[0]>16 and xn[1]<44.5 and xn[1]>43:
#            #if xn[0]>4.7 and xn[0]<4.9 and xn[1]<24.5 and xn[1]>23.5:
#                DBG=1
#                #print "DBG"
            
            O = numpy.array(xp[:-1])

            xn[d] += dt
            swx = simplex_with_xn(cache, xn, S)
            if swx:
#                if DBG:
#                    print "iskanje cudnih trikotnikov"
#                    print swx
#                    print [points[k] for k in swx]
                obrni = 1
#                if DBG:
#                    print xp
#                    print swx
#                    print [points[k][-1] for k in swx]
#                    vecs = numpy.array([numpy.array(points[p][:-1])-O for p in swx if p!=x])
#                    vecs = vecs.transpose()
#                    XN = numpy.array(xn)-O
#                    coef = numpy.linalg.solve(vecs,XN)
#                    xnz = sum(coef*[numpy.array(points[p][-1]-xp[-1])for p in swx if p!=x])+xp[-1]
#                    dd = obrni * (xnz-xp[-1]) / dt
#                    print "xnz= ",xnz,"\tdd=",dd,"\trazlika (xnz-xp[-1])=",(xnz-xp[-1])
#                    print "-----------------"
            else:
                xn[d] = xp[d]-dt
                swx = simplex_with_xn(cache, xn, S)
                if swx:
                    obrni = -1
#                    if DBG:
#                        print xp
#                        print swx
#                        print "v levo\t",[points[k] for k in swx]
#                        print "----------------"
                else:
#                    if DBG:
#                        print xp
#                        do = simplex_with_xnDBG(cache, xp, S, d)
#                        print do
                    do = simplex_with_xnDBG(cache, xp, S, d)
                    if do:
                        deltas[d] = do
                    else:
                        deltas[d] = "?"
                    continue
            
            vecs = numpy.array([numpy.array(points[p][:-1])-O for p in swx if p!=x])
            vecs = vecs.transpose()
            XN = numpy.array(xn)-O
            coef = numpy.linalg.solve(vecs,XN)
            xnz = sum(coef*[numpy.array(points[p][-1]-xp[-1])for p in swx if p!=x])+xp[-1]

            deltas[d] = obrni * (xnz-xp[-1]) / dt
            #print "DELTAS = ",deltas[d]

        if progressCallback:
            progressCallback(x*nPoints)


# calculates a linear regression on the star
def starRegression(cache, dimensions, progressCallback=None, **args):
    if len(cache.contAttributes) == 1:
        return triangles1D(cache, True)

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
def starUnivariateRegression(cache, dimensions, progressCallback = None, **args):
    if len(cache.contAttributes) == 1:
        return triangles1D(cache, True)
        
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


def triangles1D(cache, bothWays):
    data = cache.data
    attrIdx = cache.contIndices[0]
    points = [(ex[attrIdx], float(ex.getclass()), [i]) for i, ex in enumerate(data) if not ex.getclass().isSpecial()]
    points.sort()

    i, e = 0, len(points)
    while i < e:
        ex = points[i]
        ni = i+1
        while ni < e and points[ni][0] == ex[0]:
            ex[2] += points[ni][2]
            ex[1] += points[ni][1]
            ni += 1
        i = ni
    points = [(a, c/len(i), i) for a, c, i in points]        

    for delta in cache.deltas:
        delta[0] = "?"

    if len(points) < 2:
        return

    der = (points[-1][1] - points[-2][1]) / (points[-1][0] - points[-2][0])
    for pidx in points[-1][2]:
        cache.deltas[pidx][0] = der

    if bothWays:
        der = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0])
        for pidx in points[0][2]:
            cache.deltas[pidx][0] = der

        for i in range(1, len(cache.deltas)-1):
            x1, x2, x3 = points[i-1][0], points[i][0], points[i+1][0]
            sx = x1+x2+x3
            div = 3*(x1**2+x2**2+x3**2) - sx**2
            if div > 1e-6:
                der = (3*(x1*points[i-1][1] + x2*points[i][1] + x3*points[i+1][1]) - sx * (points[i-1][1] + points[i][1] + points[i+1][1])) / div
            for pidx in points[i][2]:
                cache.deltas[pidx][0] = der

    else:
        for i in range(len(cache.deltas)-1):
            der = (points[i][1] - points[i+1][1]) / (points[i][0] - points[i+1][0])
            for pidx in points[i][2]:
                cache.deltas[pidx][0] = der
        
    
# PCA
def pca(self,PCAthreshold=.8):
    if not self.points:        
        self.points = orange.ExampleTable(orange.Domain(self.contAttributes, self.data.domain.classVar), self.data).native(0)
    #x = numpy.array(list(self.points))
    x = numpy.array([[5.6359, 1.8749, 0.2587, 0.9281],[-9.0487, 7.8200, 0.6616, 0.7096],[-7.3699, -3.3882, 0.2115, 0.9430],[-2.1107, 6.4463, 0.6226, 0.1166],[-9.2594, -3.0736, 0.2573, 0.0401],[5.4889, 6.1418, 0.8147, 0.4558],[1.0844, -2.8861, 0.0941, 0.2960],[-8.0125, -7.9141, 0.0397, 0.5051],[-7.2509, -4.6146, 0.2589, 0.3370],[4.3882, 2.4636, 0.9816, 0.9167]])
    xlen = len(x)
    #print x
    #print xlen
    meanx = numpy.mean(x,0)
    #print meanx
    x = numpy.matrix([i-meanx for i in x])
    #print x
    Cov = (numpy.transpose(x)*x)/xlen
    #print "Cov = ",Cov
    evals,evecs = numpy.linalg.eig(Cov)
    sum_evals = sum(list(evals))
    eigv = [(evals[i]/sum_evals,i) for i in xrange(len(evals))]
    eigv.sort()
    eigv.reverse()
    total = 0
    for i in eigv:
        total += i[0]
        if total > PCAthreshold:
            limit = i[1]
            break
    Ev = evecs[:,0:limit+1]
    print eigv
    print Ev
    #pca_data = (Ev' * data')';
    #pca_example = (Ev' * example')'


def Dpca(self):
    self.pca()
    return
    if not self.deltas:
        self.deltas = [[None] * len(self.contAttributes) for x in xrange(len(self.data))]

    dimensions = [d for d in self.dimensions if not self.deltas[0][d]]

    if not dimensions:
        return

    if not self.points:        
        self.points = orange.ExampleTable(orange.Domain(self.contAttributes, self.data.domain.classVar), self.data).native(0)
    points = self.points
    npoints = len(points)

    if not self.tri:
        print self.dimension
        self.tri = self.triangulate(points)
    tri = self.tri

    if not self.stars:
        self.stars = [star(x, tri) for x in xrange(npoints)]
    S = self.stars

    points = import85
    nPoints = 100.0/len(points)

    self.progressBarInit()
    for x,(S,p) in enumerate(zip(self.stars,points)):
        if S==[]:
            self.deltas[x] = ['?' for i in dimensions]
            continue
        st = list(Set(reduce(lambda x,y: x+y, S)))
        lenst = len(st)
        avgy = sum([points[i][-1] for i in st])/lenst
        for di, d in enumerate(dimensions):
            avgx = sum([points[i][di] for i in st])/lenst
            sxx2 = sum([(points[i][di]-avgx)**2 for i in st])
            if sxx2:
                sxx = sum([(points[i][di]-avgx)*(points[i][-1]-avgy) for i in st])
                b = sxx/sxx2
                self.deltas[x][di] = b
            else:
                self.deltas[x][di] = '?'
        self.progressBarSet(x*nPoints)

    self.progressBarFinished()

def star1D(i,p):
    lenp = len(p)
    if lenp==1:
        return []
    if i==0:
        return [p[1]]
    elif i==lenp-1:
        return [p[lenp-2]]
    else:
        return [p[i-1],p[i+1]]

#def qing1D(cache, dimensions, progressCallback = None, persistence=0.4):
#    Dim = len(dimensions)
#    print "dimension = ",Dim
#    if not cache.points:        
#        cache.points = orange.ExampleTable(orange.Domain(cache.contAttributes, cache.data.domain.classVar), cache.data).native(0)
#    p = cache.points
#    lenp = len(p)
#    p.sort()
#    # in > 1D we have to look for the points in the cylinder, project them to 1D and continue as before
#    for d in dimensions:
#        for x, (pts,deltas) in enumerate(zip(cache.points,cache.deltas)):
#            px = pts[0:d]+pts[d+1:]
#            px = px[:-1]
#            print x,px
#            if x>10: break
#        print


def qing1D(cache, dimensions, progressCallback = None, **args):
    persistence = args.get("persistence", 0.4)
    Dim = len(dimensions)
    #print "dimension = ",Dim
    if not cache.points:        
        cache.points = orange.ExampleTable(orange.Domain(cache.contAttributes, cache.data.domain.classVar), cache.data).native(0)
    p = cache.points
    lenp = len(p)
    p.sort()
    
    # in > 1D we have to look for the points in the cylinder, project them to 1D and continue as before
    if Dim>1:
        pass
        
    sosedi = [(p[i],star1D(i,p)) for i in xrange(lenp)]
    mini = []
    maxi = []
    for i,s in enumerate(sosedi):
        if s[0][1] < min([j[1] for j in s[1]]):
            mini += [i]
        elif s[0][1] > max([j[1] for j in s[1]]):
            maxi += [i]
            
    #print "vsi minimumi: ",mini
    #print "vsi maximumi: ",maxi
    mini1 = deepcopy(mini)

    for i in mini:
        fv = p[i][1]
        left = [j for j in maxi if j<i]
        if left:
            left = max(left)
            left = (math.fabs(fv-p[left][1]),left)
        right = [j for j in maxi if j>i]
        if right:
            right = min(right)
            right = (math.fabs(fv-p[right][1]),right)
        t = [(i,fv)]
        ti = []
        if left:
            ti += [left]
        if right:
            ti += [right]
        mti = min(ti)
        if mti[0] <= persistence:
            maxi.remove(mti[1])
            mini1.remove(i)
    #print "min. po krajsanju: ",mini1
    #print "max. po krajsanju: ",maxi
    ext = [(p[m],-1) for m in mini1]+[(p[m],1) for m in maxi]
    ext.sort()
    elist = zip(mini1,[-1]*len(mini1))+zip(maxi,[1]*len(maxi))
    elist.sort()
    ekstremi = []
    if elist[0][1]==-1: # ce je najprej minimum
        ekstrem = min([(p[k][-1],k,-1) for k in xrange(maxi[0])]) # poiscemo pravi min med tockami do prvega maxa
    else:
        ekstrem = max([(p[k][-1],k,1) for k in xrange(mini1[0])]) # sicer poiscemo prvi max med tockami do prvega mina
    ekstremi += [ekstrem]
    for i in xrange(1,len(elist)-1):
        if elist[i][-1]==1:
            ekstrem = max([(p[k][-1],k,1) for k in xrange(elist[i-1][0],elist[i+1][0]+1)])
        else:
            ekstrem = min([(p[k][-1],k,-1) for k in xrange(elist[i-1][0],elist[i+1][0]+1)])
        ekstremi += [ekstrem]        

    if elist[-1][1]==-1: # ce je zadnji ekstrem minimum
        ekstrem = min([(p[k][-1],k,-1) for k in xrange(maxi[-1],lenp)]) # poiscemo pravi min med tockami do prvega maxa
    else:
        ekstrem = max([(p[k][-1],k,1) for k in xrange(mini1[-1],lenp)]) # sicer poiscemo prvi max med tockami do prvega mina
    ekstremi += [ekstrem]
    lookup = [k[1:] for k in ekstremi]
    if progressCallback:
        nPoints = 100.0/lenp
    for x, deltas in enumerate(cache.deltas):
        for d in dimensions:
            #print x,p[x]
            # ce je x ekstrem, ima odvod 0, oz. gledamo levo/desno limito, ce je na robu
            if x in [j[0] for j in lookup]:
                if x == lookup[0][0]:
                    #deltas[d] = -lookup[0][1]
                    deltas[d] = (p[lookup[1][0]][1] - p[x][1])/(p[lookup[1][0]][0] - p[x][0])
                elif x == lookup[-1][0]:
                    #deltas[d] = lookup[-1][1]
                    deltas[d] = (p[lookup[-2][0]][1] - p[x][1])/(p[lookup[-2][0]][0] - p[x][0])
                else:
                    deltas[d] = 0.
                continue
            j=0
            while lookup[j][0] < x:
                j+=1
            if lookup[j][1] == 1:
                #deltas[d] = 1. #obrni * (xnz-xp[-1]) / dt
                deltas[d] = (p[x][1] - p[lookup[j][0]][1])/(p[x][0] - p[lookup[j][0]][0])
            else:
                #deltas[d] = -1.
                deltas[d] = (p[x][1] - p[lookup[j][0]][1])/(p[x][0] - p[lookup[j][0]][0])
        if progressCallback:
            progressCallback(x*nPoints)

    #print [k[-1] for k in ekstremi]


# regression in a tube
def tubedRegression(cache, dimensions, progressCallback = None, **args):
    if not cache.findNearest:
        cache.findNearest = orange.FindNearestConstructor_BruteForce(cache.data, distanceConstructor=orange.ExamplesDistanceConstructor_Euclidean(), includeSame=True)
        
    if not cache.attrStat:
        cache.attrStat = orange.DomainBasicAttrStat(cache.data)

    normalizers = cache.findNearest.distance.normalizers

    if progressCallback:
        nExamples = len(cache.data)
        nPoints = 100.0/nExamples/len(dimensions)

    effNeighbours = len(cache.contAttributes) > 1 and cache.nNeighbours or len(cache.deltas)
    
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

            nn = cache.findNearest(ref_example, 0, True)
            nn = [ex for ex in nn if not ex[contIdx].isSpecial()][:effNeighbours]
            mx = [abs(ex[contIdx] - ref_x) for ex in nn]
            if not mx:
                cache.deltas[exi][d] = "?"
                continue
            if max(mx) < 1e-10:
                kw = math.log(.001)
            else:
                kw = math.log(.001) / max(mx)**2
            for ex in nn[:effNeighbours]:
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
            if div:# and i<40:
                b = (Sxy*n - Sx*Sy) / div
                
#                div = Sx*Sy/n - Sxy
#                if abs(div) < 1e-10:
#                    cache.errors[exi][d] = 1
#                else:
#                    B = ((Syy - Sy**2/n) - (Sxx - Sx**2/n)) / 2 / div
#
#                    b_p = -B + math.sqrt(B**2+1)
#                    a = Sy/n - b_p * Sx/n
#                    error1 = 1/(1+b_p**2) * (Syy + a**2 + b_p**2*Sxx - 2*a*Sy + 2*a*b_p*Sx - 2*b_p*Sxy)
#
#                    b_2 = -B - math.sqrt(B**2+1)
#                    a = Sy/n - b_p * Sx/n
#                    error2 = 1/(1+b_p**2) * (Syy + a**2 + b_p**2*Sxx - 2*a*Sy + 2*a*b_p*Sx - 2*b_p*Sxy)
#                    
#                    if error1 < error2 and error1 >= 0:
#                        cache.errors[exi][d] = error1
#                    elif error2 >= 0:
#                        cache.errors[exi][d] = error2
#                    else:
#                        cache.errors[exi][d] = 42
#                        print error1, error2
                            
                a = (Sy - b*Sx)/n
                err = (n * a**2 + b**2 * Sxx + Syy + 2*a*b*Sx - 2*a*Sy - 2*b*Sxy)
                tot = Syy - Sy**2/n
                mod = tot - err
                merr = err/(n-2)
                if merr < 1e-10:
                    F = 0
                    Fprob = 1
                else:
                    F = mod/merr
                    Fprob = statc.fprob(F, 1, int(n-2))
                cache.errors[exi][d] = Fprob
#                        print "%.4f" % Fprob,
                #print ("%.3f\t" + "%.0f\t"*6 + "%f\t%f") % (w, ref_x, ex_x, n, a, b, merr, F, Fprob)
                cache.deltas[exi][d] = b
            else:
                cache.deltas[exi][d] = "?"

            if progressCallback:
                progressCallback((nExamples*di+exi)*nPoints)

        normalizers[cache.contIndices[d]] = oldNormalizer


def createClassVar(attributes, MQCNotation = False):
    import orngMisc
    if MQCNotation:
        return orange.EnumVariable("Q", values = ["%s(%s)" % ("".join(["+-"[x] for x in v if x<2]), ", ".join([attr for attr,x in zip(attributes, v) if x<2])) for v in orngMisc.LimitedCounter([3]*len(attributes))])
    else:
        return orange.EnumVariable("Q", values = ["Q(%s)" % ", ".join(["+-"[x]+attr for attr, x in zip(attributes, v) if x<2]) for v in orngMisc.LimitedCounter([3]*len(attributes))])

    
def createQTable(cache, data, dimensions, outputAttr = -1, threshold = 0, MQCNotation = False, derivativeAsMeta = False, differencesAsMeta = False, correlationsAsMeta = False, originalAsMeta = False):
    nDimensions = len(dimensions)
    
    needQ = outputAttr < 0 or derivativeAsMeta
    if needQ:
        qVar = createClassVar([cache.attributes[i][0] for i in dimensions], MQCNotation)
        
    if outputAttr >= 0:
        classVar = orange.FloatVariable("df/d"+cache.attributes[outputAttr][0])
    else:
        classVar = qVar
            
    dom = orange.Domain(data.domain.attributes, classVar)
    dom.addmetas(data.domain.getmetas())
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

    corMetaIDs = []
    if correlationsAsMeta:
        for dim in dimensions:
            metaVar = orange.FloatVariable("corr(%s)" % cache.attributes[dim][0])
            metaID = orange.newmetaid()
            dom.addmeta(metaID, metaVar)
            corMetaIDs.append(metaID)
        metaVar = orange.FloatVariable("corr")
        metaID = orange.newmetaid()
        dom.addmeta(metaID, metaVar)
        corMetaIDs.append(metaID)

    if originalAsMeta:
        originalID = orange.newmetaid()
        dom.addmeta(originalID, data.domain.classVar)
    else:
        originalID = 0


    paded = orange.ExampleTable(dom, data)

    for i, (pad, alldeltas) in enumerate(zip(paded, cache.deltas)):
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

        if correlationsAsMeta:
            if hasattr(cache, "errors"):
                maxerr = -1e20
                for id, val in zip(corMetaIDs, [cache.errors[i][d] for d in dimensions]):
                    if val == None:
                        pad.setmeta(id, "?")
                    else:
                        pad.setmeta(id, val)
                        maxerr = max(maxerr, val)
                pad.setmeta(corMetaIDs[-1], maxerr)
            else:
                minder = 0
                for id, val in zip(corMetaIDs[:-1], deltas):
                    if type(val) == str:
                        pad.setmeta(id, "?")
                    else:
                        pad.setmeta(id, abs(val))
                        minder = min(minder, abs(val))
                pad.setmeta(corMetaIDs[-1], minder)
                
    return paded, derivativeID, metaIDs, corMetaIDs, originalID


def pade(data, attributes = None, method = tubedRegression, outputAttr = -1, threshold = 0, MQCNotation = False, derivativeAsMeta = False, differencesAsMeta = False, correlationsAsMeta = False, originalAsMeta = False):
    cache = makeBasicCache(data)
    cache.deltas = [[None] * len(cache.contAttributes) for x in xrange(len(data))]
    if method == tubedRegression:
        cache.errors = [[None] * len(cache.contAttributes) for x in xrange(len(data))]

    cache.nNeighbours = 30

    if not attributes:
        attributes = range(len(cache.contAttributes))
        
    dimensions = [data.domain.index(attr) for attr in attributes]

    if outputAttr != -1:
        outputAttr = data.domain.index(outputAttr)
        if outputAttr not in dimensions:
            dimensions.append(outputAttr)

    method(cache, dimensions)
    return createQTable(cache, data, dimensions, outputAttr, threshold, MQCNotation, derivativeAsMeta, differencesAsMeta, correlationsAsMeta, originalAsMeta)


### Quin-like measurement of quality
#
# We consider these measures completely inappropriate for various reasons.
# We nevertheless implemented them for the sake of comparison.

# puts examples from 'data' into the corresponding leaves of the subtree
def splitDataOntoNodes(node, data):
    if not node.branches:
        node.setattr("leafExamples", data)
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
                spairs += 1
    return samb, sacc, spairs


def computeAmbiguityAccuracy(tree, data, clsID):
    splitDataOntoNodes(tree.tree, data)
    l = tree.domain.constraintAttributes[:]
    l.reverse()
    amb, acc, pairs = computeAmbiguityAccuracyNode(tree.tree, l, clsID)
    return amb/pairs, acc/(pairs-amb)


def CVByNodes(data, dimensions = None, method = None, **dic):
    import orngTree
    cv = orange.MakeRandomIndicesCV(data, 10)
    for fold in range(10):
        train = data.select(cv, fold, negate=1)
        test = data.select(cv, fold)
        pa, qid, did, cid = pade(train, dimensions, method, originalAsMeta=True, **dic)
        tree = orngTree.TreeLearner(pa, maxDepth=4)

        mb, cc = computeAmbiguityAccuracy(tree, test, -1)
        amb += mb
        acc += cc
    return amb/10, acc/10


### Better measures of quality (as in better-than-Quin)
#
def checkDirectionAccuracyForPair(model, reference, direction, clsID, reversedAttributes):
    constraint = int(model(reference))
    prediction = 0

    for attr in reversedAttributes:
        constraint, tc = constraint/3, constraint%3
        if tc==2:
            continue

        v1, v2 = reference[attr], direction[attr]
        if v1.isSpecial() or v2.isSpecial():
            continue

        if tc:
            c = -cmp(v1, v2)
        else:
            c = cmp(v1, v2)

        if not prediction:
            prediction = c
        elif prediction != c:
            return -1

    if not prediction:
        return -3
    
    return c == cmp(reference.getclass(), direction.getclass())


def computeDirectionAccuracyForPairs(model, data, meter, weightK, clsID, nTests = 0):
    nTests = nTests or 10*len(data)

    reversedAttrs = model.domain.constraintAttributes[:]
    reversedAttrs.reverse()

    actTests = acc = amb = unre = 0
    for i in range(10*len(data)):
        distance = 0
        while not distance:
            ref, dir = data.randomexample(), data.randomexample()
            distance = meter(ref, dir)
        weight = math.exp(weightK * distance**2)
        actTests += weight
        diracc = checkDirectionAccuracyForPair(model, ref, dir, -1, reversedAttrs)
        if diracc == -1:
            amb += weight
        elif diracc == -3:
            unre += weight
        elif diracc:
            acc += weight

    return acc/actTests, amb/actTests, unre/actTests
        

def CVByPairs(data, dimensions = None, method = None, **dic):
    import orngTree
    cv = orange.MakeRandomIndicesCV(data, 10)
    meter = orange.ExamplesDistanceConstructor_Euclidean(data)

    maxDist = 0
    for i in range(100):
        maxDist = max(maxDist, meter(data.randomexample(), data.randomexample()))
    weightK = 10.0 / maxDist

    acc = amb = unre = 0
    for fold in range(10):
        train = data.select(cv, fold, negate=1)
        test = data.select(cv, fold)
        pa, qid, did, cid = pade(train, dimensions, method, originalAsMeta=True, **dic)
        tree = orngTree.TreeLearner(pa, maxDepth=4)

        tacc, tamb, tunre = computeDirectionAccuracyForPairs(tree, data, meter, weightK, -1)
        acc += tacc
        amb += tamb
        unre += tunre
        
    return acc/10, amb/10, unre/10


import re
sidx = {"+": "0", "-": "1", None: "2"}
numb = r"[+-]?\d*(\.\d*)?(e[+-]\d+)?"
constraint = r"M(?P<signs>[+-](,[+-])*)\((?P<attrs>[^ ,)]*(,[^ ,)]*)*)\)\s*\(Cov=(?P<cov>(\*\*)|("+numb+r"%="+numb+r"))\(\?(?P<amb>"+numb+r")%\)/(?P<xmpls>"+numb+r")\)"
re_constraint = re.compile(constraint)
re_node = re.compile(r"\s*(?P<splitattr>[^ ]*)\s*<=\s*(?P<thresh>[\-0-9.]*)\s*\[\s*"+constraint+r"\]")

def readQuinTreeRec(fle, domain):
    node = orange.TreeNode()
    line = fle.readline()

    match = re_node.search(line)
    if match:
        splitattr, thresh = match.group("splitattr", "thresh")
        node.branchSelector = orange.Classifier(lambda ex,rw, attr=domain[splitattr], thr=float(thresh): ex[attr] > thr)
        node.branchDescriptions = ["<= "+thresh, "> "+thresh]
        node.branches = [readQuinTreeRec(fle, domain), readQuinTreeRec(fle, domain)]
        node.branchSizes = [node.branches[0].nExamples, node.branches[1].nExamples]
    else:
        match = re_constraint.search(line)
        if not match:
            raise "Cannot read line '%s'" % line

    attrs, signs, cov, amb, xmpls = match.group("attrs", "signs", "cov", "amb", "xmpls")

    mqc = dict(zip(attrs.split(","), signs.split(",")))
    node.nodeClassifier = orange.DefaultClassifier(defaultVal = int("".join([sidx[mqc.get(attr.name, None)] for attr in domain.attributes]), 3))
    node.setattr("coverage", cov == "**" and -1 or float(cov[:cov.index("%")]))
    node.setattr("ambiguity", float(amb))
    node.setattr("nExamples", float(xmpls))
    return node

def readQuinTree(fname, domain):
    qVar = createClassVar([attr.name for attr in domain.attributes])
    domain.setattr("constraintAttributes", domain.attributes)
    return orange.TreeClassifier(domain = domain, classVar = domain.classVar, tree = readQuinTreeRec(open(fname), domain))

