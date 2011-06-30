#
# Module Orange Contingency Tables
# --------------------------------
#
# CVS Status: $Id$
#
# Author: Aleks Jakulin (jakulin@acm.org)
# (Copyright (C)2004 Aleks Jakulin)
#
# Purpose: Management of contingency tables and various heuristics.
#          For performance reasons, handling of 2 and 3-way tables
#          is handled separately.
#
# Project initiated on 2003/01/30
#
# ChangeLog:
#


import numpy, orange, statc
from random import *

_log2 = 1.0/numpy.log(2)
_log2e = numpy.log(2)

def Flatten(m):
    if len(numpy.shape(m)) > 1:
        v = numpy.ravel(m)
    else:
        v = m
    return v

def Probabilities(m):
    t = numpy.sum(Flatten(m))
    if t == 0:
        return 0
    else:
        return m/t

def Entropy(m):
    v = Flatten(m)
    pv = Probabilities(v)
    lv = numpy.log(numpy.clip(pv,1e-6,1.0))*_log2
    return -numpy.dot(pv,lv)


class ContingencyTable3:
    def InteractionInformation(self):
        abc = Entropy(self.m)
        return (-Entropy(self.a)-Entropy(self.b)-Entropy(self.c)+Entropy(self.ab)+Entropy(self.bc)+Entropy(self.ac)-abc)

    def CMI(self):
        # the three conditional mutual informations
        abc = Entropy(self.m)
        c = (-Entropy(self.c)+Entropy(self.bc)+Entropy(self.ac)-abc)
        b = (-Entropy(self.b)+Entropy(self.ab)+Entropy(self.bc)-abc)
        a = (-Entropy(self.a)+Entropy(self.ab)+Entropy(self.ac)-abc)
        return (a,b,c)

    def JaccardInteraction(self):
        abc = Entropy(self.m)
        return (-Entropy(self.a)-Entropy(self.b)-Entropy(self.c)+Entropy(self.ab)+Entropy(self.bc)+Entropy(self.ac)-abc)/abc

    def NormDivergence(self):
        try:
            return self.kirknorm
        except:
            s = 0.0
            for x in range(len(self.values[0])):
                for y in range(len(self.values[1])):
                    for z in range(len(self.values[2])):
                        s += self.Divergence(x,y,z)[1]
            self.kirknorm = s
            return s

    def IPF(self,tolerance=1e-6,maxiterations=100):
        d = numpy.shape(self.m)
        p = self.pm
        px = self.pab
        py = self.pac
        pz = self.pbc
        pxx = self.pa
        pyy = self.pb
        pzz = self.pc
        v = 1.0/(d[0]*d[1]*d[2])
        model = numpy.ones(d,numpy.float)*v

        iterations = 0
        diff = 1e30
        pdiv = 1e50
        while iterations < maxiterations and abs(diff) > tolerance:
            # FIT
            for c in range(3):
                if c == 0:
                    mx = Probabilities(numpy.sum(model,axis=2))
                elif c == 1:
                    my = Probabilities(numpy.sum(model,axis=1))
                else:
                    mz = Probabilities(numpy.sum(model,axis=0))
                for x in xrange(d[0]):
                    for y in xrange(d[1]):
                        for z in xrange(d[2]):
                            if c == 0:
                                model[x,y,z] *= px[x,y]/max(mx[x,y],1e-16)
                            elif c == 1:
                                model[x,y,z] *= py[x,z]/max(my[x,z],1e-16)
                            else:
                                model[x,y,z] *= pz[y,z]/max(mz[y,z],1e-16)
            # EVALUATE
            div = 0.0
            for x in xrange(d[0]):
                for y in xrange(d[1]):
                    for z in xrange(d[2]):
                        if p[x,y,z]>0:
                            div += p[x,y,z]*numpy.log(p[x,y,z]/model[x,y,z])
            #print iterations, Entropy(model), div
            iterations += 1
            diff = pdiv-div
            pdiv = div
        self.gis = model
        return div*_log2

    def KSA(self):
        d = numpy.shape(self.m)
        self.kirk = numpy.ones(d,numpy.float)

        # normalize kirkwood approximation
        sumx = 0.0
        for x in xrange(d[0]):
            for y in xrange(d[1]):
                for z in xrange(d[2]):
                    pkirkwood  = self.pab[x,y]*self.pbc[y,z]*self.pac[x,z]
                    q = self.pa[x]*self.pb[y]*self.pc[z]
                    if q > 0:
                        pkirkwood /= self.pa[x]*self.pb[y]*self.pc[z]
                    else:
                        pkirkwood = 0.0
                    sumx += pkirkwood
                    self.kirk[x,y,z] = pkirkwood
        self.kirk /= sumx

        div = 0.0
        for x in xrange(d[0]):
            for y in xrange(d[1]):
                for z in xrange(d[2]):
                    if self.pm[x,y,z]>0:
                        div += self.pm[x,y,z]*numpy.log(self.pm[x,y,z]/self.kirk[x,y,z])
        return (div*_log2,sumx)

    def Divergence(self,x,y,z):
        ptrue = self.pm[x,y,z]
        pkirkwood  = self.pab[x,y]*self.pbc[y,z]*self.pac[x,z]
        pkirkwood /= self.pa[x]*self.pb[y]*self.pc[z]
        if ptrue > 1e-6:
            div = numpy.log(ptrue/pkirkwood)
        else:
            div = 0.0
        return (ptrue,pkirkwood,_log2*div)

    def NDivergence(self,x,y,z):
        norm = 1.0/self.NormDivergence()
        ptrue = self.pm[x,y,z]
        pkirkwood  = self.pab[x,y]*self.pbc[y,z]*self.pac[x,z]
        pkirkwood /= self.pa[x]*self.pb[y]*self.pc[z]
        if ptrue > 1e-6:
            div = numpy.log(ptrue/(pkirkwood*norm))
        else:
            div = 0.0
        return (ptrue,pkirkwood*norm,_log2*div)

    def __init__(self, m, names, values):
        self.names  = names
        self.values = values
        m = numpy.array(m,numpy.float)
        self.m = m
        self.bc = numpy.sum(m,axis=0)
        self.ac = numpy.sum(m,axis=1)
        self.ab = numpy.sum(m,axis=2)

        self.a = numpy.sum(self.ab,axis=1)
        self.b = numpy.sum(self.ab,axis=0)
        self.c = numpy.sum(self.ac,axis=0)
        self.total = numpy.sum(self.a)

        self.pm = Probabilities(self.m)
        self.pab = Probabilities(self.ab)
        self.pbc = Probabilities(self.bc)
        self.pac = Probabilities(self.ac)
        self.pa = Probabilities(self.a)
        self.pb = Probabilities(self.b)
        self.pc = Probabilities(self.c)
        dof = 0
        (ni,nj,nk) = numpy.shape(self.m)
        for ii in xrange(ni):
            for jj in xrange(nj):
                for kk in xrange(nk):
                    if self.m[ii,jj,kk] > 0:
                        dof += 1
        self.dof = dof-1
        return

class ContingencyTable2:
    def InteractionInformation(self):
        return Entropy(self.a)+Entropy(self.b)-Entropy(self.m)

    def JaccardInteraction(self):
        c = Entropy(self.m)
        if c > 0:
            return (Entropy(self.a)+Entropy(self.b)-c)/c
        else:
            return 0

    def Divergence(self,x,y):
        ptrue = self.pm[x,y]
        pkirkwood = self.pa[x]*self.pb[y]
        if ptrue > 1e-6:
            div = numpy.log(ptrue/pkirkwood)
        else:
            div = 0.0
        return (ptrue,pkirkwood,_log2*div)

    def ChiSquareP(self):
        E = numpy.outer(self.pa, self.pb) * self.total
        return statc.chisqprob(numpy.sum((E-self.m)**2 / E.clip(min=0.000001)), (len(self.pa)-1)*(len(self.pb)-1))
#        return numpy.sum((E-self.m)**2 / E)
      
    def Bootstrap(self,N,limit):
        # prepare lookup
        hits = 0
        nlimit = limit*_log2e
        f = Flatten(self.m)
        p = Probabilities(f)
        LUT = numpy.zeros((self.total,),numpy.int)
        c = 0
        for i in xrange(len(f)):
            for j in xrange(f[i]):
                LUT[c] = i
                c += 1
        assert(c == self.total)
        for i in xrange(N):
            nt = numpy.zeros((len(f),),numpy.float)
            for j in xrange(c):
                nt[ LUT[randint(0,c-1)] ] += 1
            q = Probabilities(nt)
            loss = 0.0
            for j in xrange(len(f)):
                if q[j] > 1e-6 and p[j] > 0.0:
                    loss += q[j]*numpy.log(q[j]/p[j])
                #loss += p[j]*numpy.log(max(p[j],1e-5)/max(q[j],1e-6))
            if loss >= nlimit:
                hits += 1
        return float(hits)/N

    def Name(self,x,y):
        s = "%s=%s,%s=%s"%(self.names[0],self.values[0][x],self.names[1],self.values[1][y])
        return s

    def __init__(self, m, names, values):
        self.names  = names
        self.values = values
        m = numpy.array(m,numpy.float)
        self.m = m

        self.a = numpy.sum(self.m,axis=1)
        self.b = numpy.sum(self.m,axis=0)
        self.total = numpy.sum(self.a)

        self.pa = Probabilities(self.a)
        self.pb = Probabilities(self.b)
        self.pm = Probabilities(self.m)
        dof = 0
        (ni,nj) = numpy.shape(self.m)
        for ii in xrange(ni):
            for jj in xrange(nj):
                if self.m[ii,jj] > 0:
                    dof += 1
        self.dof = dof-1 # degrees of freedom is the number of fields with non-zero counts
        return

def get3Int(t,a,b,c,wid=None,prior=0):
    ni = len(a.values)
    nj = len(b.values)
    nk = len(c.values)
    M = prior*numpy.ones((ni,nj,nk),numpy.float)
    if wid == None: # no weighting
        for ex in t:
            if not (ex[a].isSpecial() or ex[b].isSpecial() or ex[c].isSpecial()):
                M[int(ex[a]),int(ex[b]),int(ex[c])] += 1
    else:
        for ex in t:
            if not (ex[a].isSpecial() or ex[b].isSpecial() or ex[c].isSpecial()):
                M[int(ex[a]),int(ex[b]),int(ex[c])] += ex.getmeta(wid)
    N = [a.name,b.name,c.name]
    V = [[a.values[k] for k in range(ni)],[b.values[k] for k in range(nj)],[c.values[k] for k in range(nk)]]
    c = ContingencyTable3(M,N,V)
    return c

def get2Int(t,a,b,wid=None,prior=0):
    ni = len(a.values)
    nj = len(b.values)
    M = prior*numpy.ones((ni,nj),numpy.float)
    if wid == None: # no weighting
        for ex in t:
            if not (ex[a].isSpecial() or ex[b].isSpecial()):
                M[int(ex[a]),int(ex[b])] += 1
    else:
        for ex in t:
            if not (ex[a].isSpecial() or ex[b].isSpecial()):
                M[int(ex[a]),int(ex[b])] += ex.getmeta(wid)
    N = [a.name,b.name]
    V = [[a.values[k] for k in range(ni)],[b.values[k] for k in range(nj)]]
    c = ContingencyTable2(M,N,V)
    return c

def getPvalue(lim,table):
    # import statistics
    # return 1-statisticsc.chi_squared(table.dof, 2.0*lim*table.total*_log2e)
    import statc
#    print 2.0*lim*table.total*_log2e, table.dof
    if 2.0*lim*table.total*_log2e <= 0.0:
        return 1.0 - statc.chisqprob(1.e-10, table.dof)
    return 1.0 - statc.chisqprob(2.0*lim*table.total*_log2e, table.dof)

def getPvalueDOF(lim,table,dof):
    # import statisticsc
    # return 1-statisticsc.chi_squared(dof,2.0*lim*table.total*_log2e)
    import statc
    return 1.0 - statc.chisqprob(2.0*lim*table.total*_log2e, dof)
