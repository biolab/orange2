#
# Module Orange Dependencies
# --------------------------
#
# CVS Status: $Id$
#
# Author: Aleks Jakulin (jakulin@acm.org)
#
# Purpose: Analysis of dependencies between attributes.
#          2-WAY INTERACTIONS
#
# Project initiated on 2003/05/13
#
# ChangeLog:
#   - 2003/05/19:
#       Maximum spanning tree algorithm.
#   - 2003/09/14:
#       Friedman's TAN constructor
#   - 2003/09/24:
#       Dissimilarity matrix output

import orange, orngCI
from orngInteract import *
from orngInteract import _nicefloat
import math

class _Item:
    def __init__(self):
        self.t = 0
        self.d = {}

    def add(self,values):
        self.t += 1
        if self.d.has_key(values):
            self.d[values] += 1
        else:
            self.d[values] = 1
        
    def entropy(self):
        e = 0.0
        if self.t == 0:
            return 0.0
        n = float(self.t)
        for i in self.d.values():
            p = i/n
            e -= p*math.log(p)/math.log(2)
        return e
        

class _StatsGatherer:
    def __init__(self,set):
        self.ones = []
        self.twos = []
        self.n = len(set)
        self.names = set
        # attributes
        for i in range(self.n):
            self.ones.append(_Item())
            for j in range(i):
                self.twos.append(_Item())
                    
    def add(self,v):
        c1 = 0
        c2 = 0
        # attributes
        for i in range(self.n):
            self.ones[c1].add((v[i]))
            c1 += 1
            for j in range(i):
                self.twos[c2].add((v[i],v[j]))
                c2 += 1

    def prin(self):
        # get entropies
        ent = {}
        c1 = 0
        c2 = 0
        # attributes
        for i in range(self.n):
            ent[(i)]=self.ones[c1].entropy()
            c1 += 1
            for j in range(i):
                ent[(i,j)]=self.twos[c2].entropy()
                ent[(j,i)]=ent[(i,j)] # commutativity
                c2 += 1
        return ent

class DependenceMatrix(InteractionMatrix):
    def __init__(self, t, include_class=1, interactions_too = 1):
        t = self._prepare(t)

        NA = len(t.domain.attributes)

        self.names = []
        for i in range(NA):
            st = '%s'%t.domain.attributes[i].name # copy
            self.names.append(st)

        if include_class:
            self.names.append('%s'%t.domain.classVar.name)
            NA += 1

        stats = _StatsGatherer(self.names)
        for ex in t:
            # convert attribute values into numbers
            stats.add([int(ex[i]) for i in range(NA)])
        self.ents = stats.prin() # entropy look-up

        if interactions_too:
            self.gains = []
            for i in t.domain.attributes:
                self.gains.append(orange.MeasureAttribute_info(i,t))

            abc = orngCI.FeatureByCartesianProduct()
            self.igain = {}
            self.corr = {}
            for i in range(1,NA):
                for j in range(i):
                    if i < len(t.domain.attributes):
                        (cart, profit) = abc(t,[t.domain.attributes[i],t.domain.attributes[j]])
                        scdomain = orange.Domain([cart,t.domain.classVar])
                        sctrain = t.select(scdomain)
                        ci = orange.MeasureAttribute_info(cart,sctrain)
                        self.igain[(i,j)] = ci-self.gains[i]-self.gains[j]
                    self.corr[(i,j)] = self.ents[(i)]+self.ents[(j)]-self.ents[(i,j)]

    def dump(self):
        NA = len(self.names)
        for i in range(1,NA):
            for j in range(i):
                t = '%s+%s'%(self.names[i],self.names[j])
                print "%30s\t%2.4f\t%2.4f\t%2.4f\t%2.4f\t%2.4f"%(t,self.igain[(i,j)],self.corr[(i,j)],self.igain[(i,j)]+self.corr[(i,j)],self.gains[i],self.gains[j])

    def exportGraph(self, f, n_int=1, print_bits = 1, black_white = 0, significant_digits = 2, pretty_names = 1, postscript=1, spanning_tree = 1, TAN=1):
        NA = len(self.names)

        ### SELECTION OF INTERACTIONS AND ATTRIBUTES ###

        # prevent crashes
        n_int = min(n_int,NA)

        links = []
        maxlink = -1e6
        if n_int == 1 and spanning_tree:
            # prepare table
            lmm = []
            for i in range(1,NA):
                ei = self.ents[(i)]
                for j in range(i):
                    ej = self.ents[(j)]
                    if TAN:
                        v = self.igain[(i,j)]+self.corr[(i,j)]
                    else:
                        v = ei+ej-self.ents[(i,j)] # chow-liu
                    if ei > ej:
                        lmm.append((abs(v),v,ej,(j,i)))
                    else:
                        lmm.append((abs(v),v,ei,(i,j)))
            lmm.sort()
            maxlink = lmm[-1][0]
            # use Prim's algorithm here
            mapped = []
            for i in range(NA):
                mapped.append(i)
            n = NA
            idx = -1 # running index in the sorted array of possible links
            while n > 1:
                # find the cheapest link
                while 1:
                    (av,v,e,(i,j)) = lmm[idx]
                    idx -= 1
                    if mapped[i] != mapped[j]:
                        break
                links.append((v,(i,j),e))
                toremove = mapped[j]
                for k in range(NA):
                    if mapped[k] == toremove:
                        mapped[k] = mapped[i]
                n -= 1
        else:
            # select the top
            for i in range(NA):
                e = self.ents[(i)]
                if e > 0.0:
                    lmm = []
                    for j in range(NA):
                        if i != j:
                            lmm.append((self.ents[(j)]+e-self.ents[(i,j)],(i,j)))
                    lmm.sort()
                    maxlink = max(lmm[-1][0],maxlink)
                    links += [(v,p,e) for (v,p) in lmm[-n_int:]]

        # output the attributes
        f.write("digraph G {\n")

        if print_bits:
            shap = 'record'
        else:
            shap = 'box'

        for n in range(NA):
            t = '%s'%self.names[n]
            if pretty_names:
                t = string.replace(t,"ED_","")
                t = string.replace(t,"D_","")
                t = string.replace(t,"M_","")
                t = string.replace(t," ","\\n")
                t = string.replace(t,"-","\\n")
                t = string.replace(t,"_","\\n")
            if print_bits:
                t = "{%s|%s}"%(t,_nicefloat(self.ents[(n)],significant_digits))
            f.write("\tnode [ shape=%s, label = \"%s\"] %d;\n"%(shap,t,n))
            
        ### EDGE DRAWING ###
        for (v,(i,j),e) in links:
            if v > 0:
                c = v/e
                perc = int(100*v/maxlink + 0.5)

                mc = _nicefloat(100.0*c,significant_digits)
                style = ''
                if postscript:
                    style += "style=\"setlinewidth(%d)\","%(abs(perc)/30+1)
                if not black_white:
                    l = 0.3+0.7*perc/100.0
                    style += 'color="0.5 %f %f",'%(l,1-l) # adjust saturation
                f.write("\t%d -> %d [%slabel=\"%s%%\",weight=%d];\n"%(j,i,style,mc,(perc/30+1)))
        f.write("}\n")

    def exportDissimilarityMatrix(self, truncation = 1000, pretty_names = 1, normalization = 1):
        NA = len(self.names)

        ### BEAUTIFY THE LABELS ###

        labels = []
        for i in range(NA):
            t = '%s'%self.names[i]
            if pretty_names:
                t = string.replace(t,"ED_","")
                t = string.replace(t,"D_","")
                t = string.replace(t,"M_","")
            labels.append(t)

        ### CREATE THE DISSIMILARITY MATRIX ###

        diss = []        
        for x in range(1,NA):
            newl = []
            for y in range(x):
                t = self.corr[(x,y)]
                if normalization:
                    t /= self.ents[(x)]+self.ents[(y)]
                if t*truncation > 1:
                    t = 1.0 / t
                else:
                    t = truncation
                newl.append(t)
            diss.append(newl)
        return (diss, labels)



if __name__== "__main__":
    import orange

    n = 'd_cmc'
    t = orange.ExampleTable('%s.tab'%n)
    im = DependenceMatrix(t,include_class=0)
    
    f = open('%s_d.dot'%n,'w')
    im.exportGraph(f,n_int=1,significant_digits=3)
    f.close()
