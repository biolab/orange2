#
# Module Orange Interactions
# --------------------------
#
# CVS Status: $Id$
#
# Author: Aleks Jakulin (jakulin@acm.org)
#
# Purpose: Analysis of dependencies between attributes given the class.
#          3-WAY INTERACTIONS
#
# Project initiated on 2003/05/08
#
# ChangeLog:
#   - 2003/05/09:
#       fixed a problem with domains that need no preprocessing
#       fixed the decimal point printing problem
#       added the support for dissimilarity matrix, used for attribute clustering
#   - 2003/05/10:
#       fixed a problem with negative percentages of less than a percent
#   - 2003/05/12:
#       separated the 'prepare' function
#   - 2003/09/18:
#       added support for cluster coloring
#       cleaned up backwards-incompatible changes (grrr) (color changes, discData)
#       added color-coded dissimilarity matrix export

import orange, orngCI
import warnings, math, string

def _nicefloat(f,sig):
    # pretty-float formatter
    i = int(f)
    s = '%1.0f'%f
    n = sig-len('%d'%abs(f)) # how many digits is the integer part
    if n > 0:
        # we can put a few decimals at the end
        fp = abs(f)-abs(i)
        s = ''
        if f < 0:
            s += '-'
        s += '%d'%abs(i) + ('%f'%fp)[1:2+n]
        
    return s

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


class InteractionMatrix:
    def _prepare(self, t):
        # prepares an Orange table so that it doesn't contain continuous
        # attributes or missing values

        ### DISCRETIZE VARIABLES ###
        
        newatt = []
        oldatt = []
        entroD = orange.EntropyDiscretization()
        equiD = orange.EquiNDiscretization(numberOfIntervals = 2)
        for i in t.domain.attributes:
            if i.varType == 2:
                d = entroD(i,t)
                if len(d.values) < 2:
                    # prevent discretization into a single value
                    d = equiD(i,t)
                    d.name = 'E'+d.name
                warnings.warn('Discretizing %s into %s with %d values.'%(i.name,d.name,len(d.values)))
                newatt.append(d)
            else:
                oldatt.append(i)
        if len(newatt) > 0:
            t = t.select(oldatt+newatt+[t.domain.classVar])
        
        ### FIX MISSING VALUES ###
            
        special_attributes = []
        all_attributes = [i for i in t.domain.attributes]+[t.domain.classVar]
        for i in range(len(all_attributes)):
            for j in t:
                if j[i].isSpecial():
                    special_attributes.append(i)
                    break
        # create new attributes
        if len(special_attributes) > 0:
            # prepare attributes
            newatts = []
            for i in range(len(all_attributes)):
                old = all_attributes[i]
                if i in special_attributes:
                    oldv = [v for v in old.values]
                    assert('.' not in oldv)
                    new = orange.EnumVariable(name='M_'+old.name, values=oldv+['.'])
                    warnings.warn('Removing special values from %s into %s.'%(old.name,new.name))
                    newatts.append(new)
                else:
                    newatts.append(old)
            # convert table
            exs = []
            newd = orange.Domain(newatts)
            for ex in t:
                nex = []
                for i in range(len(newatts)):
                    if ex[i].isSpecial():
                        v = newatts[i]('.')
                    else:
                        v = newatts[i](int(ex[i]))
                    nex.append(v)
                exs.append(orange.Example(newd,nex))
            t = orange.ExampleTable(exs)
        return t
        
    def __init__(self, t, save_data=1, dependencies_too=0):
        t = self._prepare(t)
        if save_data:
            self.discData = t   # save the discretized data

        ### PREPARE INDIVIDUAL ATTRIBUTES ###

        # Get the class entropy
        l = orange.MajorityLearner(t)
        p = l(t[0],orange.GetProbabilities)
        self.entropy = 0.0
        for i in p:
            if i > 1e-6:
                self.entropy -= i*math.log(i)/math.log(2)

        # Attribute Preparation
        NA = len(t.domain.attributes)
        
        self.names = []
        self.gains = []
        for i in range(NA):
            self.gains.append(orange.MeasureAttribute_info(t.domain.attributes[i],t))
            # fix the name
            st = '%s'%t.domain.attributes[i].name # copy
            self.names.append(st)

        if dependencies_too:
            stats = _StatsGatherer(self.names+[t.domain.classVar.name])
            for ex in t:
                # convert attribute values into numbers
                stats.add([int(ex[i]) for i in range(NA+1)])
            self.ents = stats.prin() # entropy look-up

        ### COMPUTE INTERACTION GAINS ###

        abc = orngCI.FeatureByCartesianProduct()
        self.ig = []
        self.list = []
        self.abslist = []
        for i in range(1,NA):
            line = []
            for j in range(i):
                # create Cartesian attribute
                (cart, profit) = abc(t,[t.domain.attributes[i],t.domain.attributes[j]])
                scdomain = orange.Domain([cart,t.domain.classVar])
                sctrain = t.select(scdomain)
                ci = orange.MeasureAttribute_info(cart,sctrain)
                igv = ci-self.gains[i]-self.gains[j]
                line.append(igv)
                self.list.append((igv,(igv,i,j)))
                self.abslist.append((abs(igv),(igv,i,j)))
            self.ig.append(line)
        self.list.sort()
        self.abslist.sort()

        self.attlist = []
        for i in range(NA):
            self.attlist.append((self.gains[i],i))
        self.attlist.sort()

    def exportGraph(self, f, absolute_int=10, positive_int = 0, negative_int = 0, best_attributes = 0, print_bits = 1, black_white = 0, significant_digits = 2, postscript = 1, pretty_names = 1, url = 0, widget_coloring=1):
        NA = len(self.names)

        ### SELECTION OF INTERACTIONS AND ATTRIBUTES ###

        # prevent crashes
        best_attributes = min(best_attributes,len(self.attlist))
        positive_int = min(positive_int,len(self.list))
        absolute_int = min(absolute_int,len(self.list))
        negative_int = min(negative_int,len(self.list))
        
        # select the top interactions
        ins = []
        if positive_int > 0:
            ins += self.list[-positive_int:]
        ins += self.list[:negative_int]
        if absolute_int > 0:
            ins += self.abslist[-absolute_int:]

        # pick best few attributes
        atts = []
        if best_attributes > 0:
            atts += [i for (x,i) in self.attlist[-best_attributes:]]

        ints = []
        max_igain = -1e6
        min_gain = 1e6 # lowest information gain of involved attributes
        # remove duplicates and sorting keys
        for (x,v) in ins:
            if v not in ints:
                ints.append(v)
                # add to attribute list
                (ig,i,j) = v
                max_igain = max(abs(ig),max_igain)
                for x in [i,j]:
                    if x not in atts:
                        atts.append(x)
                        min_gain = min(min_gain,self.gains[x])

        # fill-in the attribute list with all possibly more important attributes
        ## todo

        ### NODE DRAWING ###

        # output the attributes
        f.write("digraph G {\n")

        if print_bits:
            shap = 'record'
        else:
            shap = 'box'

        for i in atts:
            t = '%s'%self.names[i]
            if pretty_names:
                t = string.replace(t,"ED_","")
                t = string.replace(t,"D_","")
                t = string.replace(t,"M_","")
                t = string.replace(t," ","\\n")
                t = string.replace(t,"-","\\n")
                t = string.replace(t,"_","\\n")
            if print_bits:
                r = self.gains[i]*100.0/self.entropy
                t = "{%s|%s%%}"%(t,_nicefloat(r,significant_digits))
            if not url:
                f.write("\tnode [ shape=%s, label = \"%s\"] %d;\n"%(shap,t,i))
            else:
                f.write("\tnode [ shape=%s, URL = \"%d\", label = \"%s\"] %d;\n"%(shap,i,t,i))
            
        ### EDGE DRAWING ###

        for (ig,i,j) in ints:
            perc = int(abs(ig)*100.0/max(max_igain,self.attlist[-1][0])+0.5)

            if self.entropy > 1e-6:
                mc = _nicefloat(100.0*ig/self.entropy,significant_digits)
            else:
                mc = _nicefloat(0.0,significant_digits)                
            if postscript:
                style = "style=\"setlinewidth(%d)\","%(abs(perc)/30+1)
            else:
                style = ''
            if black_white:
                color = 'black'
                if ig > 0:
                    dir = "both"
                else:
                    style = 'style=dashed,'
                    dir = 'none'
            else:            
                if ig > 0:
                    if widget_coloring:
                        color = "green"
                    else:
                        color = '"0.0 %f 0.9"'%(0.3+0.7*perc/100.0) # adjust saturation
                    dir = "both"
                else:
                    if widget_coloring:
                        color = "red"
                    else:
                        color = '"0.5 %f 0.9"'%(0.3+0.7*perc/100.0) # adjust saturation
                    dir = 'none'
            if not url:
                f.write("\t%d -> %d [dir=%s,%scolor=%s,label=\"%s%%\",weight=%d];\n"%(i,j,dir,style,color,mc,(perc/30+1)))
            else:
                f.write("\t%d -> %d [URL=\"%d-%d\",dir=%s,%scolor=%s,label=\"%s%%\",weight=%d];\n"%(i,j,min(i,j),max(i,j),dir,style,color,mc,(perc/30+1)))

        f.write("}\n")
        
    def exportDissimilarityMatrix(self, truncation = 1000, pretty_names = 1, print_bits = 0, significant_digits = 2, show_gains = 1, color_coding = 0, color_gains = 0, jaccard=0):
        NA = len(self.names)

        ### BEAUTIFY THE LABELS ###

        labels = []
        maxgain = max(self.gains)
        for i in range(NA):
            t = '%s'%self.names[i]
            if pretty_names:
                t = string.replace(t,"ED_","")
                t = string.replace(t,"D_","")
                t = string.replace(t,"M_","")
            r = self.gains[i]
            if print_bits:
                if self.entropy > 1e-6:
                    t = "%s (%s%%)"%(t,_nicefloat(r*100.0/self.entropy,significant_digits))
                else:
                    t = "%s (0%%)"%(t)
            if show_gains: # a bar indicating the feature importance
                if maxgain > 1e-6:
                    t += ' '+'*'*int(8.0*r/maxgain+0.5)
            labels.append(t)

        ### CREATE THE DISSIMILARITY MATRIX ###
        
        if jaccard:
            # create the lookup of 3-entropies
            ent3 = {}
            maxx = 1e-6
            for i in range(1,NA):
                for j in range(i):
                    e = self.ents[(i,j)]+self.ents[(j,NA)]+self.ents[(i,NA)]
                    e -= self.ents[(i)]+self.ents[(j)]+self.ents[(NA)]
                    e -= self.ig[i-1][j]
                    ent3[(i,j)] = e
                    if e > 1e-6:
                        e = abs(self.ig[i-1][j])/e
                    else:
                        e = 0.0
                    maxx = max(maxx,e)
            # check the information gains...
            if color_gains:
                for i in range(NA):
                    e = self.gains[i]
                    if self.ents[(i,NA)] > 1e-6:
                        e /= self.ents[(i,NA)]
                    else:
                        e = 0.0
                    ent3[(i)] = e 
                    maxx = max(maxx,e)
        else:
            maxx = self.abslist[-1][0]
            if color_gains:
                maxx = max(maxx,self.attlist[-1][0])
        if color_gains:
            if maxx > 1e-6:
                cgains = [0.5*(1-i/maxx) for i in self.gains]
            else:
                cgains = [0.0 for i in self.gains]                
        diss = []        
        for i in range(1,NA):
            newl = []
            for j in range(i):
                d = self.ig[i-1][j]
                if jaccard:
                    if ent3[(i,j)] > 1e-6:
                        d /= ent3[(i,j)]
                    else:
                        d = 0.0
                if color_coding:
                    if maxx > 1e-6:
                        if maxx > 1e-6:
                            t = 0.5*(1-d/maxx)
                        else:
                            t = 0.0
                    else:
                        t = 0
                else:
                    # transform the IG into a distance
                    ad = abs(d)
                    if ad*truncation > 1:
                        t = 1.0 / ad
                    else:
                        t = truncation
                newl.append(t)
            diss.append(newl)

        if color_gains:
            return (diss,labels,cgains)
        else:
            return (diss,labels)

    def getClusterAverages(self, clust):
        assert(len(self.attlist) == clust.n)
        # get the max value
        #d = max(self.attlist[-1][0],self.abslist[-1][0])
        d = self.abslist[-1][0]
        # prepare a lookup
        LUT = {}
        for (ig,(igv,i,j)) in self.list:
            LUT[i,j] = igv
            LUT[j,i] = igv

        cols = []
        merges = []
        for i in range(clust.n):
            merges.append((0.0,[clust.n-i-1]))
        merges.append("sentry")
        p = clust.n
        for i in range(clust.n-1):
            a = merges[p+clust.merging[i][0]] # cluster 1
            b = merges[p+clust.merging[i][1]] # cluster 2
            na = len(a[1])
            nb = len(b[1])
            # compute cross-average
            sum = 0.0
            for x in a[1]:
                for y in b[1]:
                    sum += LUT[x,y]
            avg = (a[0]*(na*na-na) + b[0]*(nb*nb-nb) + 2*sum)/(math.pow(na+nb,2)-na-nb)
            clustercolor = 0.5*(1-avg/d)
            intercluster = 0.5*(1-sum/(d*na*nb))
            cols.append((clustercolor,intercluster)) # positive -> red, negative -> blue
            merges.append((avg,a[1]+b[1]))
        return cols

if __name__== "__main__":
    t = orange.ExampleTable('zoo.tab')
    im = InteractionMatrix(t)
    
    # interaction graph    
    f = open('zoo.dot','w')
    im.exportGraph(f,significant_digits=3)
    f.close()

    # interaction clustering
    import orngCluster
    (diss,labels) = im.exportDissimilarityMatrix()
    c = orngCluster.DHClustering(diss)
    NCLUSTERS = 6
    c.domapping(NCLUSTERS)
    print "Clusters:"
    for j in range(1,NCLUSTERS+1):
        print "%d: "%j,
        # print labels of that cluster
        for i in range(len(labels)):
            if c.mapping[i] == j:
                print labels[i],
        print