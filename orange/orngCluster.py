# ORANGE CLUSTERING
#    by Alex Jakulin (jakulin@acm.org)
#
#       based on Struyf, Hubert, Rousseeuw:
#                Integrating Robust Clustering Techniques in S-PLUS
#                Computational Statistics and Data Analysis, 26, 17-37
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Version 1.7 (11/10/2002)

import orngCRS

class HClustering:
# in merging pairs occur. negative numbers identify elements in the order
# list, positive the consecutive number (starting from 1) of previous merges.
# Height refers to merges
#
# Metric: 1- euclidean
#         2- manhattan (default)
#
# Methods: 1- average
#          2- single
#          3- complete
#          4- ward
#          5-weighted
#
        def extract_graph(self,data,filename = 'graph.dot'):
                names = []
                for i in data.domain.attributes:
                        t = string.replace(i.name," ","_")
                        names.append(string.replace(i.name,"-","_"))
                f = open(filename,'w')
                f.write("digraph G {\n")
                j = 0
                for i in self.merging:
                        l = "%d"%j
                        j = j+1
                        n = "%d"%j
                        if i[0]<0:
                                a = names[-1-i[0]]
                        else:
                                a = "%d"%(i[0])
                        if i[1]<0:
                                b = names[-1-i[1]]
                        else:
                                b = "%d"%(i[1])
                        f.write("\t%s[ label = \"%s\"];\n"%(n,l))
                        f.write("\t%s -> %s;\n"%(a,n))
                        f.write("\t%s -> %s;\n"%(b,n))
                f.write("}\n")

        def extract_subgroups(self, n):
                # create n best bound sets
                bounds = []
                merges = []
                for i in range(self.n):
                    #merges.append([self.order[self.n-i-1]-1])
                    merges.append([self.n-i-1])
                merges.append("sentry")
                p = self.n
                for i in range(min(n,self.n-1)):
                        merges.append(merges[p+self.merging[i][0]]+merges[p+self.merging[i][1]])
                        bounds.append(merges[-1])
                return bounds

        def extract_hierarchy(self):
                #
                # export the hierarchy in a format understood by induce()
                #
                bounds = []
                merges = []
                for i in range(self.n):
                    merges.append(self.n-i-1)
                merges.append("sentry")
                p = self.n
                c = 0
                for i in range(self.n-1):
                        merges.append(['%d-%.2f'%(c,self.height[i]),merges[p+self.merging[i][0]],merges[p+self.merging[i][1]]])
                        c = c+1
                return merges[-1]
                

        def domapping(self,kk):
                height = 0.0
                merges = []
                for i in range(self.n):
                    #merges.append([self.order[self.n-i-1]-1])
                    merges.append([self.n-i-1])
                merges.append("sentry")
                p = self.n
                deletes = [p]
                for i in range(self.n-kk):
                        height += self.height[i]
                        merges.append(merges[p+self.merging[i][0]]+merges[p+self.merging[i][1]])
                        deletes.append(p+self.merging[i][0])
                        deletes.append(p+self.merging[i][1])
                p = 0
                deletes.sort()
                for i in deletes:
                    del merges[p+i]
                    p = p-1
                self.mapping = [0]*self.n
                for i in range(len(merges)):
                    for j in merges[i]:
                        self.mapping[j] = i+1
                #return height
                    
        def __init__(self, distrlist, metric=2, method=4):
                assert(metric == 1 or metric == 2)
                assert(method >= 1 or method <= 5)
                if len(distrlist) > 1:
                        (a,b,c,d,e) = orngCRS.HCluster(distrlist,metric, method)
                        self.n = a
                        self.merging = []
                        for i in range(self.n-1):
                            self.merging.append([b[i],b[i+self.n-1]])
                        self.order = c
                        self.height = d
                        self.ac = e
                        self.mapping = []
                        self.values = distrlist
                else:
                        if len(distrlist) == 1:
                                self.n = 1
                                self.merging = []
                                self.order = [1]
                                self.ac = 0.0
                                self.mapping = []
                                self.values = [[1]]
                                self.height = [1.0]
                        else:
                                self.n = 0
                                self.merging = []
                                self.order = []
                                self.ac = 0.0
                                self.mapping = []
                                self.values = []
                                self.height = []
                
class MClustering:
#
# Metric: 1- euclidean
#         2- manhattan (default)
#
    def __init__(self, distrlist,k,metric=2):
        assert(metric == 1 or metric == 2)
        if (len(distrlist) > k):
                (a,b,c,d,e,f) = orngCRS.MCluster(distrlist,k,metric)
                self.n = a
                self.k = b
                self.mapping = c
                self.medoids = d
                self.cdisp = e
                self.disp = f
        else:
                self.n = len(distrlist)
                self.k = self.n
                self.mapping = range(1,self.n+1)
                self.medoids = range(1,self.n+1)
                self.cdisp = [0]*self.n
                self.disp = 0
                self.values = distrlist

class FClustering:
#
# Metric: 1- euclidean
#         2- manhattan (default)
#
    def __init__(self, distrlist,k,metric=2):
      assert(metric == 1 or metric == 2)
      if not(k >= 1):
        raise "FClustering: Negative or zero number of clusters"
      if not(k < len(distrlist)/2):
        raise "FClustering: Not enough data for this number of clusters"
      (a,b,c,d,e,f,g,h) = orngCRS.FCluster(distrlist,k,metric)
      self.n = a
      self.k = b
      self.value = c
      self.iterations = d
      self.membership = e
      self.mapping = f
      self.cdisp = g
      self.disp = h

class DHClustering(HClustering):
# in merging pairs occur. negative numbers identify elements in the order
# list, positive the consecutive number (starting from 1) of previous merges.
# Height refers to merges
#
# The diss should contain a dissimilarity matrix, 
#
# Methods: 1- average
#          2- single
#          3- complete
#          4- ward
#          5-weighted
#
    def __init__(self, diss, method=4):
        assert(method >= 1 or method <= 5)
        if len(diss) < 1:
                self.n = 1
                self.merging = []
                self.order = [1]
                self.ac = 0.0
                self.mapping = []
                self.values = [[1]]
                self.height = [1.0]
        else:
                (a,b,c,d,e) = orngCRS.DHCluster(diss, method)
                self.n = a
                self.merging = []
                for i in range(self.n-1):
                    self.merging.append([b[i],b[i+self.n-1]])
                self.order = c
                self.height = d
                self.ac = e
                self.mapping = []
        
class DMClustering:
#
#
    def __init__(self, diss,k):
        if len(diss) <= k-1: #len = 1 if two elements ---- then k can be 2 or less
                self.n = len(distrlist)+1
                self.k = self.n
                self.mapping = range(1,self.n+1)
                self.medoids = range(1,self.n+1)
                self.cdisp = [0]*self.n
                self.disp = 0
                self.values = distrlist
        else:
                (a,b,c,d,e,f) = orngCRS.DMCluster(diss,k)
                self.n = a
                self.k = b
                self.mapping = c
                self.medoids = d 
                self.cdisp = e
                self.disp = f


class DFClustering:
    def __init__(self, diss,k):
        if not(k >= 1):
          raise "DFClustering: Negative or zero number of clusters"
        if not(k <= len(diss)/2):
          raise "DFClustering: Not enough data for this number of clusters"
        (a,b,c,d,e,f,g,h) = orngCRS.DFCluster(diss,k)
        self.n = a
        self.k = b
        self.value = c
        self.iterations = d
        self.membership = e
        self.mapping = f
        self.cdisp = g
        self.disp = h
