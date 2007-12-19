import orange
from math import log,exp
from urllib import urlopen
import os.path

class orngMeSH(object):
    def __init__(self):
        self.path = "data"
        self.reference = None
        self.cluster = None
        self.ratio = 1
        self.statistics = None
        self.calculated = False
        
        self.ref_att = "Unknown"
        self.clu_att = "Unknown"
        self.solo_att = "Unknown"
		
        #we calculate log(i!)
        self.lookup = [0]
        for i in range(1, 8000):
            self.lookup.append(self.lookup[-1] + log(i))		
        self.dataLoaded = self.__loadOntologyFromDisk()		

    def setDataDir(self,dataDir):
        self.path = dataDir
        self.dataLoaded = self.dataLoaded or self.__loadOntologyFromDisk()
 
    def getDataDir(self):
        """Default for dataDir is "data", by calling these two methods the user can change the directory of the local "fast" data base.
        This influences downloadGO() and downloadAnnotation(...). above directory also buffers compound annotation"""
        return self.path

    def downloadOntology(self,callback=None):
        # ftp://nlmpubs.nlm.nih.gov/online/mesh/.meshtrees/mtrees2008.bin
        # ftp://nlmpubs.nlm.nih.gov/online/mesh/.asciimesh/d2008.bin
        ontology = urlopen("ftp://nlmpubs.nlm.nih.gov/online/mesh/.asciimesh/d2008.bin")
        size = int(ontology.info().getheader("Content-Length"))
        rsize = size

        results = list()

        for i in ontology:
            rsize -= len(i)
            line = i.rstrip("\t\n")
            if(line == "*NEWRECORD"):
                if(len(results) > 0 and results[-1][1] == []):				# we skip nodes with missing mesh id
                    results[-1] = ["",[],"No description."]
                else:
                    results.append(["",[],"No description."])	
                if(len(results)%400 == 0):
                    if not callback:
                        callback(int(rsize*100/size))
                    print "remaining " + str(rsize*100/size) + "%."			
	
            parts = line.split(" = ")
            if(len(parts) == 2 and len(results)>0):
                if(parts[0] == "MH"):
                    results[-1][0] = parts[1].strip("\t ") 					

                if(parts[0] == "MN"):
                    results[-1][1].append(parts[1].strip("\t "))

                if(parts[0] == "MS"):
                    results[-1][2] = parts[1].strip("\t ")

        ontology.close()
		
        __dataPath = os.path.join(os.path.dirname(__file__), self.path)
        output = file(os.path.join(__dataPath,'mesh-ontology.dat'), 'w')

        for i in results:
            #			print i[0] + "\t"
            output.write(i[0] + "\t")
            g=len(i[1])			
            for k in i[1]:
                g -= 1
                if(g > 0):
                    #					print k + ";"
                    output.write(k + ";")
                else:
                    #					print k + "\t" + i[2]			
                    output.write(k + "\t" + i[2] + "\n")

        output.close()
        self.__loadOntologyFromDisk()
        print "Ontology database has been updated."

    def findSubset(self,examples,meshTerms, callback = None):
        """ function examples which have at least one node on their path from list meshTerms
            findSubset(all,['Aspirine']) will return a dataset with examples are annotated as Aspirine """
        # clone        
        newdata = orange.ExampleTable(examples.domain)
        self.solo_att = self.__findMeshAttribute(examples)
        ids = list()
        
        # we couldn't find any mesh attribute
        if self.solo_att == "Unknown":
            return newdata
        
        for i in meshTerms:
            ids.extend(self.toID[i])

        for e in examples:
            try:
                ends = eval(e[self.solo_att].value)
            except SyntaxError:
                print "Error in parsing ", e[self.solo_att].value
                continue
            endids = list()
            for i in ends:
                if self.toID.has_key(i):
                    endids.extend(self.toID[i])
            allnodes = self.__findParents(endids)
            
            # calculate intersection            
            isOk = False            
            for i in allnodes:
                if ids.count(i) > 0:
                    isOk = True
                    break

            if isOk:      # intersection between example mesh terms and observed term group is None
                newdata.append(e)

        return newdata	

#    def findTerms(self,CIDs):
#        """ returns a nested list (?) or a graph (?) or better a dictionary (?) with terms that apply to CIDs.
#        term in this structure could probably be an object that holds its name, id, description(?) or alternatively just
#        an termID (probably better). If the later, than dictionary should be available for mapping of termID to name (termID2name)
#        and termID to description (termID2description), and also for mapping of name to ID (name2termID). """

        # at the moment it returns just a list of mesh terms
#        ret = dict()
#        if(not self.dataLoaded):
#            print "Annotation and ontology has never been loaded! Use function setDataDir(path) to fix the problem."
#            return ret

#        for i in CIDs:
#            if(self.fromCID.has_key(i)):
#                ret[i] = self.fromCID[i]
#        return ret
#
    
    def findFrequentTerms(self,data,minSizeInTerm, treeData = False, callback=False):
        """ Function iterates thru examples in data. For each example it computes a list of associated terms. At the end we get (for each term) number of examples which have this term. """
        # we build a dictionary 		meshID -> [description, noReference, [cids] ]
        self.statistics = dict()
        self.calculated = False
        self.solo_att = self.__findMeshAttribute(data)

        # post processing variables
        ret = dict()
        ids = []
        succesors = dict()		# for each term id -> list of succesors
        succesors["tops"] = []
        
        # if we can't identify mesh attribute we return empty data structures
        if self.solo_att == "Unknown":
            if treeData:
                return succesors, tops
            else:
                return ret

        # plain frequency
        for i in data:
            try:
                endNodes = eval(i[self.solo_att].value)	# for every CID we look up for end nodes in mesh. for every end node we have to find its ID	
            except SyntaxError:
                print "Error in parsing ",i[self.solo_att].value
                continue
            # we find ID of end nodes
            endIDs = []
            for k in endNodes:
                if(self.toID.has_key(k)):					# this should be always true, but anyway ...
                    endIDs.extend(self.toID[k])
                else:
                    print "Current ontology does not contain MeSH term ", k, "." 

            # we find id of all parents
            allIDs = self.__findParents(endIDs)
			
            for k in allIDs:						        # for every meshID we update statistics dictionary
                if(not self.statistics.has_key(k)):		    # first time meshID
                    self.statistics[k] = [self.toDesc[self.toName[k]], 0, [] ]
                self.statistics[k][1] += 1				    # counter increased

        # post processing
        for i in self.statistics.iterkeys():
            if(self.statistics[i][1] >= minSizeInTerm ) : 
                ret[i] = [self.statistics[i][0],self.statistics[i][1],self.statistics[i][2]]
                ids.append(i)

        # very nice print and additional info (top terms in tree, succesors) for printing tree in widget
        if treeData: # we have to reorder results
            # we build a list of succesors. for each node we know which are its succesors in mesh ontology
            for i in ids:
                succesors[i] = []
                for j in ids:
                    if(i != j and self.__isPrecedesor(i,j)):
                        succesors[i].append(j)
            
            # for each node from list above we remove its indirect succesors
            # only  i -1-> j   remain
            for i in succesors.iterkeys():
                #print  "obdelujemo ", i
                succs = succesors[i]
                #print "nasl ", succs
                second_level_succs = []
                for k in succs:     
                    second_level_succs.extend(succesors[k])
                #print "nivo 2 ", second_level_succs
                for m in second_level_succs:
                    #print i, " ", m 
                    if succesors[i].count(m)>0:
                        succesors[i].remove(m)
			
            # we make a list of top nodes
            tops = list(ids)
            for i in ids:
                for j in succesors[i]:
                    tops.remove(j)

            # we pack tops table and succesors hash
            succesors["tops"] = tops
            return succesors,ret 
            
        return ret
        

    def findEnrichedTerms(self,reference, cluster, pThreshold=0.05, treeData = False, callback=False):
        """ like above, but only includes enriched terms (with p value equal or less than pThreshold). Returns a list of (term_id,  term_description, countRef, countCluster, p-value,	enrichment/deprivement, list of corrensponding cids ... anything else necessary). It printOrder is true function returns results in nested lists. This means that at printing time we know if there is any relationship betwen terms"""

        self.clu_att = self.__findMeshAttribute(cluster)
        self.ref_att = self.__findMeshAttribute(reference)
	
        if((not self.calculated or self.reference != reference or self.cluster != cluster) and self.ref_att != "Unknown" and self.clu_att != "Unknown"):	# Do have new data? Then we have to recalculate everything.
            self.reference = reference
            self.cluster = cluster			
            self.__calculateAll()

        # declarations
        ret = dict()
        ids = []
        succesors = dict()		# for each term id -> list of succesors
        succesors["tops"] = []
        
        # if some attributes were unknown
        if (self.clu_att == "Unknown" or self.ref_att == "Unknown"):
            if treeData:
                return ret
            else:
                return succesors, ret

        for i in self.statistics.iterkeys():
            if(self.statistics[i][3] <= pThreshold ) :#or self.statistics[i][4] <= pThreshold ): # 
                ret[i] = [self.statistics[i][0],self.statistics[i][1],self.statistics[i][2],self.statistics[i][3],self.statistics[i][4]]
                ids.append(i)
        
        # very nice print and additional info (top terms in tree, succesors) for printing tree in widget
        if treeData: # we have to reorder results
            # we build a list of succesors. for each node we know which are its succesors in mesh ontology
            for i in ids:
                succesors[i] = []
                for j in ids:
                    if(i != j and self.__isPrecedesor(i,j)):
                        succesors[i].append(j)
            
            # for each node from list above we remove its indirect succesors
            # only  i -1-> j   remain
            for i in succesors.iterkeys():
                succs = succesors[i]
                second_level_succs = []
                for k in succs:     
                    second_level_succs.extend(succesors[k])
                for m in second_level_succs:
                    if succesors[i].count(m)>0:
                        succesors[i].remove(m)
			
            # we make a list of top nodes
            tops = list(ids)
            for i in ids:
                for j in succesors[i]:
                    tops.remove(j)

            # we pack tops table and succesors hash
            succesors["tops"] = tops
            return succesors,ret 
            
        return ret

    def printMeSH(self,terms):
        """for a structured list of terms prints a MeSH ontology. Together with ontology should print things like number
        of compounds, p-values (enrichment), cids, ... see Printing the Tree in orngTree documentation for example of such
        an implementation. The idea is to have only function for printing out the nested list of terms. """
        
    def findCompounds(self,terms, CIDs):
        """from CIDs found those compounds that match terms from the list"""
        # why do we need such a specialized function?

    def __findParents(self,endNodes):
        """for each end node in endNodes function finds all nodes on the way up to the root"""		
        res = []
        for n in endNodes:
            tmp = n
            res.append(tmp)
            for i in range(n.count(".")):
                tmp = tmp.rstrip("1234567890").rstrip(".")
                if(tmp not in res):
                    res.append(tmp)
        return res

    def __findMeshAttribute(self,data):
        """ function tries to find attribute which contains list os mesh terms """
        # we get a list of attributes
        dom = data.domain.attributes
        for k in data:              # for each example
            for i in dom:           # for each attribute
                att = str(i.name)
                try:                                         # we try to use eval()
                    r = eval(str(k[att].value))        
                    if type(r) == list:         # attribute type should be list
                        if self.dataLoaded:         # if ontology is loaded we perform additional test
                            for i in r:
                                if self.toID.has_key(i): return att
                        else:               # otherwise we return list attribute
                            return att
                except SyntaxError:
                    continue
                except NameError:
                    continue   
        print "Program was unable to determinate MeSH attribute."
        return "Unknown"

    def __isPrecedesor(self,a,b):
        """ function returns true if in Mesh ontology exists path from term id a to term id b """
        if b.count(a) > 0:
            return True
        return False
		
    def __calculateAll(self):
        """calculates all statistics"""
        # we build a dictionary 		meshID -> [description, noReference,noCluster, enrichment, deprivement, [cids] ]
        self.statistics = dict()
        n = len(self.reference) 										# reference size
        cln = len(self.cluster)											# cluster size
        # frequency from reference list
        for i in self.reference:
            try:
                endNodes = eval(i[self.ref_att].value)	    # for every CID we look up for end nodes in mesh. for every end node we have to find its ID	
            except SyntaxError:                     # where was a parse error
                print "Error in parsing ",i[self.ref_att].value
                n=n-1
                continue
            #we find ID of end nodes
            endIDs = []
            for k in endNodes:
                if(self.toID.has_key(k)):					# this should be always true, but anyway ...
                    endIDs.extend(self.toID[k])
                else:
                    print "Current ontology does not contain MeSH term ", k, "." 

            # endIDs may be empty > in this case we can skip this example
            if len(endIDs) == 0:
                n = n-1
                continue

            # we find id of all parents
            allIDs = self.__findParents(endIDs)
			
            for k in allIDs:						        # for every meshID we update statistics dictionary
                if(not self.statistics.has_key(k)):		    # first time meshID
                    self.statistics[k] = [self.toDesc[self.toName[k]], 0, 0, 0.0, 0.0, [] ]
                self.statistics[k][1] += 1				    # increased noReference
		
        # frequency from cluster list
        for i in self.cluster:
            try:
                endNodes = eval(i[self.clu_att].value)	# for every CID we look up for end nodes in mesh. for every end node we have to find its ID	
            except SyntaxError:
                print "Error in parsing ",i[self.clu_att].value
                cln = cln - 1
                continue
            # we find ID of end nodes
            endIDs = []
            for k in endNodes:
                if(self.toID.has_key(k)):				
                    endIDs.extend(self.toID[k])	# for every endNode we add all corensponding meshIDs
					
            # endIDs may be empty > in this case we can skip this example
            if len(endIDs) == 0:
                cln = cln-1
                continue
					
            # we find id of all parents
            allIDs = self.__findParents(endIDs)								

            for k in allIDs:						# for every meshID we update statistics dictionary
                self.statistics[k][2] += 1				# increased noCluster 

        self.ratio = float(cln)/float(n)
        
        # enrichment
        for i in self.statistics.iterkeys():
            self.statistics[i][3] = self.__calcEnrichment(n,cln,self.statistics[i][1],self.statistics[i][2])    # p enrichment
            self.statistics[i][4] = float(self.statistics[i][2]) / float(self.statistics[i][1] ) / self.ratio   # fold enrichment

        self.calculated = True		

    def __calcEnrichment(self,n,c,t,tc):
        """n - total number of chemicals ie. size(cluster + reference)
        c - cluster size ie. size(cluster)
        t - number of all chemicals in observed term group
        tc - number of cluster chemicals in observed term group"""
        #print "choose ", n, " ", c, " ", t, " ", tc		
        result=0
        for i in range(0,tc):
            result = result + exp(self.log_comb(t,i) + self.log_comb(n-t,c-i))
        result = result*1.0 / exp(self.log_comb(n,c))
        return (1.0-result)

    def log_comb(self,n, m): # it returns log(nCr(n,m))
        return self.lookup[n] - self.lookup[n-m] - self.lookup[m]

    def __loadOntologyFromDisk(self):
        """ Function loads MeSH ontology (pair cid & MeSH term) and MeSH graph data (graph structure) """
        self.toID = dict()  				#   name -> [ID] be careful !!! One name can match multiple IDs!
        self.toName = dict()				#   ID -> name
        self.toDesc = dict()				# 	 name -> description		

        __dataPath = os.path.join(os.path.dirname(__file__), self.path)
		
        try:		
            # reading graph structure from file
            d = file(os.path.join(__dataPath,'mesh-ontology.dat'))
        except IOError:
            print os.path.join(__dataPath,'mesh-ontology.dat') + " does not exist! Please use function setDataDir(path) to fix this problem."
            return False
			
        # loading ontology graph
        t=0
        for i in d:
            t += 1
            parts = i.split("\t")		# delimiters are tabs
            if(len(parts) != 3):
                print "error reading ontology ", parts[0]

            parts[2] = parts[2].rstrip("\n\r")
            ids = parts[1].split(";")

            self.toID[parts[0]] = ids	# append additional ID
            self.toDesc[parts[0]] = parts[2]
			
            for r in ids:
                self.toName[r] = parts[0]
				
        print "Current MeSH ontology contains ", t, " mesh terms."
        return True

#load data
#reference_data = orange.ExampleTable("data/D06-reference.tab")
#cluster_data = orange.ExampleTable("data/D06-cluster.tab")

#testing code
#t = MeshBrowser()
#res = t.findEnrichedTerms(reference_data,cluster_data,0.02)
#print res
#t.downloadOntology()
#res = t.findEnrichedTerms(reference_data,cluster_data,0.02)