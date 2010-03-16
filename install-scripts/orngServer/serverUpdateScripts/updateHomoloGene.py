##!interval=7
##!contact=ales.erjavec@fri.uni-lj.si

import obiHomoloGene
import orngServerFiles

import orngEnviron
import os, sys
import gzip, shutil

from getopt import getopt

opt = dict(getopt(sys.argv[1:], "u:p:", ["user=", "password="])[0])

username = opt.get("-u", opt.get("--user", "username"))
password = opt.get("-p", opt.get("--password", "password"))

path = os.path.join(orngEnviron.bufferDir, "tmp_HomoloGene")
serverFiles = orngServerFiles.ServerFiles(username, password)

try:
    os.mkdir(path)
except OSError:
    pass
filename = os.path.join(path, "homologene.data")
obiHomoloGene.HomoloGene.download_from_NCBI(filename)
uncompressed = os.stat(filename).st_size
import gzip, shutil
f = gzip.open(filename + ".gz", "wb")
shutil.copyfileobj(open(filename), f)
f.close()

#serverFiles.create_domain("HomoloGene")
print "Uploading homologene.data"
serverFiles.upload("HomoloGene", "homologene.data", filename + ".gz", title="HomoloGene",
                   tags=["genes", "homologs", "HomoloGene", "#compression:gz",
                         "#uncompressed:%i" % uncompressed, 
                         "#version:%i" % obiHomoloGene.HomoloGene.VERSION])
serverFiles.unprotect("HomoloGene", "homologene.data")

####
# InParanioid Orthologs update
####

organisms = {"3702": "A.thaliana",
            "9913": "B.taurus",
            "6239": "C.elegans",
            "3055": "C.reinhardtii",
            "7955": "D.rerio",
            "352472": "D.discoideum",
            "7227":  "D.melanogaster",
            "562":  "E.coliK12",
            #"11103", # Hepatitis C virus
            "9606": "H.sapiens",
            "10090": "M.musculus",
            #"2104",  # Mycoplasma pneumoniae
            "4530": "O.sativa",
            "5833": "P.falciparum",
            #"4754",  # Pneumocystis carinii
            "10116": "R.norvegicus",
            "4932": "S.cerevisiae",
            "4896":  "S.pombe",
            "31033": "T.rubripes"
            #"8355",  # Xenopus laevis
            #"4577",  # Zea mays
            }

import urllib2
combined_orthologs = []
        
def gen(i=0):
    while True:
        yield str(i)
        i += 1

from collections import defaultdict
unique_cluster_id = defaultdict(gen().next)
         
organisms = sorted(organisms.values())

for i, org1 in enumerate(organisms):
    for org2 in organisms[i+1:]:
        print "http://inparanoid.sbc.su.se/download/current/orthoXML/InParanoid.%s-%s.orthoXML" % (org1, org2)
        try:
            stream = urllib2.urlopen("http://inparanoid.sbc.su.se/download/current/orthoXML/InParanoid.%s-%s.orthoXML" % (org1, org2))
        except Exception, ex:
            print >> sys.stderr, ex
            continue
        orthologs = obiHomoloGene._parseOrthoXML(stream)
        orthologs = [(unique_cluster_id[org1, org2, clid], taxid, gene_symbol) for (clid, taxid , gene_symbol) in orthologs]
        
        combined_orthologs.extend(orthologs)
        
#import cPickle
#cPickle.dump(combined_orthologs, open("orthologs.pck", "wb"))
#combined_orthologs = cPickle.load(open("orthologs.pck"))

import sqlite3

filename  = os.path.join(path, "InParanoid.sqlite")
con = sqlite3.connect(filename)
con.execute("drop table if exists homologs")
con.execute("create table homologs (groupid text, taxid text, geneid text)")
con.execute("create index group_index on homologs(groupid)")
con.execute("create index geneid_index on homologs(geneid)")
con.executemany("insert into homologs values (?, ?, ?)", combined_orthologs)
con.commit()



file = open(filename, "rb")
gzfile = gzip.GzipFile(filename + ".gz", "wb")
shutil.copyfileobj(file, gzfile)
gzfile.close()

print "Uploading InParanoid.sqlite"
serverFiles.upload("HomoloGene", "InParanoid.sqlite", filename + ".gz", title="InParanoid: Eukaryotic Ortholog Groups",
                   tags=["genes", "homologs", "orthologs", "InParanoid", "#compression:gz",
                         "#uncompressed:%i" % os.stat(filename).st_size,
                         "#version:%i" % obiHomoloGene.InParanoid.VERSION])
serverFiles.unprotect("HomoloGene", "InParanoid.sqlite")
        
        
            