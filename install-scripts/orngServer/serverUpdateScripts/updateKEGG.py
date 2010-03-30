##!interval=7
##!contact=ales.erjavec@fri.uni-lj.si

import obiKEGG, obiGene, obiTaxonomy
import orngServerFiles, orngEnviron
import os, sys, tarfile, urllib2, shutil
from getopt import getopt

import obiData
obiKEGG.borg_class(obiData.FtpDownloader) #To limit the number of connections


opt = dict(getopt(sys.argv[1:], "u:p:", ["user=", "password="])[0])

username = opt.get("-u", opt.get("--user", "username"))
password = opt.get("-p", opt.get("--password", "password"))

tmp_path = os.path.join(orngEnviron.bufferDir, "tmp_KEGG/")

#u = obiKEGG.Update(local_database_path=path)
serverFiles=orngServerFiles.ServerFiles(username, password)

#def output(self, *args, **kwargs):
#    print args, kwargs
    
#serverFiles = type("bla", (object,), dict(upload=output, unprotect=output))()


try:
    shutil.rmtree(tmp_path)
except Exception, ex:
    pass

try:
    os.mkdir(tmp_path)
except Exception, ex:
    pass

realPath = os.path.realpath(os.curdir)
os.chdir(tmp_path)

obiKEGG.DEFAULT_DATABASE_PATH = tmp_path

uncompressedSize = lambda filename: sum(info.size for info in tarfile.open(filename).getmembers())

def tar(filename, mode="w:gz", add=[]):
    f = tarfile.open(filename, mode)
    for path in add:
        f.add(path)
    f.close()
    return uncompressedSize(filename)

print "KEGGGenome.download()"
obiKEGG.KEGGGenome.download()

genome = obiKEGG.KEGGGenome()
        
essential_organisms = genome.essential_organisms()
common_organisms = genome.common_organisms()

files=["genes/genome"]

print "Uploading kegg_genome.tar.gz"

size = tar("kegg_genome.tar.gz", add=files)
serverFiles.upload("KEGG", "kegg_genome.tar.gz", "kegg_genome.tar.gz", title="KEGG Genome",
                   tags=["kegg", "genome", "taxonomy", "essential", "#uncompressed:%i" % size, "#compression:tar.gz", 
                         "#version:%s" % obiKEGG.KEGGGenome.VERSION, "#files:%s" % "!@".join(files)])
serverFiles.unprotect("KEGG", "kegg_genome.tar.gz")

print "KEGGEnzymes.download()"
obiKEGG.KEGGEnzymes.download()
enzymes = obiKEGG.KEGGEnzymes()

print "KEGGCompounds.download()"
obiKEGG.KEGGCompounds.download()
compounds = obiKEGG.KEGGCompounds()

print "KEGGReactions.download()"
obiKEGG.KEGGReactions.download()
reactions = obiKEGG.KEGGReactions()

files = ["ligand/enzyme/", "ligand/reaction/", "ligand/compound/"]
size = tar("kegg_ligand.tar.gz", add=files)

print "Uploading kegg_ligand.tar.gz"
serverFiles.upload("KEGG", "kegg_ligand.tar.gz", "kegg_ligand.tar.gz", title="KEGG Ligand",
                   tags=["kegg", "enzymes", "compunds", "reactions", "essential", "#uncompressed:%i" % size,
                         "#compression:tar.gz", "#version:v1.0", "#files:%s" % "!@".join(files)])
serverFiles.unprotect("KEGG", "kegg_ligand.tar.gz")


### KEGG Reference Pathways
############################

print 'KEGGPathway.download_pathways("map")'
obiKEGG.KEGGPathway.download_pathways("map")

files = ["pathway/map/"]

size = tar("kegg_pathways_map.tar.gz", add=files)

print "Uploading kegg_pathways_map.tar.gz"
serverFiles.upload("KEGG", "kegg_pathways_map.tar.gz", "kegg_pathways_map.tar.gz", title="KEGG Reference pathways (map)",
                   tags=["kegg", "map", "pathways", "reference", "essential", "#uncompressed:%i" % size,
                         "#compression:tar.gz", "#version:%s" % obiKEGG.KEGGPathway.VERSION, "#files:%s" % "!@".join(files)])
serverFiles.unprotect("KEGG", "kegg_pathways_map.tar.gz")

print 'KEGGPathway.download_pathways("ec")'
obiKEGG.KEGGPathway.download_pathways("ec")

files = ["pathway/ec/", "xml/kgml/metabolic/ec/"]

size = tar("kegg_pathways_ec.tar.gz", add=files)

print "Uploading kegg_pathways_ec.tar.gz"
serverFiles.upload("KEGG", "kegg_pathways_ec.tar.gz", "kegg_pathways_ec.tar.gz", title="KEGG Reference pathways (ec)",
                   tags=["kegg", "ec", "pathways", "reference", "essential", "#uncompressed:%i" % size,
                         "#compression:tar.gz", "#version:%s" % obiKEGG.KEGGPathway.VERSION, "#files:%s" % "!@".join(files)])
serverFiles.unprotect("KEGG", "kegg_pathways_ec.tar.gz")

print 'KEGGPathway.download_pathways("ko")'
obiKEGG.KEGGPathway.download_pathways("ko")

files = ["pathway/ko/", "xml/kgml/metabolic/ko/", "xml/kgml/non-metabolic/ko/"]

size = tar("kegg_pathways_ko.tar.gz", add=files)

print "Uploading kegg_pathways_ko.tar.gz"
serverFiles.upload("KEGG", "kegg_pathways_ko.tar.gz", "kegg_pathways_ko.tar.gz", title="KEGG Reference pathways (ko)",
                   tags=["kegg", "ko", "pathways", "reference", "essential", "#uncompressed:%i" % size,
                         "#compression:tar.gz", "#version:%s" % obiKEGG.KEGGPathway.VERSION, "#files:%s" % "!@".join(files)])
serverFiles.unprotect("KEGG", "kegg_pathways_ko.tar.gz")


for org_code in common_organisms:
    org_name = genome[org_code].definition
    
    ### KEGG Genes
    ##############
    
    print "KEGGGenes.download(%s)" % org_code
    obiKEGG.KEGGGenes.download(org_code)

    genes = obiKEGG.KEGGGenes(org_code)
    
    filename = "kegg_genes_%s.tar.gz" % org_code
    files = [os.path.split(obiKEGG.KEGGGenes.filename(org_code))[0]]
    
    size = tar(filename, add=files)
    
    print "Uploading", filename
    serverFiles.upload("KEGG", filename, filename, title="KEGG Genes for " + org_name,
                       tags=["kegg", "genes", org_name, "#uncompressed:%i" % size, "#compression:tar.gz",
                             "#version:%s" % obiKEGG.KEGGGenes.VERSION, "#files:%s" % "!@".join(files)] + (["essential"] if org_code in essential_organisms else []))
    serverFiles.unprotect("KEGG", filename)
    
    ### KEGG Pathways
    #################
    
    print "KEGGPathway.download_pathways(%s)" % org_code
    obiKEGG.KEGGPathway.download_pathways(org_code)
    
    filename = "kegg_pathways_%s.tar.gz" % org_code
    files = [obiKEGG.KEGGPathway.directory_png(org_code, path="").lstrip("/"), 
             obiKEGG.KEGGPathway.directory_kgml(org_code, path="").lstrip("/"),
             obiKEGG.KEGGPathway.directory_kgml(org_code, path="").lstrip("/").replace("metabolic", "non-metabolic")]
    
    size = tar(filename, add=files)
    
    print "Uploading", filename
    serverFiles.upload("KEGG", filename, filename, title="KEGG Pathways for " + org_name,
                       tags=["kegg", "genes", org_name, "#uncompressed:%i" % size, "#compression:tar.gz",
                             "#version:%s" % obiKEGG.KEGGPathway.VERSION, "#files:%s" % "!@".join(files)] + (["essential"] if org_code in essential_organisms else []))
    serverFiles.unprotect("KEGG", filename)
    
    
brite_ids = [line.split()[-1] for line in urllib2.urlopen("ftp://ftp.genome.jp/pub/kegg/brite/br/").read().splitlines() if line.split()[-1].endswith(".keg")]
ko_brite_ids = [line.split()[-1] for line in urllib2.urlopen("ftp://ftp.genome.jp/pub/kegg/brite/ko/").read().splitlines() if line.split()[-1].endswith(".keg")]

for id in brite_ids + ko_brite_ids:
    print "KEGGBrite.download(%s)" % id.split(".")[0]
    obiKEGG.KEGGBrite.download(id.split(".")[0])
    
files = ["brite/ko/", "brite/br/"]
size = tar("kegg_brite.tar.gz", add=files)

print "Uploading kegg_brite.tar.gz"
serverFiles.upload("KEGG", "kegg_brite.tar.gz", "kegg_brite.tar.gz", title="KEGG Brite",
                   tags=["kegg", "brite", "essential", "#uncompressed:%i" % size,
                         "#compression:tar.gz", "#version:%s" % obiKEGG.KEGGBrite.VERSION, "#files:%s" % "!@".join(files)])
serverFiles.unprotect("KEGG", "kegg_brite.tar.gz")

os.chdir(realPath)
