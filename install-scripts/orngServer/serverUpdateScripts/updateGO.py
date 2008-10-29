import obiGO, obiGenomicsUpdate, orngEnviron, orngServerFiles
import os, sys, shutil, urllib2, tarfile

tmpDir = os.path.join(orngEnviron.bufferDir, "tmp_GO")
try:
    os.mkdir(tmpDir)
except Exception:
    pass

serverFiles=orngServerFiles.ServerFiles("username", "password")

u = obiGO.Update(local_database_path = tmpDir)

uncompressedSize = lambda filename: sum(info.size for info in tarfile.open(filename).getmembers())

if u.IsUpdatable(obiGO.Update.UpdateOntology, ()):
    u.UpdateOntology()
    filename = os.path.join(tmpDir, "gene_ontology_edit.obo.tar.gz")
    ##load the ontology to test it
    o = obiGO.Ontology(filename)
    ##upload the ontology
##    print "Uploading gene_ontology_edit.obo.tar.gz"
    serverFiles.upload("GO", "gene_ontology_edit.obo.tar.gz", filename, title = "Gene Ontology (GO)",
                       tags=["gene", "ontology", "GO", "essential", "#uncompressed:%i" % uncompressedSize(filename)])
    serverFiles.unprotect("GO", "gene_ontology_edit.obo.tar.gz")

from obiGeneMatch import _dbOrgMap

exclude = ["goa_uniprot", "goa_pdb", "GeneDB_tsetse", "reactome", "goa_zebrafish", "goa_rat", "goa_mouse"]
lines = [line.split("\t") for line in urllib2.urlopen("ftp://ftp.genome.jp/pub/kegg/genes/taxonomy").readlines() if not line.startswith("#")]
keggOrgNames = dict([(line[1].strip(), line[-1][:-5].strip().replace("(", "").replace(")", "") if line[-1].endswith("(EST)\n") else line[-1].strip()) for line in lines if len(line)>1])

additionalNames = {"goa_arabidopsis":"Arabidopsis thaliana", "sgn":"Solanaceae", "PAMGO_Oomycetes":"Oomycete"}
essentialOrgs = ["goa_human", "sgd", "mgi", "dictyBase"]

for org in u.GetAvailableOrganisms():
    if org in exclude:
        continue
    if u.IsUpdatable(obiGO.Update.UpdateAnnotation, (org,)):
        u.UpdateAnnotation(org)
        filename = os.path.join(tmpDir, "gene_association." + org + ".tar.gz")
        ##tload the annotations to test them
        a = obiGO.Annotations(filename)
        ##upload the annotation
        if org in _dbOrgMap:
            orgName = keggOrgNames[_dbOrgMap[org]]
        elif org in additionalNames:
            orgName = additionalNames[org]
        else:
            orgName = org
            print "unknown organism name translation for:", org
##        print "Uploading", "gene_association." + org + ".tar.gz"
        serverFiles.upload("GO", "gene_association." + org + ".tar.gz", filename, title = "GO Annotations for " + orgName,
                           tags=["gene", "annotation", "GO", orgName, "#uncompressed:%i" % uncompressedSize(filename),
                                 "#organism:"+orgName] + (["essential"] if org in essentialOrgs else []))
        serverFiles.unprotect("GO", "gene_association." + org + ".tar.gz")
        
        