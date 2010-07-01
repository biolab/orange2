##!interval=7
##!contact=ales.erjavec@fri.uni-lj.si

import obiGO, obiTaxonomy, obiGene, obiGenomicsUpdate, orngEnviron, orngServerFiles
import os, sys, shutil, urllib2, tarfile
from getopt import getopt

opt = dict(getopt(sys.argv[1:], "u:p:", ["user=", "password="])[0])

username = opt.get("-u", opt.get("--user", "username"))
password = opt.get("-p", opt.get("--password", "password"))

from collections import defaultdict

tmpDir = os.path.join(orngEnviron.bufferDir, "tmp_GO")
try:
    os.mkdir(tmpDir)
except Exception:
    pass

serverFiles = orngServerFiles.ServerFiles(username, password)

u = obiGO.Update(local_database_path = tmpDir)

uncompressedSize = lambda filename: sum(info.size for info in tarfile.open(filename).getmembers())

def pp(*args, **kw): print args, kw

if u.IsUpdatable(obiGO.Update.UpdateOntology, ()):
    u.UpdateOntology()
    filename = os.path.join(tmpDir, "gene_ontology_edit.obo.tar.gz")
    ##load the ontology to test it
    o = obiGO.Ontology(filename)
    del o
    ##upload the ontology
    print "Uploading gene_ontology_edit.obo.tar.gz"
    serverFiles.upload("GO", "gene_ontology_edit.obo.tar.gz", filename, title = "Gene Ontology (GO)",
                       tags=["gene", "ontology", "GO", "essential", "#uncompressed:%i" % uncompressedSize(filename), "#version:%i" % obiGO.Ontology.version])
    serverFiles.unprotect("GO", "gene_ontology_edit.obo.tar.gz")

#from obiGeneMatch import _dbOrgMap
#
#exclude = ["goa_uniprot", "goa_pdb", "GeneDB_tsetse", "reactome", "goa_zebrafish", "goa_rat", "goa_mouse"]
#lines = [line.split("\t") for line in urllib2.urlopen("ftp://ftp.genome.jp/pub/kegg/genes/taxonomy").readlines() if not line.startswith("#")]
#keggOrgNames = dict([(line[1].strip(), line[-1][:-5].strip().replace("(", "").replace(")", "") if line[-1].endswith("(EST)\n") else line[-1].strip()) for line in lines if len(line)>1])

#additionalNames = {"goa_arabidopsis":"Arabidopsis thaliana", "sgn":"Solanaceae", "PAMGO_Oomycetes":"Oomycete"}
#essentialOrgs = ["goa_human", "sgd", "mgi", "dictyBase"]

orgMap = {"352472":"44689", "562":"83333", "3055":None, "7955":None, "11103":None, "2104":None, "4754":None, "31033":None, "8355":None, "4577":None}

#commonOrgs = dict([(obiGO.from_taxid(orgMap.get(id, id)).pop(), orgMap.get(id, id)) for id in obiTaxonomy.common_taxids() if orgMap.get(id, id) != None])
commonOrgs = dict([(obiGO.from_taxid(id), id) for id in obiTaxonomy.common_taxids() if obiGO.from_taxid(id) != None])

essentialOrgs = [obiGO.from_taxid(id) for id in obiTaxonomy.essential_taxids()]

exclude = ["goa_uniprot", "goa_pdb", "GeneDB_tsetse", "reactome", "goa_zebrafish", "goa_rat", "goa_mouse"]

updatedTaxonomy = defaultdict(set)
import obiTaxonomy

for org in u.GetAvailableOrganisms():
    if org in exclude or org not in commonOrgs:
        continue
    
    if u.IsUpdatable(obiGO.Update.UpdateAnnotation, (org,)):
        u.UpdateAnnotation(org)
        filename = os.path.join(tmpDir, "gene_association." + org + ".tar.gz")
        
        ## Load the annotations to test them and collect all taxon ids from them
        a = obiGO.Annotations(filename, genematcher=obiGene.GMDirect())
        taxons = set([ann.taxon for ann in a.annotations])
        for taxId in [t.split(":")[-1] for t in taxons if "|" not in t]: ## exclude taxons with cardinality 2
            updatedTaxonomy[taxId].add(org)
        del a
        ## Upload the annotation
#        if org in _dbOrgMap:
#            orgName = keggOrgNames[_dbOrgMap[org]].split("(")[0].strip()
#        elif org in additionalNames:
#            orgName = additionalNames[org]
#        else:
#            orgName = org
        orgName = obiTaxonomy.name(commonOrgs[org])
#            print "unknown organism name translation for:", org
        print "Uploading", "gene_association." + org + ".tar.gz"
        serverFiles.upload("GO", "gene_association." + org + ".tar.gz", filename, title = "GO Annotations for " + orgName,
                           tags=["gene", "annotation", "ontology", "GO", orgName, "#uncompressed:%i" % uncompressedSize(filename),
                                 "#organism:"+orgName, "#version:%i" % obiGO.Annotations.version] + (["essential"] if org in essentialOrgs else []))
        serverFiles.unprotect("GO", "gene_association." + org + ".tar.gz")
        
try:
    import cPickle
    tax = cPickle.load(open(os.path.join(tmpDir, "taxonomy.pickle")))
except Exception:
    tax = {}

## Upload taxonomy if any differences in the updated taxonomy
if any(tax.get(key, set()) != updatedTaxonomy.get(key, set()) for key in set(updatedTaxonomy)):
    tax.update(updatedTaxonomy)
    cPickle.dump(tax, open(os.path.join(tmpDir, "taxonomy.pickle"), "w"))
    serverFiles.upload("GO", "taxonomy.pickle", os.path.join(tmpDir, "taxonomy.pickle"), title="GO taxon IDs",
                       tags = ["GO", "taxon", "organism", "essential", "#version:%i" % obiGO.Taxonomy.version])
    serverFiles.unprotect("GO", "taxonomy.pickle")