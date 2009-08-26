##!interval=7
##!contact=ales.erjavec@fri.uni-lj.si

import obiKEGG, obiGenomicsUpdate, obiGene, obiTaxonomy
import orngServerFiles, orngEnviron
import os, sys, tarfile, urllib2
from getopt import getopt

opt = dict(getopt(sys.argv[1:], "u:p:", ["user=", "password="])[0])

username = opt.get("-u", opt.get("--user", "username"))
password = opt.get("-p", opt.get("--password", "password"))

path = os.path.join(orngEnviron.bufferDir, "tmp_kegg/")

u = obiKEGG.Update(local_database_path=path)
serverFiles=orngServerFiles.ServerFiles(username, password)

for i in range(3):
    try:
        lines = [line.split("\t") for line in urllib2.urlopen("ftp://ftp.genome.jp/pub/kegg/genes/taxonomy").readlines() if not line.startswith("#")]
        break
    except Exception, ex:
        print ex
        
keggOrgNames = dict([(line[1].strip(), line[-1][:-5].strip().replace("(", "").replace(")", "") if line[-1].endswith("(EST)\n") else line[-1].strip()) for line in lines if len(line)>1])

essentialOrgs = [obiKEGG.from_taxid(id) for id in obiTaxonomy.essential_taxids()] #["hsa", "ddi", "sce", "mmu"]

orgMap = {"562":"511145", "2104":"272634", "5833":"36329", "4896":"284812", "11103":None, "4754":None, "4577":None}

commonOrgs = [obiKEGG.from_taxid(orgMap.get(id, id)) for id in obiTaxonomy.common_taxids() if orgMap.get(id, id) != None]

fix = {"eath":"ath", "ecre":"cre", "eosa":"osa"}

commonOrgs = [fix.get(code, code) for code in commonOrgs]

uncompressedSize = lambda filename: sum(info.size for info in tarfile.open(filename).getmembers())

realPath = os.path.realpath(os.curdir)
os.chdir(path)

for func, args in u.GetDownloadable() + u.GetUpdatable():
#for func, args in [(obiKEGG.Update.UpdateOrganism, (org,)) for org in commonOrgs[5:9]]:
    if func == obiKEGG.Update.UpdateOrganism and args[0] in commonOrgs:
        org = args[0]
        
        orgName = keggOrgNames.get(org, org)
        try:
            print func, org
            func(u, org)
            organism = obiKEGG.KEGGOrganism(org, genematcher=obiGene.GMDirect(), local_database_path=path)
            genes = list(organism.genes) ## test to see if the _genes.pickle was created
            
            print os.path.join(path, "genes", u.api._rel_org_dir(org), "_genes.pickle"), "exists:", os.path.exists(os.path.join(path, "genes", u.api._rel_org_dir(org), "_genes.pickle"))
#            assert(os.path.exists(os.path.join(path, "genes", u.api._rel_org_dir(org), "_genes.pickle")))
            print genes[:5]
            print path
        except Exception, ex:
            print "Error:", ex
            continue
        filename = "kegg_organism_" + org + ".tar.gz"
        rel_path = u.api._rel_org_dir(org)
        files = [os.path.normpath("pathway//"+rel_path),
                 os.path.normpath("genes//"+rel_path+"//"+"_genes.pickle"),
                 os.path.normpath(org+"_genenames.pickle")]
        title = "KEGG Pathways and Genes for " + orgName
        tags = ["KEGG", "gene", "pathway", orgName, "#organism:"+orgName, "#version:%i" % obiKEGG.KEGGOrganism.version] +(["essential"] if org in essentialOrgs else [])
    elif func == obiKEGG.Update.UpdateReference:
        func(u)
        filename = "kegg_reference.tar.gz"
        files = [os.path.normpath("pathway//map"),
                 os.path.normpath("pathway//map_title.tab")]
        title = "KEGG Reference Pathways"
        tags = ["KEGG", "reference", "pathway", "essential", "#version:%i" % obiKEGG.KEGGOrganism.version]
    elif func == obiKEGG.Update.UpdateEnzymeAndCompounds:
        func(u)
        filename = "kegg_enzyme_and_compounds.tar.gz"
        files = [os.path.normpath("ligand//compound//"),
                 os.path.normpath("ligand//enzyme//")]
        title = "KEGG Enzymes and Compounds"
        tags = ["KEGG", "enzyme", "compound"]
    elif func == obiKEGG.Update.UpdateTaxonomy:
        func(u)
        filename = "kegg_taxonomy.tar.gz"
        title = "KEGG Taxonomy"
        files = [os.path.normpath("genes//taxonomy"),
                 os.path.normpath("genes//genome")]
        tags = ["KEGG", "taxonomy", "organism", "essential"]
    elif func == obiKEGG.Update.UpdateOrthology:
        func(u)
        filename = "kegg_orthology.tar.gz"
        files = [os.path.normpath("brite//ko//ko00001.keg")]
        title = "KEGG Orthology"
        tags = ["KEGG", "orthology", "essential"]
    else:
        continue
    filepath = os.path.join(path, filename)
    tFile = tarfile.open(filepath, "w:gz")
    for file in files:
        tFile.add(file)
    tFile.close()
    print "Uploading", filename
    serverFiles.upload("KEGG", filename, filepath, title=title, tags=tags+["#uncompressed:%i" % uncompressedSize(filepath)])
    serverFiles.unprotect("KEGG", filename)
        
