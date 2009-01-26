##!interval=7
##!contact=ales.erjavec@fri.uni-lj.si

import obiKEGG, obiGenomicsUpdate
import orngServerFiles, orngEnviron
import os, sys, tarfile, urllib2
from getopt import getopt

opt = dict(getopt(sys.argv[1:], "u:p:", ["user=", "password="])[0])

username = opt.get("-u", opt.get("--user", "username"))
password = opt.get("-p", opt.get("--password", "password"))


path = os.path.join(orngEnviron.bufferDir, "tmp_kegg/")

u = obiKEGG.Update(local_database_path=path)
serverFiles=orngServerFiles.ServerFiles(username, password)

lines = [line.split("\t") for line in urllib2.urlopen("ftp://ftp.genome.jp/pub/kegg/genes/taxonomy").readlines() if not line.startswith("#")]
keggOrgNames = dict([(line[1].strip(), line[-1][:-5].strip().replace("(", "").replace(")", "") if line[-1].endswith("(EST)\n") else line[-1].strip()) for line in lines if len(line)>1])

essentialOrgs = ["hsa", "ddi", "sce", "mmu"]

uncompressedSize = lambda filename: sum(info.size for info in tarfile.open(filename).getmembers())

realPath = os.path.realpath(os.curdir)
os.chdir(path)

for func, args in u.GetDownloadable() + u.GetUpdatable():
    if func == obiKEGG.Update.UpdateOrganism:
        org = args[0]
        if len(org) > 3 and org.startswith("d"):
            continue
        orgName = keggOrgNames.get(org, org)
        func(u, org)
        filename = "kegg_organism_" + org + ".tar.gz"
        rel_path = u.api._rel_org_dir(org)
        files = [os.path.normpath("pathway//"+rel_path),
                 os.path.normpath("genes//"+rel_path),
                 os.path.normpath(org+"_genenames.pickle")]
        title = "KEGG Pathways and Genes for " + orgName
        tags = ["KEGG", "gene", "pathway", orgName, "#organism:"+orgName] +(["essential"] if org in essentialOrgs else [])
    elif func == obiKEGG.Update.UpdateReference:
        func(u)
        filename = "kegg_reference.tar.gz"
        files = [os.path.normpath("pathway//map"),
                 os.path.normpath("pathway//map_title.tab")]
        title = "KEGG Reference Pathways"
        tags = ["KEGG", "reference", "pathway", "essential"]
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
    filepath = os.path.join(path, filename)
    tFile = tarfile.open(filepath, "w:gz")
    for file in files:
        tFile.add(file)
    tFile.close()
    print "Uploading", filename
    serverFiles.upload("KEGG", filename, filepath, title=title, tags=tags+["#uncompressed:%i" % uncompressedSize(filepath)])
    serverFiles.unprotect("KEGG", filename)
        
