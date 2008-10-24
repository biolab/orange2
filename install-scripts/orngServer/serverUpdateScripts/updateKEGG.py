import obiKEGG, obiGenomicsUpdate
import orngServerFiles, orngEnviron
import os, tarfile, urllib2

path = os.path.join(orngEnviron.bufferDir, "tmp_kegg")

u = obiKEGG.Update(local_database_path=path)
serverFiles=orngServerFiles.ServerFiles("username", "password")

lines = [line.split("\t") for line in urllib2.urlopen("ftp://ftp.genome.jp/pub/kegg/genes/taxonomy").readlines() if not line.startswith("#")]
keggOrgNames = dict([(line[1].strip(), line[-1][:-5].strip().replace("(", "").replace(")", "") if line[-1].endswith("(EST)\n") else line[-1].strip()) for line in lines if len(line)>1])

essentialOrgs = ["hsa", "ddi", "sce", "mmu"]

uncompressedSize = lambda filename: sum(info.size for info in tarfile.open(filename).getmember())

realPath = os.path.realpath(os.curdir)
os.chdir(path)
                        
for func, args in u.GetDownloadable() + u.GetUpdatable():
    if func == obiKEGG.UpdateOrganism:
        org = args[0]
        orgName = keggOrgNames.get(org, org)
        func(u, org)
        filename = "kegg_organism_" + org + ".tar.gz"
        rel_path = u.api._rel_org_dir(org)
        files = [os.path.normpath("pathway//"+rel_path),
                 os.path.normpath("genes//"+rel_path),
                 os.path.normpath(org+"_genenames.pickle")]
        title = "KEGG Pathways and Genes for " + orgName
        tags = ["KEGG", "gene", "pathway", orgName, "#organism:"+orgName] +(["essential"] if org in essentialOrgs else [])
    elif func == Update.UpdateReference:
        func(u)
        filename = "kegg_reference.tar.gz"
        files = [os.path.normpath("pathway//map"),
                 os.path.normpath("pathway//map_title.tab")]
        title = "KEGG Reference Pathways"
        tags = ["KEGG", "reference", "pathway", "essential"]
    elif func == Update.UpdateEnzymeAndCompounds:
        func(u)
        filename = "kegg_enzyme_and_compounds.tar.gz"
        files = [os.path.normpath("ligand//compound//"),
                 os.path.normpath("ligand//enzyme//")]
        title = "KEGG Enzymes and Compounds"
        tags = ["KEGG", "enzyme", "compound"]
    elif func == Update.UpdateTaxonomy:
        filename = "kegg_taxonomy.tar.gz"
        title = "KEGG Taxonomy"
        files = [os.path.normpath("genes//taxonomy"),
                 os.path.normpath("genes//genome")]
        tags = ["KEGG", "taxonomy", "organism", "essential"]
    elif func == Update.UpdateOrthology:
        filename = "kegg_othology.tar.gz"
        files = [os.path.normpath("brite//ko//ko00001.keg")]
        title = "KEGG Orhology"
        tags = ["KEGG", "orhology", "essential"]
    filepath = os.path.join(path, filename)
    tFile = tarfile.open(filepath)
    for file in files:
        tFile.add(file)
    tFile.close()
    serverFiles.upload("KEGG", filename, filepath, title=title, tags=tags+["#uncompressed:%i" % uncompressedSize(filepath)])
    serverFiles.unprotect("KEGG", filename)
        