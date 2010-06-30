##interval:7
import obiGene, obiTaxonomy
import orngServerFiles, orngEnviron
import sys, os
from gzip import GzipFile
from getopt import getopt

opt = dict(getopt(sys.argv[1:], "u:p:", ["user=", "password="])[0])

tmpdir = os.path.join(orngEnviron.bufferDir, "tmp_NCBIGene_info")
try:
    os.mkdir(tmpdir)
except Exception, ex:
    pass

username = opt.get("-u", opt.get("--user", "username"))
password = opt.get("-p", opt.get("--password", "password"))

gene_info_filename = os.path.join(tmpdir, "gene_info")
gene_history_filename = os.path.join(tmpdir, "gene_history")

obiGene.NCBIGeneInfo.get_geneinfo_from_ncbi(gene_info_filename)
obiGene.NCBIGeneInfo.get_gene_history_from_ncbi(gene_history_filename)

info = open(gene_info_filename, "rb")
hist = open(gene_history_filename, "rb")

taxids = obiGene.NCBIGeneInfo.common_taxids()
essential = obiGene.NCBIGeneInfo.essential_taxids()

genes = dict([(taxid, []) for taxid in taxids])
for gi in info:
    if any(gi.startswith(id + "\t") for id in taxids):
        genes[gi.split("\t", 1)[0]].append(gi.strip())

history = dict([(taxid, []) for taxid in taxids])
for hi in hist:
    if any(hi.startswith(id + "\t") for id in taxids): 
        history[hi.split("\t", 1)[0]].append(hi.strip())

        
sf = orngServerFiles.ServerFiles(username, password)

for taxid, genes in genes.items():
    filename = os.path.join(tmpdir, "gene_info.%s.db" % taxid)
    f = open(filename, "wb")
    f.write("\n".join(genes))
    f.flush()
    f.close()
    print "Uploading", filename
    sf.upload("NCBI_geneinfo", "gene_info.%s.db" % taxid, filename,
              title = "NCBI gene info for %s" % obiTaxonomy.name(taxid),
              tags = ["NCBI", "gene info", "gene_names", obiTaxonomy.name(taxid)] + (["essential"] if taxid in essential else []))
    sf.unprotect("NCBI_geneinfo", "gene_info.%s.db" % taxid)
    
    filename = os.path.join(tmpdir, "gene_history.%s.db" % taxid)
    f = open(filename, "wb")
    f.write("\n".join(history.get(taxid, "")))
    f.flush()
    f.close()
    print "Uploading", filename
    sf.upload("NCBI_geneinfo", "gene_history.%s.db" % taxid, filename,
              title = "NCBI gene history for %s" % obiTaxonomy.name(taxid),
              tags = ["NCBI", "gene info", "history", "gene_names", obiTaxonomy.name(taxid)] + (["essential"] if taxid in essential else []))
    sf.unprotect("NCBI_geneinfo", "gene_history.%s.db" % taxid)
