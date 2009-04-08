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
obiGene.NCBIGeneInfo.get_geneinfo_from_ncbi(gene_info_filename)
info = open(gene_info_filename, "rb")

taxids = obiTaxonomy.common_taxids()
essential = obiTaxonomy.essential_taxids()

genes = dict([(taxid, []) for taxid in taxids])
for gi in info:
    if any(gi.startswith(id + "\t") for id in taxids):
        genes[gi.split("\t", 1)[0]].append(gi.strip())

##genes = dict([(taxid, []) for taxid in taxids])
##for gi in info:
##    if any(gi.tax_id == id for id in taxids):
##        genes[gi.tax_id].append(repr(gi))
        
sf = orngServerFiles.ServerFiles(username, password)

for taxid, genes in genes.items():
    filename = os.path.join(tmpdir, "gene_info.%s.db" % taxid)
    f = open(filename, "wb")
##    f.write("\n".join([str(gi) for gi in sorted(genes, key=lambda gi:int(gi.gene_id))]))
    f.write("\n".join(genes))
    f.flush()
    f.close()
    print "Uploading", filename
    sf.upload("NCBI_geneinfo", "gene_info.%s.db" % taxid, filename,
              title = "NCBI gene info for %s" % obiTaxonomy.name(taxid),
              tags = ["NCBI", "gene info", "gene_names", obiTaxonomy.name(taxid)] + (["essential"] if taxid in essential else []))
    sf.unprotect("NCBI_geneinfo", "gene_info.%s.db" % taxid)


