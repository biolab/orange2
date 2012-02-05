##!interval=7
##!contact=ales.erjavec@fri.uni-lj.si

import obiKEGG2, obiGene, obiTaxonomy, obiGeneSets
import os, sys, tarfile, urllib2, shutil, cPickle
from getopt import getopt

from Orange.misc import serverfiles, ConsoleProgressBar
DOMAIN = "KEGG"

opt = dict(getopt(sys.argv[1:], "u:p:", ["user=", "password="])[0])

username = opt.get("-u", opt.get("--user", "username"))
password = opt.get("-p", opt.get("--password", "password"))

sf = serverfiles.ServerFiles(username, password)

genome = obiKEGG2.KEGGGenome()
common = genome.common_organisms()

rev_taxmap = dict([(v, k) for k, v in genome.TAXID_MAP.items()])


for org in common:
    
    #####################
    # Create gene aliases
    #####################
    
#    genes = obiKEGG2.KEGGGenes(org)
#    
#    pb = ConsoleProgressBar("Retriving KEGG ids for %r:" % org)
#    genes.pre_cache(progress_callback=pb.set_state)
#    aliases = []
#    for key, entry in genes.iteritems():
#        aliases.append(set([key]) | set(entry.alt_names))
#    pb.finish()
#    
#    taxid = obiKEGG2.to_taxid(org)
#    ids_filename = "kegg_gene_id_aliases_" + taxid + ".pickle"
#    filename = serverfiles.localpath(DOMAIN, ids_filename)
#    
#    cPickle.dump(aliases, open(filename, "wb"))
#    
#    print "Uploading", ids_filename
#    sf.upload(DOMAIN, ids_filename, filename,
#              "KEGG Gene id aliases",
#              tags=["KEGG", "genes", "aliases", 
#                    "#version:%s" % obiKEGG2.MatcherAliasesKEGG.VERSION
#                    ],
#              )
#    sf.unprotect(DOMAIN, ids_filename)
    
    ##########################
    # Create pathway gene sets
    ##########################
    
    organism = obiKEGG2.KEGGOrganism(org)
    ge = genome[org]
    
    taxid = rev_taxmap.get(ge.taxid, ge.taxid)
    gene_sets = obiGeneSets.keggGeneSets(taxid)
    
    print "Uploading pathway gene sets for", taxid, "(%s)" % org
    obiGeneSets.register_serverfiles(gene_sets, sf)
    