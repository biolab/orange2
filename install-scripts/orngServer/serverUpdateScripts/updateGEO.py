##!interval=7
##!contact=blaz.zupan@fri.uni-lj.si

import obiTaxonomy
import sys
import orngServerFiles
reload(orngServerFiles)
from getopt import getopt
import cPickle
import re
import ftplib
import obiGEO
reload(obiGEO)

DOMAIN = "GEO"
GDS_INFO = "gds_info.pickled"
TITLE = "Gene Expression Omnibus data sets information"
TAGS = ["Gene Expression Omnibus", "data sets", "GEO", "GDS"]

FTP_NCBI = "ftp.ncbi.nih.gov"
NCBI_DIR = "pub/geo/DATA/SOFT/GDS"

opt = dict(getopt(sys.argv[1:], "u:p:", ["user=", "password="])[0])
username = opt.get("-u", opt.get("--user", "username"))
password = opt.get("-p", opt.get("--password", "password"))
server = orngServerFiles.ServerFiles(username, password)

# check if the DOMAIN/files are already on the server, else, create
if DOMAIN not in server.listdomains():
    # DOMAIN does not exist on the server, create it
    server.create_domain(DOMAIN)

localfile = orngServerFiles.localpath(DOMAIN, GDS_INFO)

path = orngServerFiles.localpath(DOMAIN)
if GDS_INFO in server.listfiles(DOMAIN):
    print "Updating info file from server ..."
    orngServerFiles.update(DOMAIN, GDS_INFO)
else:
    print "Creating a local path..."
    orngServerFiles.createPathForFile(localfile)
    f = file(localfile, "wb")
    cPickle.dump(({}, {}), f, True)
    f.close()
    server.upload(DOMAIN, GDS_INFO, localfile, TITLE, TAGS)
    server.protect(DOMAIN, GDS_INFO, "0")

# read the information from the local file
gds_info, excluded = cPickle.load(file(localfile, "rb"))
# excluded should be a dictionary (GEO_ID, TAX_ID)

# uncomment the following if need to refresh the data base
# gds_info, excluded = ({}, {})

# list of common organisms may have changed, rescan excluded list
excluded = dict([(id, taxid) for id, taxid in excluded.items() 
                 if taxid not in obiTaxonomy.common_taxids()])
excluded.update([(id, info["taxid"]) for id, info in gds_info.items() 
                 if info["taxid"] not in obiTaxonomy.common_taxids()])
gds_info = dict([(id, info) for id, info in gds_info.items() 
                 if info["taxid"] in obiTaxonomy.common_taxids()])

# get the list of GDS files from NCBI directory

print "Retreiving ftp directory ..."
ftp = ftplib.FTP(FTP_NCBI)
ftp.login()
ftp.cwd(NCBI_DIR)
dirlist = []
ftp.dir(dirlist.append)
m = re.compile("GDS[0-9]*")
gds_names = [m.search(d).group(0) for d in dirlist if m.search(d)]
gds_names = [name for name in gds_names if not(name in gds_info or name in excluded)]
skipped = []

if len(gds_names):
    for count, gds_name in enumerate(gds_names):
        print "%3d of %3d -- Adding %s ..." % (count+1, len(gds_names), gds_name)
        try:
            gds = obiGEO.GDS(gds_name)
            if gds.info["taxid"] not in obiTaxonomy.common_taxids():
                excluded[gds_name] = gds.info["taxid"]
                print "... excluded (%s)." % gds.info["sample_organism"]
            else:
                gds_info.update({gds_name: gds.info})
                f = file(localfile, "wb")
                cPickle.dump((gds_info, excluded), f, True)
                f.close()
                print "... added."
        except:
            print "... skipped (error)"
            skipped.append(gds_name)
    
    print "Updating %s:%s on the server ..." % (DOMAIN, GDS_INFO)
 
    server.upload(DOMAIN, GDS_INFO, localfile, TITLE, TAGS)
    server.protect(DOMAIN, GDS_INFO, "0")
else:
    print "No update required."

print
print "GDS data sets: %d" % len(gds_info)
print "Organisms:"
organisms = [info["sample_organism"] for info in gds_info.values()]
for org in set(organisms):
    print "  %s (%d)" % (org, organisms.count(org))
