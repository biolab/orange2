import obiTaxonomy
import orngServerFiles
##!interval=7
##!contact=ales.erjavec@fri.uni-lj.si

import orngEnviron
import os, sys, tarfile

from getopt import getopt

opt = dict(getopt(sys.argv[1:], "u:p:", ["user=", "password="])[0])

username = opt.get("-u", opt.get("--user", "username"))
password = opt.get("-p", opt.get("--password", "password"))

path = os.path.join(orngEnviron.bufferDir, "tmp_Taxonomy")
serverFiles = orngServerFiles.ServerFiles(username, password)
u = obiTaxonomy.Update(local_database_path=path)

uncompressedSize = lambda filename: sum(info.size for info in tarfile.open(filename).getmembers())

if u.IsUpdatable(obiTaxonomy.Update.UpdateTaxonomy, ()):
    u.UpdateTaxonomy()
    serverFiles.upload("Taxonomy", "ncbi_taxonomy.tar.gz", os.path.join(path, "ncbi_taxonomy.tar.gz"), title ="NCBI Taxonomy",
                       tags=["NCBI", "taxonomy", "organism names", "essential", "#uncompressed:%i" % uncompressedSize(os.path.join(path, "ncbi_taxonomy.tar.gz"))])
    serverFiles.unprotect("Taxonomy", "ncbi_taxonomy.tar.gz")
