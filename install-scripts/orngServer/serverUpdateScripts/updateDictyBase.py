##interval:7
import orngServerFiles, orngEnviron
import sys, os
from gzip import GzipFile
from getopt import getopt
import tempfile
from obiDicty import DictyBase
import shutil

opt = dict(getopt(sys.argv[1:], "u:p:", ["user=", "password="])[0])

tmpdir = tempfile.mkdtemp("dictybase")

username = opt.get("-u", opt.get("--user", "username"))
password = opt.get("-p", opt.get("--password", "password"))

print username, password

base = DictyBase.pickle_data()
filename = os.path.join(tmpdir, "tf")

f = open(filename, 'wb')
f.write(base)
f.close()

dom = DictyBase.domain
fn = DictyBase.filename

sf = orngServerFiles.ServerFiles(username, password)

try:
    sf.create_domain('dictybase')
except:
    pass

print filename

sf.upload(dom, fn, filename, title="dictyBase gene aliases",
    tags=DictyBase.tags)
sf.unprotect(dom, fn)

shutil.rmtree(tmpdir)
