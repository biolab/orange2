import os, re

file_re = re.compile(r'/(?P<fname>.*)/(?P<version>.*)/(?P<date>.*)//')
dir_re = re.compile(r'D/(?P<dname>.*)////')

def listfiles(dirname):
    if os.path.isdir("CVS"):
        f = open("CVS/Entries", "rt")
        
        for line in f:
            fnd = file_re.match(line)
            if fnd:
                outf.write("%s%s=%s\n" % ((dirname, ) + fnd.group("fname", "version")))
                continue

            fnd = dir_re.match(line)
            if fnd:
                dname = fnd.group("dname")
                os.chdir(dname)
                listfiles(dirname + dname + "/")
                os.chdir("..")
        f.close()

outf = open(r"c:\inetpub\wwwusers\orange\download\whatsup.txt", "wt")
listfiles("")
outf.close()