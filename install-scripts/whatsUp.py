import os, re

file_re = re.compile(r'/(?P<fname>.*)/(?P<version>.*)/(?P<date>.*)//')
dir_re = re.compile(r'D/(?P<dname>.*)////')

os.chdir("c:/inetpub/wwwusers/orange/download/lastStable")
exclude = [x.lower().replace("\\", "/")[:-1] for x in open("orange/exclude.lst", "rt").readlines()]
exclude.append("orange/doc/datasets")

def listfiles(baseurl, basedir, subdir):
    if os.path.isdir(baseurl+subdir+"CVS"):
        f = open(baseurl+subdir+"CVS/Entries", "rt")
        
        for line in f:
            fnd = file_re.match(line)
            if fnd:
                fname, version = fnd.group("fname", "version")
                if (basedir+subdir+fname).lower() not in exclude:
                    outf.write("%s=%s:%s\n" % (basedir+subdir+fname, version, baseurl+subdir+fname))
                else:
                    print basedir+subdir+fname
                continue

            fnd = dir_re.match(line)
            if fnd:
                dname = fnd.group("dname")
                if (basedir+subdir+dname).lower() not in exclude:
                    listfiles(baseurl, basedir, subdir+dname+"/")
        f.close()

    
outf = open(r"../whatsup.txt", "wt")
listfiles("orange/", "orange/", "")
listfiles("Genomics/", "orange/orangeWidgets/Genomics/", "")
outf.close()