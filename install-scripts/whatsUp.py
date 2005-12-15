import os, re

file_re = re.compile(r'[^/D]*/(?P<fname>.*)/(?P<version>.*)/(?P<date>.*)/.*/')
dir_re = re.compile(r'[^/]*D/(?P<dname>.*)////')

os.chdir("c:/inetpub/wwwusers/orange/download/update")
exclude = [x.lower().replace("\\", "/")[:-1] for x in open("exclude.lst", "rt").readlines()]
#exclude.append("doc/datasets")

def listfiles(baseurl, basedir, subdir):
    if os.path.isdir(baseurl+subdir+"CVS"):
        for fna in ["CVS/Entries", "CVS/Entries.log"]:
            if os.path.exists(baseurl+subdir+fna):
                f = open(baseurl+subdir+fna, "rt")
        
                for line in f:
                    fnd = file_re.match(line)
                    if fnd:
                        fname, version = fnd.group("fname", "version")
                        if (basedir+subdir+fname).lower() not in exclude:
#                            print "XXX " + basedir+subdir+fname
                            outf.write("%s=%s:%s\n" % (basedir[7:]+subdir+fname, version, baseurl+subdir+fname))
#                            print "%s=%s:%s\n" % (basedir[7:]+subdir+fname, version, baseurl+subdir+fname)
                        else:
                            print "EXC " + basedir+subdir+fname
                            continue

                    fnd = dir_re.match(line)
                    if fnd:
                        dname = fnd.group("dname")
                        if (basedir+subdir+dname).lower() not in exclude:
#                            print "going to " + basedir+subdir+dname
                            #print baseurl, basedir, subdir+dname+"/"
#                            print dname
                            listfiles(baseurl, basedir, subdir+dname+"/")
                        else:
                            print "EXC " + basedir+subdir+dname
                f.close()

outf = open(r"../whatsup.txt", "wt")

listfiles("./", "orange/", "")
listfiles("./OrangeWidgets/Genomics/", "orange/OrangeWidgets/Genomics/", "")
outf.close()
