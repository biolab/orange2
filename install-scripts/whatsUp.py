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

def listgroups(basedir, exclude, basetext):
    for dir in os.listdir(basedir):
        if not os.path.isdir(os.path.join(basedir, dir)) or dir.lower() in exclude: continue
        outf.write("+" + basetext + dir + "\n")

    
outf = open(r"../whatsup.txt", "wt")


# find groups
outf.write("+Orange Root\n")
outf.write("+Orange Documentation\n")

listgroups(".\\orange", ["orangewidgets", "doc", "cvs"] , "")
listgroups(".\\orange\\orangeWidgets", ["cvs"] , "OrangeWidgets\\")      # add default groups of widgets
listgroups(".", ["orange", "source"], "OrangeWidgets\\")    # directories in the \laststable are widget groups

listfiles("orange/", "", "")
listfiles("Genomics/", "orangeWidgets/Genomics/", "")
outf.close()