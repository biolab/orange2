import os, re, sys

basedir = sys.argv[1]
fileprefix = sys.argv[2]

if basedir[-1] != "\\":
    basedir += "\\"
#basedir = "c:\\janez\\orange\\"

print "Constructing file lists for Orange in '%s', prefix is '%s'" % (basedir, fileprefix)

exclude = [x.lower().replace("/", "\\")[:-1] for x in open(basedir+"orange\\exclude.lst", "rt").readlines()]
file_re = re.compile(r'/(?P<fname>.*)/(?P<version>.*)/(?P<date>.*)/[^/]*/')

def computeMD(filename):
    return "<MD>"

def buildListLow(root_dir, here_dir, there_dir, regexp, outf, recursive):
    if not os.path.exists(root_dir+here_dir):
        return
    
    whatsDownEntries = None
    directories = []
    for fle in os.listdir(root_dir+here_dir):
        tfle = root_dir+here_dir+fle
        #print there_dir+fle
        if fle == "CVS" or ("orange\\"+there_dir+fle).lower() in exclude:
            continue
        if os.path.isdir(tfle):
            if recursive:
                directories.append((here_dir, there_dir, fle))
        else:
            if not regexp or regexp.match(fle):
                if not whatsDownEntries:
                    outf.write('\nSetOutPath "$INSTDIR\\%s"\n' % there_dir)

                    entriesfile = open(root_dir+here_dir+"CVS\\Entries", "rt")
                    whatsDownEntries = {}
                    for line in entriesfile:
                        ma = file_re.match(line)
                        if ma:
                            fname, version, date = ma.groups()
                            whatsDownEntries[fname] = (there_dir+fname, version, computeMD(root_dir+here_dir+fname))
                    entriesfile.close()
                                                
                outf.write('File "%s"\n' % tfle)
                outf.write('FileWrite $6 "%s:%s=%s$\\r$\\n"\n' % whatsDownEntries[fle])

    for here_dir, there_dir, fle in directories:
        buildListLow(root_dir, here_dir+fle+"\\", there_dir+fle+"\\", regexp, outf, recursive)


def buildList(root, here, there, regexp, fname, recursive=1, mode="wt"):
    outf = open(fileprefix+"_"+fname+".inc", mode)
    buildListLow(root, here, there, regexp and re.compile(regexp, re.IGNORECASE), outf, recursive)
    outf.close()
    
buildList(basedir, "orange\\", "", ".*[.]pyd?\Z", "base", 0)
buildList(basedir, "orange\\orangeWidgets\\", "orangeWidgets\\", ".*[.]((py)|(png))\\Z", "widgets")
buildList(basedir, "orange\\orangeCanvas\\", "orangeCanvas\\", ".*[.]((py)|(png))\\Z", "canvas")

buildList(basedir, "genomics\\", "orangeWidgets\\Genomics\\", ".*[.]py\\Z", "genomics", 0)
buildList(basedir, "genomics\\GO\\", "orangeWidgets\\Genomics\\GO\\", "", "genomics", 0, "at")
buildList(basedir, "genomics\\Annotation\\", "orangeWidgets\\Genomics\\Annotation\\", "", "genomics", 0, "at")
buildList(basedir, "genomics\\Genome Map\\", "orangeWidgets\\Genomics\\Genome Map\\", "", "genomics", 0, "at")

buildList(basedir, "orange\\doc\\", "doc\\", "style.css\\Z", "doc", 0)
buildList(basedir, "orange\\doc\\reference\\", "doc\\reference\\", "", "doc", 0, "at")
buildList(basedir, "orange\\doc\\modules\\", "doc\\modules\\", "", "doc", 0, "at")
buildList(basedir, "orange\\doc\\ofb\\", "doc\\ofb\\", "", "doc", 0, "at")
