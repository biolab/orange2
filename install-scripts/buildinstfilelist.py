import os, re, sys, md5

basedir = sys.argv[1]
fileprefix = sys.argv[2]

if basedir[-1] != "\\":
    basedir += "\\"
#basedir = "c:\\janez\\orange\\"

print "Constructing file lists for Orange in '%s', prefix is '%s'" % (basedir, fileprefix)

exclude = [x.lower().replace("/", "\\")[:-1] for x in open(basedir+"orange\\exclude.lst", "rt").readlines()]
file_re = re.compile(r'/(?P<fname>.*)/(?P<version>.*)/(?P<date>.*)/[^/]*/')

def computeMD(filename):
    existing = open(filename, "rb")
    currmd = md5.new()
    currmd.update(existing.read())
    existing.close()
    return currmd.hexdigest()

def excluded(fname):
    fname = fname.lower()
    for ex in exclude:
        if ex==fname[:len(ex)]:
            print "Excluded %s (as %s)" % (fname, ex)
            return 1

outfs = ""
hass = ""

def buildListLow(root_dir, here_dir, there_dir, regexp, recursive):
    global outfs, hass
    
    if not os.path.exists(root_dir+here_dir):
        return
    
    whatsDownEntries = None
    directories = []
    for fle in os.listdir(root_dir+here_dir):
        tfle = root_dir+here_dir+fle
        #print there_dir+fle
        if fle == "CVS" or excluded("orange\\"+there_dir+fle):
            continue
        if os.path.isdir(tfle):
            if recursive:
                directories.append((here_dir, there_dir, fle))
        else:
            if not regexp or regexp.match(fle):
                if not whatsDownEntries:
                    outfs += '\nSetOutPath "$INSTDIR\\%s"\n' % there_dir
                    if not there_dir:
                        hass += 'FileWrite $6 "+Orange Root$\\r$\\n"\n'
                    elif there_dir[:3] == "doc" and (len(there_dir)==3 or there_dir[3]=="\\"):
                        if len(there_dir)==3:
                            hass += 'FileWrite $6 "+Orange Documentation$\\r$\\n"\n'
                    else:
                        hass += 'FileWrite $6 "+%s$\\r$\\n"\n' % there_dir[:-1]

                    entriesfile = open(root_dir+here_dir+"CVS\\Entries", "rt")
                    whatsDownEntries = {}
                    for line in entriesfile:
                        ma = file_re.match(line)
                        if ma:
                            fname, version, date = ma.groups()
                            whatsDownEntries[fname] = (there_dir+fname, version, computeMD(root_dir+here_dir+fname))
                    entriesfile.close()
                                                
                outfs += 'File "%s"\n' % tfle
                outfs += 'FileWrite $6 "%s=%s:%s$\\r$\\n"\n' % whatsDownEntries[fle]

    for here_dir, there_dir, fle in directories:
        buildListLow(root_dir, here_dir+fle+"\\", there_dir+fle+"\\", regexp, recursive)


def buildList(root, here, there, regexp, fname, recursive=1):
    global outfs, hass
    outfs = hass = ""
    buildListLow(root, here, there, regexp and re.compile(regexp, re.IGNORECASE), recursive)
    open(fileprefix+"_"+fname+".inc", "wt").write(hass+outfs)

def buildLists(rhter, fname):
    global outfs, hass
    outfs = hass = ""
    for root, here, there, regexp, recursive in rhter:
        buildListLow(root, here, there, regexp and re.compile(regexp, re.IGNORECASE), recursive)
    open(fileprefix+"_"+fname+".inc", "wt").write(hass+outfs)
        
buildList(basedir, "orange\\", "", ".*[.]pyd?\Z", "base", 0)
buildList(basedir, "orange\\orangeWidgets\\", "orangeWidgets\\", ".*[.]((py)|(png))\\Z", "widgets")
buildList(basedir, "orange\\orangeCanvas\\", "orangeCanvas\\", ".*[.]((py)|(png))\\Z", "canvas")

buildLists([(basedir, "genomics\\", "orangeWidgets\\Genomics\\", ".*[.]py\\Z", 0),
            (basedir, "genomics\\GO\\", "orangeWidgets\\Genomics\\GO\\", "", 0),
            (basedir, "genomics\\Annotation\\", "orangeWidgets\\Genomics\\Annotation\\", "", 0),
            (basedir, "genomics\\Genome Map\\", "orangeWidgets\\Genomics\\Genome Map\\", "", 0)], "genomics")

buildLists([(basedir, "orange\\doc\\", "doc\\", "style.css\\Z", 0),
            (basedir, "orange\\doc\\reference\\", "doc\\reference\\", "", 0),
            (basedir, "orange\\doc\\modules\\", "doc\\modules\\", "", 0),
            (basedir, "orange\\doc\\ofb\\", "doc\\ofb\\", "", 0)], "doc")
