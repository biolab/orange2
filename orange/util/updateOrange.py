import os, re, orange, httplib, urllib

re_vline = re.compile(r'(?P<fname>.*)=(?P<version>.*)')
re_widget = re.compile(r'OrangeWidgets/(?P<category>.*)/.*')

downfile = os.path.dirname(orange.__file__) + "/whatsdown.txt"
httpconnection = httplib.HTTPConnection('localhost')
updateGroups = []
dontUpdateGroups = []

def readVersionFile(data):
    versions = {}
    for line in data:
        if line:
            if line[0] == "+":
                updateGroups.append(line[1:-1])
            elif line[0] == "-":
                dontUpdateGroups.append(line[1:-1])
            else:
                fnd = re_vline.match(line)
                if fnd:
                    fname, version = fnd.group("fname", "version")
                    versions[fname] = [int(x) for x in version.split(".")]
    return versions

def writeVersionFile():
    vf = open(downfile, "wt")
    itms = downstuff.items()
    itms.sort(lambda x,y:cmp(x[0], y[0]))
    for g in updateGroups:
        vf.write("+%s\n" % g)
    for g in dontUpdateGroups:
        vf.write("-%s\n" % g)
    for fname, version in itms:
        vf.write("%s=%s\n" % (fname, reduce(lambda x,y:x+"."+y, [`x` for x in version])))
    vf.close()

def download(fname):
    httpconnection.request("GET", urllib.quote(fname))
    r = httpconnection.getresponse()
    if r.status != 200:
        raise "Got '%s' while downloading '%s'" % (r.reason, fname)
    return r.read()
    
def updatefile(fname, version):
    dname = os.path.dirname(fname)
    if dname and not os.path.exists(dname):
        os.makedirs(dname)

    print "downloading %s" % fname
    newscript = download("/orangeUpdate/"+fname)
    
    if not os.path.exists("test/"+os.path.dirname(fname)):
        os.makedirs("test/"+os.path.dirname(fname))
    nf = open("test/"+fname, "wt")
    nf.write(newscript)
    nf.close()

    downstuff[fname] = version
    writeVersionFile()

upstuff = readVersionFile(download("/whatsup.txt").split())
vf = open(downfile)
downstuff = readVersionFile(vf.readlines())
vf.close()

itms = upstuff.items()
itms.sort(lambda x,y:cmp(x[0], y[0]))
for fname, version in itms:
    if downstuff.has_key(fname):
        if downstuff[fname] < upstuff[fname]:
            updatefile(fname, version)
    else:
        fnd = re_widget.match(fname)
        if fnd:
            category = fnd.group("category")
            if category in updateGroups:
                updatefile(fname, version)
            elif category not in dontUpdateGroups:
                updatefile(fname, version)
        else:
            updatefile(fname, version)
