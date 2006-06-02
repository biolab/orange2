from HTMLParser import HTMLParser
from operator import add
import re, sys, os

#execfile("constants.py")

toCopy = ["ofb", "reference", "modules", "datasets", "style.css", "Orange.hs"]

class TOCEntry:
    def __init__(self, title, url, dummy = 0):
        self.title = title
        self.url = url
        self.subentries = []
        self.dummyLink = dummy

    def __repr__(self):
        return self.title

index = {}
TOCRoot = TOCEntry("Index", "./default.htm")
TOCStack = [TOCRoot]

files = [("../style.css", None)]

IDs = {}
IDCounter = 0

def removeNewlines(s): return s.replace("\n", " __HTMLHELP_NEWLINE ")
def addNewlines(s): return s.replace(" __HTMLHELP_NEWLINE ", "\n")

def copydir(dir):
    if os.path.isdir(dir):
        tryMk("processed/" + dir)
        for f in os.listdir(dir):
            copydir(dir+"/"+f)
    else:
        file("processed/"+dir, "wb").write(file(dir, "rb").read())

def printToc(e, s=0):
    print " "*(s*2), e.title
    for se in e.subentries:
        printToc(se, s+1)
        
def tryMk(dir):
    try:
        os.mkdir(dir)
    except:
        pass

def newID(s):
    global IDCounter
    IDs[s] = "JH%i" % IDCounter
    IDCounter += 1

class IndexStore:
    def __init__(self, aname, entries = None):
        self.name = aname
        if entries:
            self.entries = entries[:]
        else:
            self.entries = []
        self.subentries = {}

def addIndex(indexStoreList, name, title=None, filename=None, counter = None):
    if title:
        cateS = IndexStore(name, [(title, filename, counter)])
        newID("%s#HH%i" % (filename, counter))
    else:
        cateS = IndexStore(name)
    indexStoreList.setdefault(name.lower(), cateS)
    return cateS

def findIndices(page, filename, title):
    lowpage = page.lower()
    outp = []
    lastout = 0
    counter = 0
    lidx = 0
    H2Entry = None
    re_idx = re.compile(r'<index(\s+name\s*=\s*(?P<name>("[^"]*"|0)))?>')
    re_h2 = re.compile(r'<h2(\s+toc\s*=\s*(?P<toc>("[^"]*"|0)))?>')
    re_h3 = re.compile(r'<h3(\s+toc\s*=\s*(?P<toc>("[^"]*"|0)))?>')
    while 1:
        idxm = re_idx.search(lowpage, lidx)
        indx = idxm and idxm.start() or -1

        h2m = re_h2.search(lowpage, lidx)
        h2 = h2m and h2m.start() or -1
        
        h3m = re_h3.search(lowpage, lidx)
        h3 = h3m and h3m.start() or -1

        indices = filter(lambda x:x>=0, (indx, h2, h3))
        if not indices:
            break
        
        idx = min(indices)
        if idx == indx:
            nameg = idxm.group("name")
            eidx = lowpage.find("</index>", idx)
            
            nextidxm = re_idx.search(lowpage, indx+5)
            missingendtag = nextidxm and nextidxm.start() < eidx

            if nameg:
                b, e = idxm.span("name")
                name = addNewlines(page[b+1:e-1])
            else:
                if missingendtag or eidx == -1:
                    print "Warning: missing end of index in '%s'" % filename
                    lidx = idxm.end()
                    continue
                if eidx - idxm.end() > 50:
                    print "Warning: suspiciously long index in '%s'" % filename
                    lidx = indxm.end()
                    continue
                name = addNewlines(page[idxm.end():eidx])

            name = name.replace("\n", " ")
            outp.append(page[lastout:idx])
            outp.append('<a name="HH%i">' % counter)

            if eidx > -1 and (not nextidxm or eidx < nextidxm.start()):
                outp.append(page[idxm.end():eidx])
                lastout = lidx = eidx+8
            else:
                lastout = lidx = idxm.end()

            if "+" in name:
                cate, name = name.split("+")
                cateS = addIndex(index, cate)
                addIndex(index, name, title, filename, counter)
                addIndex(cateS.subentries, name, title, filename, counter)
            elif "/" in name:
                cate, name = name.split("/")
                cateS = addIndex(index, cate)
                addIndex(cateS.subentries, name, title, filename, counter)
            else:
                addIndex(index, name, title, filename, counter)
            counter += 1

        else:
            ht = idx==h2 and "h2" or "h3"
            mo = idx==h2 and h2m or h3m

            skip = 0
            toc = mo.group("toc")
            if toc:
                if toc == "0":
                    skip = 1
                else:
                    b, e = mo.span("toc")
                    name = addNewlines(page[b+1:e-1])
            else:
                eidx = lowpage.find("</%s>" % ht, idx)
                if eidx == -1:
                    print "Warning: missing end of %s in '%s'" % (ht, filename)
                    break
                if eidx - idx > 50:
                    print "Warning: suspiciously long %s in '%s'" % (ht, filename)
                    lidx = idx + 4
                    continue

                name = addNewlines(page[idx+4:eidx])
            
            outp.append(page[lastout:idx])
            if not skip:
                outp.append('<a name="HH%i">' % counter)
            outp.append("<%s>" % ht)
            lastout = lidx = mo.end()

            if not skip:
                newEntry = TOCEntry(name, "%s#HH%i" % (filename, counter))
                newID("%s#HH%i" % (filename, counter))
                if ht == "h2":
                    TOCStack[-1].subentries.append(newEntry)
                    H2Entry = newEntry
                else:
                    (H2Entry or TOCStack[-1]).subentries.append(newEntry)
                counter += 1

    outp.append(page[lastout:])
    return reduce(add, outp)



def writeIndexHH_store(hhk, indexStoreList):
    hhk.write("\n<UL>")
    for indexStore in indexStoreList.values():
        if indexStore.entries or indexStore.subentries:
            hhk.write(hhkentry % indexStore.name)
            if indexStore.entries:
                for entry in indexStore.entries:
                    hhk.write(hhksubentry % entry)
            else:
                for substore in indexStore.subentries.values():
                    for subentry in substore.entries:
                        hhk.write(hhksubentry % subentry)
            hhk.write(hhkendentry)
            if indexStore.subentries:
                writeIndexHH_store(hhk, indexStore.subentries)
    hhk.write("\n</UL>")
    
def writeIndexHH():
    hhk = file("processed/orange.hhk", "w")
    hhk.write(hhkhead)
    writeIndexHH_store(hhk, index)
    hhk.close()

def writeTocHHRec(hhc, node, l=0):
    spaces = " "*(l*4)
    hhc.write(hhcentry % {"spc": spaces, "name": node.title, "file": node.url})
    if node.subentries:
        hhc.write(spaces + "<UL>\n")
        for s in node.subentries:
            writeTocHHRec(hhc, s, l+1)
        hhc.write(spaces + "</UL>\n\n")
    
def writeTocHH():
    hhc = file("processed/orange.hhc", "w")
    hhc.write(hhchead)
    for s in TOCRoot.subentries:
        writeTocHHRec(hhc, s, 0)
    hhc.close()
    
def writeHHP():
    hhp = file("processed/orange.hhp", "w")
    hhp.write(hhphead)
    hhp.write("[FILES]\n")
    for filename, title in files:
        hhp.write(filename+"\n")
    hhp.close()


def writeIndexJH():
    idx = file("processed/OrangeIndex.xml", "w")
    idx.write(jh_idx)
    for name, entries in index.values():
        idx.write("    <indexitem>%s\n" % name)
        for entry in entries:
            idx.write("         <indexitem target=%s>%s</indexitem>\n" % (IDs["%s#HH%i" % (entry[1], entry[2])], entry[0]))
        idx.write("    </indexitem>\n\n")
    idx.write("</index>\n")
    idx.close()


def writeTocJHRec(toc, node, l=0):
    spaces = " "*(l*4)
    tsub = filter(lambda n:not n.dummyLink, node.subentries)

    toc.write('%s<tocitem text="%s"' % (spaces, node.title))
    if not node.dummyLink:
        toc.write(' target="%s"' % IDs[node.url])
    if node.subentries:
        toc.write(">\n")
        for s in node.subentries:
            writeTocJHRec(toc, s, l+1)
        toc.write(spaces+"</tocitem>\n\n")
    else:
        toc.write("/>\n")

def writeTocJH():
    toc = file("processed/OrangeTOC.xml", "w")
    toc.write(jh_toc)
    for s in TOCRoot.subentries:
        writeTocJHRec(toc, s, 0)
    toc.write("</toc>\n")
    toc.close()

def writeMapJH():
    map = file("processed/OrangeMap.jhm", "w")
    map.write(jh_map)
    for url, id in IDs.items():
        if title:
            map.write('    <mapID target="%s" url="%s" />\n' % (id, url))
    map.write("</map>")
    map.close()

def main():
    global TOCStack, title
    tryMk("processed")
    for dir in toCopy:
        copydir(dir)

    for dir, contname in [(".", "Index"), ("modules", "Module"), ("ofb", "Orange for Beginners"), ("reference", "Reference Guide")]:
        tryMk("processed/"+dir)

        newentry = TOCEntry(contname, dir+"/default.htm")
        newID(dir+"/default.htm")
        TOCStack = [TOCStack[0]]

        for fle in file(dir+"/hhstructure.txt"):
            level = 0
            while fle[level] == "\t":
                level += 1

            arrow = fle.find("--->")
            title = fle[:arrow].strip()
            fn = fle[arrow+4:].strip()

            dummyLink = fn[0] == "+"
            if dummyLink:
                fn = fn[1:]
            filename = dir+"/"+fn

            newentry = TOCEntry(title, filename, dummyLink)

            if level > len(TOCStack)-1:
                print "Error in '%s/hhstructure.txt' (invalid identation in line '%s')" % (dir, fle.strip())
                sys.exit()
            TOCStack = TOCStack[:level+1]
            TOCStack[-1].subentries.append(newentry)
            TOCStack.append(newentry)

            #print "Processing %s" % filename

            files.append((filename, title))
            newID(filename)

            l = removeNewlines(open(filename).read())
            page = findIndices(l, filename, title)

            page = addNewlines(page)
            file("processed/%s/%s" % (dir, fn), "w").write(page)


    writeIndexHH()
    writeTocHH()
    writeHHP()

##    writeIndexJH()
##    writeTocJH()
##    writeMapJH()


hhphead = """
[OPTIONS]
Compiled file=orange.chm
Contents file=orange.hhc
Default topic=./default.htm
Display compile progress=No
Full text search stop list file=../stop.stp
Full-text search=Yes
Index file=orange.hhk
Language=0x409
Title=Orange Documentation
"""

hhchead = """
<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML//EN">
<HTML>
<HEAD>
<meta name="GENERATOR" content="Microsoft&reg; HTML Help Workshop 4.1">
<!-- Sitemap 1.0 -->
</HEAD><BODY>
<OBJECT type="text/site properties">
    <param name="Window Styles" value="0x801227">
    <param name="ImageType" value="Folder">
</OBJECT>
<UL>
"""

hhcentry = """%(spc)s<LI><OBJECT type="text/sitemap">
%(spc)s    <param name="Name" value="%(name)s">
%(spc)s    <param name="Local" value="%(file)s">
%(spc)s</OBJECT>
"""


hhkhead = """
<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML//EN">
<HTML>
<HEAD>
<meta name="GENERATOR" content="Microsoft&reg; HTML Help Workshop 4.1">
<!-- Sitemap 1.0 -->
</HEAD><BODY>
"""

hhkentry = """<LI><OBJECT type="text/sitemap">
    <param name="Name" value="%s">"""

hhksubentry = """
    <param name="Name" value="%s">
    <param name="Local" value="%s#HH%i">
"""

hhkendentry = "\n</OBJECT>\n"


jh_idx = """
<?xml version='1.0' encoding='ISO-8859-1' ?>
<!DOCTYPE index
  PUBLIC "-//Sun Microsystems Inc.//DTD JavaHelp Index Version 1.0//EN" "index_1_0.dtd">

<index version="1.0">
"""

jh_toc = """
<?xml version='1.0' encoding='ISO-8859-1' ?>
<!DOCTYPE toc
  PUBLIC "-//Sun Microsystems Inc.//DTD JavaHelp Index Version 1.0//EN" "toc_2_0.dtd">

<toc version="1.0">
"""

jh_map = """
<?xml version='1.0' encoding='ISO-8859-1' ?>
<!DOCTYPE map
  PUBLIC "-//Sun Microsystems Inc.//DTD JavaHelp Map Version 1.0//EN"
  "http://java.sun.com/products/javahelp/map_1_0.dtd">
  
<map version="1.0">
"""

main()