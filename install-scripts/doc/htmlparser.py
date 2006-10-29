from HTMLParser import HTMLParser
from operator import add
import re, sys, os

PROCESSED = "processed/"
#execfile("constants.py")

toCopy = ["ofb", "reference", "modules", "datasets", "style.css"]

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

files = []#("style.css", None)]

IDs = {}
IDCounter = 0

def removeNewlines(s): return s # s.replace("\n", " __HTMLHELP_NEWLINE ")
def addNewlines(s): return s # s.replace(" __HTMLHELP_NEWLINE ", "\n")

def copydir(dir):
    if os.path.isdir(dir):
        tryMk(PROCESSED + dir)
        for f in os.listdir(dir):
            copydir(dir+"/"+f)
    else:
        file(PROCESSED+dir, "wb").write(file(dir, "rb").read())

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
    cateS = indexStoreList.get(name.lower(), None)
    if not cateS:
        cateS = IndexStore(name)
        indexStoreList[name.lower()] = cateS
    
    if title:
        cateS.entries.append((title, filename, counter))
        newID("%s#HH%i" % (filename, counter))

    return cateS

htags = [tg % i for tg in ["<h%i>", "<h%i "] for i in range(5)]

def findLastTag(outp, tag, start = None):
    ri = -1
    if tag == "h":
        tags = htags
    else:
        tags = "<%s>" % tag, "<%s " % tag
        
    for l in range(start or len(outp)-1, -1, -1):
        ri = max([outp[l].lower().rfind(tg) for tg in tags])
        if ri > -1:
            return l, ri

    return -1, -1        
        
    
def addAName(outp, aname, lasttag=None):
    if not lasttag or lasttag == "here":
        ri = -1
    elif lasttag == "head":
        l, ri = findLastTag(outp, "h")
    elif lasttag == "phead":
        l, ri = findLastTag(outp, "p")
        lh, rih = findLastTag(outp, "h")
        if rih > -1:
            lp2, rip2 = findLastTag(outp, "p", l-1)
            if ri == -1 or lp2 < lh:
                l, ri = lh, rih
    else:
        l, ri = findLastTag(outp, lasttag)

    if ri > -1:
        outp[l] = outp[l][:ri] + aname + outp[l][ri:]
    else:
        outp.append(aname)
        
def findIndices(page, filename, title):
    lowpage = page.lower()
    outp = []
    lastout = 0
    counter = 0
    lidx = 0
    H2Entry = None
#    re_idx = re.compile(r'<index(\s+name\s*=\s*(?P<name>("[^"]*"|0)))?>')
    re_idx = re.compile(r'<index(?P<options>\s+[^>]+)?>', re.IGNORECASE + re.DOTALL)
    re_h2 = re.compile(r'<h2(\s+toc\s*=\s*(?P<toc>("[^"]*"|0)))?>', re.IGNORECASE + re.DOTALL)
    re_h3 = re.compile(r'<h3(\s+toc\s*=\s*(?P<toc>("[^"]*"|0)))?>', re.IGNORECASE + re.DOTALL)

    re_opt_name = re.compile(r'\s*name\s*=\s*(?P<name>("[^"]*"|0))\s*', re.IGNORECASE + re.DOTALL)
    re_opt_pos = re.compile(r'\s*pos\s*=\s*(?P<pos>("[^"]*"))\s*', re.IGNORECASE + re.DOTALL)

    while 1:
        idxm = re_idx.search(lowpage, lidx)
        indx = idxm and idxm.start() or -1

        if processHeaders:
            h2m = re_h2.search(lowpage, lidx)
            h2 = h2m and h2m.start() or -1
            
            h3m = re_h3.search(lowpage, lidx)
            h3 = h3m and h3m.start() or -1
        else:
            h2 = h3 = -1

        indices = filter(lambda x:x>=0, (indx, h2, h3))
        if not indices:
            break
        
        idx = min(indices)
        if idx == indx:
            optionsg = idxm.group("options")
            posg = "phead"
            name = ""
            if optionsg:
                mg = re_opt_name.match(optionsg)
                if mg:
                    nameg = mg.group("name")
                    begopt = idxm.span("options")[0]
                    b, e = mg.span("name")
                    name = addNewlines(page[begopt+b+1:begopt+e-1])
                
                mg = re_opt_pos.match(optionsg)
                if mg:
                    posg = mg.group("posg")
                    if posg == "here":
                        posg = None
                    
            eidx = lowpage.find("</index>", idx)
            
            nextidxm = re_idx.search(lowpage, indx+5)
            missingendtag = nextidxm and nextidxm.start() < eidx

            if not name:
                if missingendtag or eidx == -1:
                    print "Warning: missing end of index in '%s'" % filename
                    lidx = idxm.end()
                    continue
                if eidx - idxm.end() > 100:
                    print "Warning: suspiciously long index in '%s'" % filename
                    lidx = indxm.end()
                    continue
                name = addNewlines(page[idxm.end():eidx])

            name = name.replace("\n", " ")
            outp.append(page[lastout:idx])
            addAName(outp, '<a name="HH%i">' % counter, posg)

            if eidx > -1 and (not nextidxm or eidx < nextidxm.start()):
                outp.append(page[idxm.end():eidx])
                lastout = lidx = eidx+8
            else:
                lastout = lidx = idxm.end()

            if "+" in name:
                catename = name.split("+")
                if len(catename) == 2:
                    cate, name = catename
                    cateS = addIndex(index, cate)
                    addIndex(index, name, title, filename, counter)
                    addIndex(cateS.subentries, name, title, filename, counter)
                else:
                    addIndex(index, name, title, filename, counter)
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
                if eidx - idx > 100:
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
    
def writeIndexHH(outputstub):
    hhk = file(PROCESSED + "%s.hhk" % outputstub, "w")
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
    
def writeTocHH(outputstub):
    hhc = file(PROCESSED + "%s.hhc" % outputstub, "w")
    hhc.write(hhchead)
    for s in TOCRoot.subentries:
        writeTocHHRec(hhc, s, 0)
    hhc.write(hhcfoot)
    hhc.close()
    
def writeHHP(outputstub, title):
    hhp = file(PROCESSED + "%s.hhp" % outputstub, "w")
    hhp.write(hhphead % {"stub": outputstub, "title": title})
    hhp.write("[FILES]\n")
    for filename, title in files:
        hhp.write(filename+"\n")
    hhp.close()

def underspace(s):
    return s.replace("_", " ")

def createCanvasCatalogPage():
    def insertFile(cat, name):
        import os
        namep = name.replace(" ", "")
        s = "widgets/catalog/" + cat + "/" + namep
        if os.path.exists(s+".htm"):
            catalogFile.write('<td><a href="%s.htm"><img src="icons/%s.png"></a></td>\n' % (s, namep) + \
                           '<td style="padding-right: 15"><a href="%s.htm">%s</a></td>\n\n' % (s, name))
            hhFile.write("\t\t%s ---> catalog/%s/%s.htm\n" % (name, cat, namep))
        else:
            catalogFile.write('<td><img style="padding: 2;" src="icons/%s.png"></td>\n' % namep + \
                           '<td style="padding-right: 15"><FONT COLOR="#bbbbbb">%s</FONT></a></td>\n\n' % name)

    def category(cat, names):
        catalogFile.write('<tr><td COLSPAN="6"><H2>%s</H2></td></tr>\n\n\n' % cat)
        catalogFile.write('<tr>\n')
        hhFile.write("\t%s ---> catalog/default.htm\n" % cat)
        for i, name in enumerate(names):
            if i and not i % 3:
                catalogFile.write('</tr><tr>')
            insertFile(cat, name)
        catalogFile.write('</tr>')

    catalogFile = file("widgets/catalog/default.htm", "w")
    catalogFile.write(cataloghead)

    hhFile = file("widgets/hhstructure.txt", "w")
    hhFile.write("Orange Canvas ---> default.htm\n")

    category("Data", ["File", "Save", "Data Table", "Select Attributes", "Data Sampler", "Select Data", "Discretize", "Continuize", "Rank"])

    category("Visualize", ["Attribute Statistics", "Distributions", "Scatterplot", "Scatterplot Matrix",
                "Radviz", "Polyviz", "Parallel Coordinates", "Survey Plot",
                "Sieve Diagram", "Mosaic Display", "Sieve Multigram"])

    category("Associate", ["Association Rules", "Association Rules Filter", "Association Rules Viewer",
              "Example Distance", "Attribute Distance", "Distance Map", "K-means Clustering",
              "Interaction Graph", "MDS", "Hierarchical Clustering"])

    category("Classify", ["Naive Bayes", "Logistic Regression", "Majority", "k-Nearest Neighbours", "Classification Tree",
                             "C4.5", "Interactive Tree Builder", "SVM", "CN2", "Classification Tree Viewer",
                             "Classification Tree Graph", "Rules Viewer", "Nomogram"])

    category("Evaluate", ["Test Learners", "Classifications", "ROC Analysis", "Lift Curve", "Calibration Plot"])

    catalogFile.write(catalogfoot)
    
def main():
    import sys
    global processHeaders
    bookname = underspace(sys.argv[1])
    outputstub = underspace(sys.argv[2])
    
    directories = []
    i = 3
    while i < len(sys.argv):
        directories.append((underspace(sys.argv[i]), underspace(sys.argv[i+1])))
        i += 2

    if outputstub == "widgets":
        createCanvasCatalogPage()
        processHeaders = False
    else:
        processHeaders = True
    
    global TOCStack, title
    tryMk(PROCESSED[:-1])
##    for dir in toCopy:
##        copydir(dir)

    re_hrefomod = re.compile(r'href\s*=\s*"\.\.[/\\](?P<module>(ofb)|(reference)|(modules))[/\\](?P<rest>[^"]+)"', re.IGNORECASE + re.DOTALL)

    for dir, contname in directories: #[(".", "Index"), ("modules", "Module"), ("ofb", "Orange for Beginners"), ("reference", "Reference Guide")]:
        tryMk(PROCESSED + dir)
        copydir(dir)
        file(PROCESSED +dir+"/style.css", "w").write(file("style.css", "r").read())
        files.append((dir+"/style.css", None))

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
            page = re_hrefomod.sub(r'href="ms-its:\g<module>.chm::/\g<module>/\g<rest>"', page)
            page = page.replace("../style.css", "ms-its:"+outputstub+".chm::/"+dir+"/style.css")
            page = page.replace('Up: <a href="../default.htm">Orange Documentation</a>', '')

            file(PROCESSED + "%s/%s" % (dir, fn), "w").write(page)


    writeIndexHH(outputstub)
    writeTocHH(outputstub)
    writeHHP(outputstub, contname)


hhphead = """
[OPTIONS]
Binary Index=Yes
Compiled file=../%(stub)s.chm
Contents file=%(stub)s.hhc
Default topic=%(stub)s/default.htm
Display compile progress=No
Full text search stop list file=../stop.stp
Full-text search=Yes
Index file=%(stub)s.hhk
Language=0x409
Title=%(title)s
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

hhcfoot = "</UL>"

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


cataloghead = """
<LINK REL=StyleSheet HREF="ms-its:widgets.chm::/widgets/style.css" TYPE="text/css" MEDIA=screen>

<h1>Catalog of Orange Widgets</h1>

<p>Orange Widgets are building blocks of Orange's graphical user's
interface and its visual programming interface. The purpose of this
documention is to describe the basic functionality of the widgets,
and show how are they used in interaction with other widgets.</p>

<p>Widgets in Orange can be used on their own, or within a separate
application. This documention will however describe them as used
within Orange Canvas, and application which trough visual programming
allows gluing of widgets together in whatcan be anything from simple
data analysis schema to a complex explorative data analysis
application.</p>

<p>In Orange Canvas, widgets are grouped according to their
functionality. We stick to the same grouping in this documentation,
and cluster widgets accoring to their arrangement withing Canvas's
toolbar.</p>

<P>The documentation refers to the last snapshot of Orange. The
version you use might miss some stuff which is already described
here. Download the new snapshot if you need it.</P>

<table>
"""

catalogfoot = """
</table>
</body></html>
"""
main()