from operator import add
from operator import add
import re, sys, os, sets

class TOCEntry:
    def __init__(self, title, url):
        self.title = title
        self.url = url
        self.subentries = []

def copydir(src, target):
    if os.path.isdir(src):
        tryMk(target)
        for f in os.listdir(src):
            copydir(src+"/"+f, target+"/"+f)
    else:
        file(target, "wb").write(file(src, "rb").read())


def tryMk(dir):
    try:
        os.mkdir(dir)
    except:
        pass


class IndexStore:
    def __init__(self, aname, entries = None):
        self.name = aname
        if entries:
            self.entries = entries[:]
        else:
            self.entries = []
        self.subentries = {}


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
        
    
def addIndex(indexStoreList, name, title=None, filename=None, counter = None):
    cateS = indexStoreList.get(name.lower(), None)
    if not cateS:
        cateS = IndexStore(name)
        indexStoreList[name.lower()] = cateS
    
    if title:
        cateS.entries.append((title, filename, counter))

    return cateS


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

        
def findIndices(filename, title):
    page = open(filename).read()
    lowpage = page.lower()
    outp = []
    lastout = 0
    counter = 0
    lidx = 0
    H2Entry = None
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
                    name = page[begopt+b+1:begopt+e-1]
                
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
                name = page[idxm.end():eidx]

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
                    name = page[b+1:e-1]
            else:
                eidx = lowpage.find("</%s>" % ht, idx)
                if eidx == -1:
                    print "Warning: missing end of %s in '%s'" % (ht, filename)
                    break
                if eidx - idx > 100:
                    print "Warning: suspiciously long %s in '%s'" % (ht, filename)
                    lidx = idx + 4
                    continue

                name = page[idx+4:eidx]
            
            outp.append(page[lastout:idx])
            if not skip:
                outp.append('<a name="HH%i">' % counter)
            outp.append("<%s>" % ht)
            lastout = lidx = mo.end()

            if not skip:
                newEntry = TOCEntry(name, "%s#HH%i" % (filename, counter))
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
    hhk = file("%s.hhk" % outputstub, "w")
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
    hhc = file("%s.hhc" % outputstub, "w")
    hhc.write(hhchead)
    for s in TOCRoot.subentries:
        writeTocHHRec(hhc, s, 0)
    hhc.write(hhcfoot)

    
def writeHHP(outputstub, title):
    hhp = file("%s.hhp" % outputstub, "w")
    hhp.write(hhphead % {"stub": outputstub, "title": title})
    hhp.write("[FILES]\n" + "\n".join(files))



def underspace(s):
    return s.replace("_", " ")

def main():
    global processHeaders, outputstub, TOCStack, title, files, index, TOCRoot, TOCStack
    
    bookname = ""
    dir = outputstub = underspace(sys.argv[1])
    
    processHeaders = outputstub != "catalog"

    files = sets.Set()
    files.add(dir+"/style.css")

    index = {}
    TOCRoot = TOCEntry("", dir + "/default.htm")
    TOCStack = [TOCRoot]

    re_hrefomod = re.compile(r'href\s*=\s*"\.\.[/\\](?P<module>(ofb)|(reference)|(modules))[/\\](?P<rest>[^"]+)"', re.IGNORECASE + re.DOTALL)
    
    for fle in file(dir+"/hhstructure.txt"):
        fle = fle.rstrip(" \n\r")
        if not fle:
            continue
        
        level = 0
        while fle[level] == "\t":
            level += 1

        arrow = fle.find("--->")
        title = fle[:arrow].strip()
        fn = fle[arrow+4:].strip()
        if not TOCRoot.title:
            TOCRoot.title = title
            

        filename = dir+"/"+fn

        newentry = TOCEntry(title, filename)

        if level > len(TOCStack)-1:
            print "Error in '%s/hhstructure.txt' (invalid identation in line '%s')" % (dir, fle.strip())
            sys.exit()
        TOCStack = TOCStack[:level+1]
        TOCStack[-1].subentries.append(newentry)
        TOCStack.append(newentry)

        if not filename in files:
            files.add(filename)

            page = findIndices(filename, title)
            page = re_hrefomod.sub(r'href="ms-its:\g<module>.chm::/\g<module>/\g<rest>"', page)
            page = page.replace("../style.css", "ms-its:"+outputstub+".chm::/"+dir+"/style.css")
            page = page.replace('Up: <a href="../default.htm">Orange Documentation</a>', '')

            file(filename, "w").write(page)    


    writeIndexHH(outputstub)
    writeTocHH(outputstub)
    writeHHP(outputstub, TOCRoot.title)

    anoun = False
    cfiles = sets.Set([f.lower() for f in files] + [dir+"/path.htm", dir+"/links.htm"])
    dir = dir.lower()
    if processHeaders:
        for fn in os.listdir(dir):
            if fn.lower()[-4:] == ".htm" and dir+"/"+fn.lower() not in cfiles:
                if not anoun:
                    print "\nFiles that are not referenced in hhstructure.txt:"
                    anoun = True
                print "  "+fn
    else:
        for dr in os.listdir(dir):
            adr = (dir+"/"+dr).lower()
            if os.path.isdir(adr):
                for fn in os.listdir(adr):
                    if fn.lower()[-4:] == ".htm" and adr+"/"+fn.lower() not in cfiles:
                        if not anoun:
                            print "\nFiles that are not used (possibly due to name mismatches):"
                            anoun = True
                        print "  "+fn


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

main()