import os, re, sys

specialfiles = ["links_up", "main_links"]
#subdirs = ["modules", "ofb", "reference"]
ewtitles = {"modules": "ORANGE MODULES", "ofb": "ORANGE FOR BEGINNERS", "reference": "ORANGE REFERENCE"}

re_htmlfile = re.compile(r"(?P<stem>.*)\.html?", re.IGNORECASE)
re_href = re.compile(r'href\n?=\n?"(?P<fname>[^\][^:>]*?\.html?)', re.IGNORECASE)
#re_relativepath = re.compile(r"\.\./("+reduce(lambda x,y:x+"|"+y, ["("+x+")" for x in subdirs])+")", re.IGNORECASE)
re_title = re.compile(r'<H1>\s*(?P<title>[^"]*?)</H1>', re.IGNORECASE)
re_links = re.compile(r'<p class="links">.*?</p>', re.DOTALL)


dont_copy = ["cvs", "links_up.htm", "main_links.htm", "writing documentation.txt", "formulas"]

def copyfile(srcpath, destpath):
    f = open(srcpath, "rb")
    s = f.read()
    f.close()

    f = open(destpath, "wb")
    f.write(s)
    f.close()
    
def copystandalone(subdir):
    if not os.path.exists(standalonedir+subdir):
        os.mkdir(standalonedir+subdir)
    files = os.listdir(docdir+subdir)
    for f in files:
        if (f.lower() in dont_copy) or (not copydatasets and f.lower() == "datasets"):
            continue
        
        subname = subdir+"\\"+f
        if os.path.isfile(docdir+subname):
            copyfile(docdir+subname, standalonedir+subname)
        else:
            copystandalone(subname)


myhtmls = {}
htmls = []
# Prepares documentation on the web: puts html files in
# a list and copies everything else
def copytree(subdir):
    if not os.path.exists(aspdir+subdir):
        os.mkdir(aspdir+subdir)
    files = os.listdir(docdir+subdir)
    for f in files:
        # Mark HTMLs      
        fn = re_htmlfile.match(f)
        if fn and (fn.group("stem") not in specialfiles):
            myhtmls[f] = 1
            htmls.append((f, subdir))
            
        # copy other stuff
        else:
            subname = subdir+"\\"+f
            if os.path.isfile(docdir+subname):
                copyfile(docdir+subname, aspdir+subname)
            else:
                copytree(subname)

class ReplaceHref:
    def __init__(self, filename):
        self.filename = filename
        
    def __call__(self, matchobj):
        ms = matchobj.group("fname")
#        ms = ms.split("#")[0]
        head, tail = os.path.split(ms)
        if not tail:
            print "NO FILE: add the filename to '%s' (in '%s')" % (ms, self.filename)
        if head and head[0]=="/": #not re_relativepath.match(head):
            return matchobj.group(0)
        if not myhtmls.has_key(tail):
            print "MISSING:  %s (used in '%s')" % (ms, self.filename)
            return matchobj.group(0)
        stem, ext = os.path.splitext(ms)
        return 'href="%s.asp' % stem

class ReplaceBody:
    def __init__(self, title):
        self.title = title
    def __call__(self, matchobj):
        return matchobj.group(0) + "\n<H1>%s</H1>" % self.title

def parsefiles():
    for fname, dir in htmls:
        if dir:
            cfname = dir + "\\"+fname
        else:
            cfname = fname
        f = open(docdir + "\\" + cfname, "rt")
        fcont = f.read()
        f.close()

        tm = re_title.search(fcont)
        if tm:
            title = tm.group("title")
            fcont = re.sub(re_title, "", fcont)
        else:
            title = ""
            print "NO TITLE: %s" % cfname

        fcont = re.sub(re_href, ReplaceHref(fname), fcont)
        fcont = re.sub(re_links, "", fcont)

        f = open(aspdir + "\\" + cfname, "wt")
        f.write(fcont)
        f.close()


        aspt = asptemplate.replace("%HTMLFILENAME", fname) \
                          .replace("%PAGETITLE", title) \
                          .replace("%EXPLORETHEWEB", ewtitles.get(dir, ""))
        if dir=="ofb":
            aspt = aspt.replace("166", "222")
        if dir=="" and fname == "default.htm":
            aspt = aspt.replace("../../", "../")

        f = open(aspdir + "\\" + os.path.splitext(cfname)[0] + ".asp", "wt")
        f.write(aspt)
        f.close()        
  
    

asptemplate = """
<% @Language = "VBScript" %>
<%
MAIN            = "%HTMLFILENAME"
LINKS_TOP       = "links_up.htm"
EXPLORE_THE_WEB = "main_links.htm"
SPOTLIGHT       = "../../spotlight.htm"
LINKS_BOTTOM    = "../../links_down.htm"

TITLE = "%PAGETITLE"
IMAGE = "/images/OrangeLogo1.gif"
LOGO  = "/images/FRIlogo.gif"
EXPLORE_THE_WEB_TITLE = "%EXPLORETHEWEB"

IMAGE_HEIGHT = 85
LOGO_HEIGHT = 35
LEFTCOL_WIDTH = 166
%>

<!-- #INCLUDE virtual="/common/template.inc" -->
"""

if len(sys.argv) != 5:
    print "Usage: process_doc <dir-with-documentation> <dir-for-standalone-doc> <dir-for-web-doc> <copy-datasets>"

docdir, standalonedir, aspdir, copydatasets = sys.argv[1:]

##standalonedir, aspdir = "c:\\2", "c:\\3"

copystandalone("")
copytree("")
parsefiles()