import os, re, sys

basedir = sys.argv[1]
if basedir[-1] != "\\":
    basedir += "\\"
#basedir = "c:\\janez\\orange\\"

exclude = [x.lower().replace("/", "\\")[:-1] for x in open(basedir+"orange\\exclude.lst", "rt").readlines()]

def buildListLow(root_dir, here_dir, there_dir, regexp, outf, recursive):
    firstInDir = 1
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
                if firstInDir:
                    outf.write('\nSetOutPath "$INSTDIR\\%s"\n' % there_dir)
                    firstInDir = 0
                outf.write('File "%s"\n' % tfle)

    for here_dir, there_dir, fle in directories:
        buildListLow(root_dir, here_dir+fle+"\\", there_dir+fle+"\\", regexp, outf, recursive)

def buildList(root, here, there, regexp, fname, recursive=1, mode="wt"):
    outf = open(fname, mode)
    buildListLow(root, here, there, regexp and re.compile(regexp, re.IGNORECASE), outf, recursive)
    outf.close()
    
buildList(basedir, "orange\\", "", ".*[.]pyd?\Z", "files_base.inc", 0)
buildList(basedir, "orange\\orangeWidgets\\", "orangeWidgets\\", ".*[.]((py)|(png))\\Z", "files_widgets.inc")
buildList(basedir, "orange\\orangeCanvas\\", "orangeCanvas\\", ".*[.]((py)|(png))\\Z", "files_canvas.inc")

buildList(basedir, "genomics\\", "orangeWidgets\\Genomics\\", ".*[.]py\\Z", "files_genomics.inc", 0)
buildList(basedir, "genomics\\GO\\", "orangeWidgets\\Genomics\\GO\\", "", "files_genomics.inc", 0, "at")
buildList(basedir, "genomics\\Annotation\\", "orangeWidgets\\Genomics\\Annotation\\", "", "files_genomics.inc", 0, "at")
buildList(basedir, "genomics\\Genome Map\\", "orangeWidgets\\Genomics\\Genome Map\\", "", "files_genomics.inc", 0, "at")

buildList(basedir, "orange\\doc\\", "doc\\", "style.css\\Z", "files_doc.inc", 0)
buildList(basedir, "orange\\doc\\reference\\", "doc\\reference\\", "", "files_doc.inc", 0, "at")
buildList(basedir, "orange\\doc\\modules\\", "doc\\modules\\", "", "files_doc.inc", 0, "at")
buildList(basedir, "orange\\doc\\ofb\\", "doc\\ofb\\", "", "files_doc.inc", 0, "at")
