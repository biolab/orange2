import os, re, sys

orangedir = sys.argv[1]
if orangedir[-1] != "\\":
    orangedir += "\\"
#orangedir = "c:\\janez\\orange\\"

exclude = [x.lower().replace("/", "\\")[:-1] for x in open(orangedir+"exclude.lst", "rt").readlines()]

def buildListLow(root_dir, here_dir, there_dir, regexp, outf, recursive):
    firstInDir = 1
    directories = []
    for fle in os.listdir(root_dir+here_dir):
        tfle = root_dir+here_dir+fle
        if fle == "CVS" or (here_dir+fle).lower() in exclude:
            continue
        if os.path.isdir(tfle):
            if recursive:
                directories.append((here_dir, there_dir, fle))
        else:
            if not regexp or regexp.match(fle):
                if firstInDir:
                    outf.write('\nSetOutPath "%s"\n' % there_dir)
                    firstInDir = 0
                outf.write('File "%s"\n' % tfle)

    for here_dir, there_dir, fle in directories:
        buildListLow(root_dir, here_dir+fle+"\\", there_dir+fle+"\\", regexp, outf, recursive)

def buildList(root, here, there, regexp, fname, recursive=1, mode="wt"):
    outf = open(fname, mode)
    buildListLow(root, here, there, regexp and re.compile(regexp, re.IGNORECASE), outf, recursive)
    outf.close()
    
buildList(orangedir, "", "$INSTDIR\\", ".*[.]pyd?\Z", "files_base.inc", 0)
buildList(orangedir, "orangeWidgets\\", "$INSTDIR\\orangeWidgets\\", ".*[.]((py)|(png))\\Z", "files_widgets.inc")
buildList(orangedir, "orangeCanvas\\", "$INSTDIR\\orangeCanvas\\", ".*[.]((py)|(png))\\Z", "files_canvas.inc")

buildList(orangedir, "orangeWidgets\\Genomics\\", "$INSTDIR\\orangeWidgets\\Genomics\\", ".*[.]py\\Z", "files_genomics.inc", 0)
buildList(orangedir, "orangeWidgets\\Genomics\\GO\\", "$INSTDIR\\orangeWidgets\\Genomics\\GO\\", "", "files_genomics.inc", 0, "at")
buildList(orangedir, "orangeWidgets\\Genomics\\Annotation\\", "$INSTDIR\\orangeWidgets\\Genomics\\Annotation\\", "", "files_genomics.inc", 0, "at")
buildList(orangedir, "orangeWidgets\\Genomics\\Genome Map\\", "$INSTDIR\\orangeWidgets\\Genomics\\Genome Map\\", "", "files_genomics.inc", 0, "at")

buildList(orangedir, "doc\\reference\\", "$INSTDIR\\doc\\reference\\", "", "files_doc.inc", 0)
buildList(orangedir, "doc\\modules\\", "$INSTDIR\\doc\\modules\\", "", "files_doc.inc", 0, "at")
buildList(orangedir, "doc\\ofb\\", "$INSTDIR\\doc\\ofb\\", "", "files_doc.inc", 0, "at")

