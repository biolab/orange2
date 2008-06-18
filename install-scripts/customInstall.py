import cgi

print 'Content-type: text/html\n\n<html><HEAD><LINK REL=StyleSheet HREF="style.css" TYPE="text/css" MEDIA=screen></HEAD><BODY><CENTER>'
print '<P>&nbsp;</P>' * 5
print '<HR>'
asdf

ORANGEROOT = "c:\\inetpub\\wwwusers\\orange\\"

argflags = [("Python", " /DINCLUDEPYTHON", 1),
            ("PyQt", " /DINCLUDEPYQT", 2),
            ("PyQwt", " /DINCLUDEPYQWT", 4),
            ("qt", " /DINCLUDEQT", 8),
            ("numeric", " /DINCLUDENUMERIC", 16),
            ("pythonwin", " /DINCLUDEPYTHONWIN", 32),
            ("dot", " /DINCLUDEDOT", 64),
            ("scriptingdoc", " /DINCLUDESCRIPTDOC", 128),
            ("datasets", " /DINCLUDEDATASETS", 256),
            ("source", " /DINCLUDESOURCE", 512),
            ("genomicsdata", " /DINCLUDEGENOMICS", 1024)
           ]

args = ""
flags = 0
form = cgi.FieldStorage()
for c, f, fl in argflags:
    if form.getvalue(c, ""):
        args += f
        flags += fl

print 'Content-type: text/html\n\n<html><HEAD><LINK REL=StyleSheet HREF="style.css" TYPE="text/css" MEDIA=screen></HEAD><BODY><CENTER>'
print '<P>&nbsp;</P>' * 5
print '<HR>'

if flags == 0x1ff and 0:
    print '<P>You have chosen the standard Orange installation with Python.</P>'
    print '<P>Click <a href="../download/Orange-complete.exe">here</a> to get it.'

elif flags == 0x80:
    print '<P>You have chosen the standard Orange installation.</P>'
    print '<P>Click <a href="../download/Orange-standard.exe">here</a> for download.'

else:
    import os, time
    
    os.chdir(ORANGEROOT+"download\\custom")
    filedir = ("%i" + ("-%2.2i" * 5)) % time.localtime()[:6]
    if os.path.exists(filedir):
        app = 1
        while os.path.exists(filedir + "-%i" % app):
            app += 1
        filedir += "%i" % app
    filename = "Orange-custom.exe"

    os.mkdir(filedir)

    os.chdir(ORANGEROOT + "scripts")
    args += ' /DORANGEDIR='+ORANGEROOT+'download\\lastStable\\orange'
    args += ' /DCWD=%s' % os.getcwd()
    args += ' /DOUTFILENAME='+ORANGEROOT+'download\\custom\\%s\\%s' % (filedir, filename)
    
    args += ' install3.nsi'
    os.spawnv(os.P_WAIT, "C:\\Program Files\\NSIS\\makensis.exe", [args])
#    if os.system('"C:\Program Files\NSIS\makensis.exe"' + args):
#    	print "<P>Error generating installation program; system administrator will be notified.</P>"
#    else:
#    	print '<P>Your installation file is ready for <a href="//www.ailab.si/orange/download/custom/%s/%s">download</a>.</P><P>(The file will disappear in approximately two hours.)' % (filedir, filename)
#    print flags, '<P>%s</P>' % args

print '<P><HR></CENTER>'

