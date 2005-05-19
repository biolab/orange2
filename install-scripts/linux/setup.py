#!/usr/bin/env python
# To use:
#       python setup.py install

# all kind of checks

# Change this to something else than ADDVERSION
OrangeVer="ADDVERSION"

# got* is used to gather information of the system and print it out at the end
gotPython = '';
gotQt = '';
gotPyQt = '';
gotNumeric = '';
gotGsl = '';
gotGcc = '';
gotQwt = '';

try:
    import sys,commands,traceback,os
    from distutils import sysconfig
    from distutils.core import setup,Extension
    from glob import glob
except:
    traceback.print_exc()
    print "Unable to import python distutils."
    print "You may want to install the python-dev package on your distribution."
    sys.exit(1)

gotPython = sys.version.split()[0]

if not hasattr(sys, 'version_info') or sys.version_info < (2,3,0,'alpha',0):
    raise SystemExit, "Python 2.3 or later is required to build Orange."

if os.geteuid() != 0:
    print "This script should be run as superuser!"
    sys.exit(1)

try:
    import qt,pyqtconfig
except:
    print "NOTE: Python Qt not installed, OrangeCanvas and OrangeWidgets will not work."
    print "You can get it at: http://www.riverbankcomputing.co.uk/pyqt/index.php"

try:
    gotPyQt = pyqtconfig.Configuration().pyqt_version_str;
except:
    gotPyQt = ''

try:
    import sipconfig
    tmp = "%x" % sipconfig.Configuration().qt_version
    gotPy = tmp.replace('0','.')
except:
    print "Sipconfig not found, Qt version could not be found!"
    gotPy = ''

try:
    import Numeric
except:
    print "Numeric Python should be installed!"
    print "You can get it at: http://numeric.scipy.org/"
    sys.exit(1)

try:
    import numeric_version
    gotNumeric = numeric_version.version
except:
    print "Can not determine Numeric version!"
    gotNumeric = "n/a"

if os.system("gsl-config --prefix > /dev/null 2>&1") != 0:
    print "GSL should be installed!"
    print "You can get it at: http://www.gnu.org/software/gsl/"
    sys.exit(1)

try:
    import qwt
    gotQwt = "n/a"
except:
    print "PyQwt not installed!"
    print "You can get it at: http://pyqwt.sourceforge.net/"
    
# catching version of GSL
try:
    import popen2
    (stdout_err, stdin) = popen2.popen4("gsl-config --version");
    gotGsl = stdout_err.readlines()[0]
except:
    print "Can not determine GSL version!"
    gotGsl = "n/a"
    
if os.system("gcc --version > /dev/null 2>&1") != 0:
    print "GCC should be installed!"
    sys.exit(1)

# catching version of GCC
try:
    (stdout_err, stdin) = popen2.popen4("gcc --version");
    tmp = stdout_err.readlines()[0]
    gotGcc = tmp.split()[2]
except:
    print "Can not determine GCC version!"
    gotGcc = "n/a"

if OrangeVer is "ADDVERSION":
    print "Version should be added manually (edit setup.py and replace ADDVERSION)"
    sys.exit(1)

if "FreeBSD" in sys.version:
    HostOS="FreeBSD"
elif "Linux" in sys.version:
    HostOS="Linux"
else:
    HostOS="unknown"

# creating custom commands
# uninstall deletes everything which was installed
from distutils.core import Command
from distutils.command.install import install

class uninstall(Command):
    description = "uninstall current version"

    user_options = [('orangepath=', None, "Orange installation path"),
                    ('docpath=', None, "Orange documentaiton path"),
                    ('libpath=', None, "Orange library path")]

    def initialize_options (self):
        self.orangepath = OrangeInstallDir
        self.docpath = OrangeInstallDoc
        self.libpath = OrangeInstallLib
        
    def finalize_options(self):
        if self.orangepath is None:
            self.orangepath = OrangeInstallDir
    
    def run(self):
        self.orangepath = os.path.join(sys.prefix, self.orangepath)
        print "Removing installation directory "+self.orangepath+" ...",
        self.rmdir(self.orangepath)
        os.rmdir(self.orangepath)
        print "done!"
        print "Removing documentation directory "+self.docpath+" ...",
        self.rmdir(self.docpath)
        os.rmdir(self.docpath)
        print "done!"
        print "Removing symbolic links for the orange libraries ...",
        for orngLib in OrangeLibList:
            os.remove(os.path.join(self.libpath, "lib"+orngLib))
        print "done!"

    def rmdir(self,top):
        for root, dirs, files in os.walk(top, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

# compile goes to orange/source directory and compiles orange
class compile(Command):
    # add Different make programs support"
    description = "compiles Orange"

    user_options = [('coptions=', None, "special compiling options")]

    def initialize_options(self):
        self.coptions = None
        self.makeCmd = None

    def finalize_options(self):
        if self.coptions is None:
            print "Default compiler options are taken..."
        if HostOS is "FreeBSD":
            self.makeCmd = "gmake"
        else:
            self.makeCmd = "make"
            
    def run(self):
        #compile Orange with make files
        SourceDir = os.path.join("source")
        os.chdir(SourceDir)
        print "Compiling... this might take a while, logging into compiling.log...",
        make=self.makeCmd+"> ../compiling.log"
        retval = os.system(make)
        if retval != 0:
            print "Compiling Orange failed... exiting!"
            sys.exit(1)
        print "done"
        os.chdir(BaseDir)

# install is 'normal distutils' install, after installation symlinks libraries
# and calles ldconfig
class install_wrap(install):
    description = "Orange specific installation"

    user_options = install.user_options

    def run(self):
        install.run(self)

        print "Linking libraries...",
        for currentLib in OrangeLibList:
            try:
                os.symlink(os.path.join(sys.prefix,
                                        OrangeInstallDir,currentLib),
                           os.path.join(OrangeInstallLib,"lib"+currentLib))
            except:
                print "problems with "+currentLib+"... ignoring"
                
        os.system("/sbin/ldconfig")
        print "success"
        print "Creating path file...",
        pth = os.path.join(sys.prefix,OrangeInstallDir,"..","orange.pth")
        os.remove(pth)

        fo = file(pth,"w+")
        fo.write(os.path.join(sys.prefix,OrangeInstallDir)+"\n")
        for root,dirs,files in os.walk(os.path.join(sys.prefix,
                                                    OrangeInstallDir)):
            for name in dirs:
                fo.write(os.path.join(root,name)+"\n")
        fo.close()
        print "success"
        print ""
        print "Python version: "+gotPython
        print "PyQt version: "+gotPyQt
        print "Qt version: "+gotPy
        print "Numeric version: "+gotNumeric
        print "Qwt version: "+gotQwt
        print "GCC version: "+gotGcc
        print "Gsl version: "+gotGsl
        
        print "Orange installation dir: "+sys.prefix+OrangeInstallDir
        print "Orange documentation dir: "+OrangeInstallDoc
        print "Orange library links in: "+OrangeInstallLib
        print ""
        print "To uninstall Orange type:"
        print "    python "+OrangeInstallDoc+"/setup.py uninstall"
        print ""
        print "It will remove Orange, Orange documentation and links to Orange libraries"
            
# preparing data for Distutils

PythonVer = "python"+sys.version[:3]
OrangeInstallDir = os.path.join("lib", PythonVer, "site-packages", "orange")
OrangeInstallDoc = os.path.join(sys.prefix, "share", "doc",
                                OrangeVer)
OrangeInstallLib = os.path.join(sys.prefix, "lib")

OrangeLibList = ['orange.so','orangene.so','orangeom.so','statc.so','corn.so']
        
BaseDir = os.getcwd()
OrangeDirs = ["orange", "orange.OrangeCanvas", "orange.OrangeWidgets"]
OrangeWidgetsDirs = []
for root,dirs,files in os.walk(os.path.join("OrangeWidgets"), topdown=False):
    if 'CVS' in dirs:
        dirs.remove('CVS')
    for name in dirs:
	tmp = "orange."+root+"."+name
        OrangeWidgetsDirs.append(tmp.replace('/','.'))
OrangeDirs += OrangeWidgetsDirs

packages =  OrangeDirs
package_dir = {"orange" : ""}

OrangeLibs = []
for currentLib in OrangeLibList:
    OrangeLibs += [os.path.join(currentLib)]

OrangeWidgetIcons = glob(os.path.join("OrangeWidgets", "icons", "*.png"))
OrangeCanvasIcons = glob(os.path.join("OrangeCanvas",  "icons", "*.png"))

data_files = [(OrangeInstallDir, OrangeLibs),
              (os.path.join(OrangeInstallDir, "OrangeWidgets", "icons"),
               OrangeWidgetIcons),
              (os.path.join(OrangeInstallDir, "OrangeCanvas", "icons"),
               OrangeCanvasIcons)]

# Adding each doc/* directory by itself
for root, dirs, files in os.walk(os.path.join("doc")):
    OrangeDocs = glob(os.path.join("", "")) # Create a Docs file list
    if 'CVS' in dirs:
        dirs.remove('CVS')  # don't visit CVS directories
    for name in files:
        OrangeDocs += glob(os.path.join(root,name))
    if(root.split('doc/')[-1] == 'doc/'):
        root = ''
    data_files += [(os.path.join(OrangeInstallDoc,root.split('doc/')[-1]),
                                 OrangeDocs)]

# we save setup.py, so we can uninstall complete orange installation
#data_files += [(OrangeInstallDoc,['setup.py'])]

long_description = """Orange, data-mining software"""

setup (name = "orange",
       version = OrangeVer,
       maintainer = "Jure Menart",
       maintainer_email = "jurem@najdi.si",
       description = "Orange Extension for Python",
       long_description = long_description,
       url = "http://www.ailab.si/orange",
       packages = packages,
       package_dir = package_dir,
       data_files = data_files,
       cmdclass = { 'uninstall': uninstall,
                    'compile'  : compile,
                    'install'  : install_wrap}
       )
