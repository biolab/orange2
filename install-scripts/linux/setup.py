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
gotNumPy = '';
gotGcc = '';
gotQwt = '';

try:
    import sys,commands,traceback,os
    from distutils import sysconfig
    from distutils.core import setup,Extension
    from glob import glob
    from stat import *
except:
    traceback.print_exc()
    print "Unable to import python distutils."
    print "You may want to install the python-dev package on your distribution."
    sys.exit(1)

gotPython = sys.version.split()[0]

if not hasattr(sys, 'version_info') or sys.version_info < (2,3,0,'alpha',0):
    raise SystemExit, "Python 2.3 or later is required to build Orange."

#if os.geteuid() != 0:
#    print "This script should be run as superuser!"
#    sys.exit(1)

try:
    import qt,pyqtconfig
except:
    print "NOTE: Python Qt not installed, OrangeCanvas and OrangeWidgets will not work."
    print "You can get it at: http://www.riverbankcomputing.co.uk/pyqt/index.php"
    # we are using Qt 2.3 - Qt 3.3

try:
    gotPyQt = pyqtconfig.Configuration().pyqt_version_str;
except:
    gotPyQt = ''
    # any PyQt that supports Qt 2.3-3.8

try:
    import sipconfig
    tmp = "%x" % sipconfig.Configuration().qt_version
    gotPy = tmp.replace('0','.')
except:
    print "Sipconfig not found, Qt version could not be found!"
    gotPy = ''
    # depends on version of PyQt

try:
    import numpy
except:
    print "NumPy should be installed!"
    print "You can get it at: http://numpy.scipy.org/"
    sys.exit(1)
    # use latest, from 23.0 on

try:
    import numpy
    gotNumPy = numpy.__version__
except:
    print "Can not determine NumPy version!"
    gotNumPy = "n/a"

try:
    import qwt
    gotQwt = "n/a"
except:
    print "PyQwt not installed!"
    print "You can get it at: http://pyqwt.sourceforge.net/"
    # depends on PyQt
    
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
    # version 3.3 on

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
from distutils.command.install_data import install_data

class uninstall(Command):
    description = "uninstall current version"

    user_options = [('orangepath=', None, "Orange installation path"),
                    ('docpath=', None, "Orange documentaiton path"),
                    ('libpath=', None, "Orange library path")]

    
    def initialize_options (self):
        self.orangepath = None
        self.docpath = None
        self.libpath = None
        self.systemUninstall = False
        
    def finalize_options(self):
        if self.orangepath == None and 'root' in self.__dict__.keys() and self.root <> None:
            self.orangepath = self.root
            self.root = None

        if self.orangepath is None:
            try:
                fo = file(sys.argv[0].replace("/setup.py", "/user_install"), "r")
                self.orangepath=fo.read()
                fo.close
            except:
                print "user_install file could not be opened, performing system uninstall"

        else:
            self.systemUninstall = False
            
        if self.orangepath is None:
            self.systemUninstall = True
            if self.orangepath is None:
                self.orangepath = os.path.join("lib", PythonVer,
                                               "site-packages", "orange")
            if self.docpath is None:
                self.docpath = os.path.join(sys.prefix, "share", "doc", "orange")
            if self.libpath is None:
                self.libpath = os.path.join(sys.prefix, "lib")
        else:
            if self.docpath is None:
                self.docpath = os.path.join(self.orangepath, "doc")
                self.orangepath = os.path.join(self.orangepath, "orange")
                
    def run(self):
        if self.systemUninstall is True:
            if os.geteuid() != 0:
                print "Uninstallation should be run as superuser!"
                sys.exit(1)
            self.orangepath = os.path.join(sys.prefix, self.orangepath)
            sysBinFile = os.path.join("/", "usr", "bin", "orange")
            print "Performing system uninstallation from: "+self.orangepath
        else:
            sysBinFile = None
            print "Performing user uninstallation from: "+self.orangepath
        print "Removing orange.pth file...",
        os.remove(os.path.join(self.orangepath, "..", "orange.pth"))
        print "done!"
        print "Removing installation directory "+self.orangepath+" ...",
        self.rmdir(self.orangepath)
        os.rmdir(self.orangepath)
        print "done!"
        print "Removing documentation directory "+self.docpath+" ...",
        self.rmdir(self.docpath)
        os.rmdir(self.docpath)
        print "done!"
        if self.systemUninstall is True:
            print "Removing symbolic links for the orange libraries ...",
            for orngLib in OrangeLibList:
                os.remove(os.path.join(self.libpath, "lib"+orngLib))
            print "done!"
            print "Removing Orange Canvas shortcut "+sysBinFile+"...",
            os.remove(sysBinFile);
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

# change correct directories in data_files (we don't know target directory in
# the 'main' script

class install_data_wrap(install_data):
    description = "Orange specific data installation"

    print "Orange specific data installation"

    global OrangeInstallDir
    
    user_options = install_data.user_options
    
    def initialize_options(self):
        install_data.initialize_options(self)

    def finalize_options(self):
        install_data.finalize_options(self)
        if self.install_dir != os.path.join(self.root, sys.prefix):
            OrangeInstallDir = os.path.join(self.install_dir, "orange")
            OrangeInstallDoc = os.path.join(self.install_dir, "doc", "orange")
#            OrangeInstallLib = os.path.join(self.install_dir, "lib")
            self.data_files = [(OrangeInstallDir, OrangeLibs),
                               (os.path.join(OrangeInstallDir,
                                             "OrangeWidgets", "icons"),
                                OrangeWidgetIcons),
                               (os.path.join(OrangeInstallDir,
                                             "OrangeCanvas", "icons"),
                                OrangeCanvasIcons),
                               (os.path.join(OrangeInstallDir, "OrangeCanvas"),
                                OrangeCanvasPyw)]
            for root, dirs, files in os.walk(os.path.join("doc")):
                OrangeDocs = glob(os.path.join("", "")) # Create a Docs file list
                if 'CVS' in dirs:
                    dirs.remove('CVS')  # don't visit CVS directories
                for name in files:
                    OrangeDocs += glob(os.path.join(root,name))
                if(root.split('doc/')[-1] == 'doc/'):
                    root = ''
                self.data_files += [(os.path.join(OrangeInstallDoc,
                                             root.split('doc/')[-1]), OrangeDocs)]

            self.data_files += [(OrangeInstallDoc,['setup.py'])]
            
    def run(self):
        install_data.run(self)

# install is 'normal distutils' install, after installation symlinks libraries
# and calles ldconfig
class install_wrap(install):
    description = "Orange specific installation"

    user_options = install.user_options + [('orangepath=',
                                            None, "Orange install directory")]

    def initialize_options(self):
        install.initialize_options(self)
        self.orangepath = None;

    def finalize_options(self):
        if self.orangepath == None and self.root <> None:
            self.orangepath = self.root
            self.root = None

        if self.orangepath == None:
            print "Using default system initialization, checking for privileges..."
            if os.geteuid() != 0:
                print "should be run as superuser or --orangepath should be used"
                sys.exit(1)
            print "done"
            self.OrangeInstallDir = os.path.join("lib", PythonVer,
                                            "site-packages", "orange")
            self.OrangeInstallDoc = os.path.join(sys.prefix, "share", "doc",
                                            "orange")
            self.OrangeInstallLib = os.path.join(sys.prefix, "lib")
        else:
            self.install_purelib = self.orangepath;
            self.install_data = self.orangepath;

        install.finalize_options(self);
 
    def run(self):
        install.run(self)

        if self.orangepath == None:
	    binFile = os.path.join(sys.prefix, self.OrangeInstallDir, "OrangeCanvas", "orngCanvas.pyw")
	    sysBinFile = os.path.join("/", "usr", "bin", "orange")
	    os.system("echo python "+binFile+" >> "+sysBinFile)
            os.chmod(sysBinFile,
                      S_IRUSR|S_IWUSR|S_IXUSR|S_IRGRP|S_IXGRP|S_IROTH|S_IXOTH)

            print "Linking libraries...",
            for currentLib in OrangeLibList:
                try:
                    os.symlink(os.path.join(sys.prefix,
                                            self.OrangeInstallDir,currentLib),
                               os.path.join(self.OrangeInstallLib,"lib"+currentLib))
                    
                except:
                    print "problems with "+currentLib+"... ignoring... ",
                    
                os.system("/sbin/ldconfig")
                print "success"
        else:
	    sysBinFile = None
            # link orange.so into liborange.so
            try:
                os.symlink(os.path.join(self.orangepath, "orange", "orange.so"), os.path.join(self.orangepath, "orange", "liborange.so"))
            except:
                print "problems creating link to liborange.so:", os.path.join(self.orangepath, "orange", "liborange.so")
            print "Libraries were not exported to the system library path (using 'non-system' installation)"
            
        print "Creating path file...",
        if self.orangepath == None:
            pth = os.path.join(sys.prefix,self.OrangeInstallDir,"..","orange.pth")
        else:
            pth = os.path.join(self.orangepath, "orange.pth")

        if os.access(pth, os.F_OK) is True:
            os.remove(pth)

        fo = file(pth,"w+")

        if self.orangepath == None:
            pthInstallDir = os.path.join(sys.prefix, self.OrangeInstallDir)
        else:
            pthInstallDir = os.path.join(self.orangepath, "orange")
            
        fo.write(pthInstallDir+"\n")
        for root,dirs,files in os.walk(pthInstallDir):
            for name in dirs:
                fo.write(os.path.join(root,name)+"\n")
        fo.close()
        print "success"

       	print "Preparing filename masks...",
	for root, dirs, files in os.walk(pthInstallDir):
	    for name in files:
		if name in OrangeLibList: # libraries must have +x flag too
			os.chmod(os.path.join(root,name),   
				 S_IRUSR|S_IWUSR|S_IXUSR|S_IRGRP|S_IXGRP|S_IROTH|S_IXOTH)

		else:
	        	os.chmod(os.path.join(root,name),
        	        	 S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH)
	    for name in dirs:
        	os.chmod(os.path.join(root,name),
                	 S_IRUSR|S_IWUSR|S_IXUSR|S_IRGRP|S_IXGRP|S_IROTH|S_IXOTH)
        print "success"
	print ""
        print "Python version: "+gotPython
        print "PyQt version: "+gotPyQt
        print "Qt version: "+gotPy
        print "NumPy version: "+gotNumPy
        print "Qwt version: "+gotQwt
        print "GCC version: "+gotGcc

        if self.orangepath != None:
            # we save orangepath for uninstallation to the file user_install
            fo = file(os.path.join(self.orangepath, "doc",
                                  "orange", "user_install"), "w+")
            fo.write(self.orangepath)
            fo.close()
            OrangeInstallDir = os.path.join(self.orangepath, "orange")
            OrangeInstallDoc = os.path.join(self.orangepath, "doc", "orange")
            print "Orange installation dir: "+OrangeInstallDir
            print "Orange documentation dir: "+OrangeInstallDoc
            print "To uninstall Orange type:"
            print ""
            print "    python "+OrangeInstallDoc+"/setup.py uninstall"
        else:
            print "Orange installation dir: "+sys.prefix+self.OrangeInstallDir
            print "Orange library links in: "+self.OrangeInstallLib
            print "Orange documentation dir: "+self.OrangeInstallDoc
            print "To uninstall Orange type:"
            print ""
            print "    python "+self.OrangeInstallDoc+"/setup.py uninstall"
            
        print ""
        print "It will remove Orange, Orange documentation and links to Orange libraries"
        if sysBinFile != None:
		print ""
		print "To run Orange Canvas run: "+sysBinFile
	else:
		print ""
		print "Orange Canvas shortcut not created ('non-system' installation)"
		print ""
		print "PLEASE ADD to your environment: LD_LIBRARY_PATH="+OrangeInstallDir+":$LD_LIBRARY_PATH ; export LD_LIBRARY_PATH"
		print ""
    
# preparing data for Distutils

PythonVer = "python"+sys.version[:3]

OrangeInstallDir = os.path.join("lib", PythonVer, "site-packages", "orange")
OrangeInstallDoc = os.path.join(sys.prefix, "share", "doc",
                                "orange")
OrangeInstallLib = os.path.join(sys.prefix, "lib")

OrangeLibList = ['orange.so', 'orangene.so','orangeom.so','statc.so','corn.so']
        
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
OrangeCanvasPyw   = glob(os.path.join("OrangeCanvas", "orngCanvas.pyw"));

data_files = [(OrangeInstallDir, OrangeLibs),
              (os.path.join(OrangeInstallDir, "OrangeWidgets", "icons"),
               OrangeWidgetIcons),
              (os.path.join(OrangeInstallDir, "OrangeCanvas", "icons"),
               OrangeCanvasIcons),
	      (os.path.join(OrangeInstallDir, "OrangeCanvas"), OrangeCanvasPyw)]

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
data_files += [(OrangeInstallDoc,['setup.py'])]

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
                    'install'  : install_wrap,
                    'install_data' : install_data_wrap}
       )
