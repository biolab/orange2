#!/usr/bin/env python

import os
import sys
import shlex
from collections import namedtuple
from ConfigParser import SafeConfigParser

from setuptools import setup

from distutils.core import Extension
from distutils import dir_util, spawn, log

import glob
import numpy

import sipdistutils

try:
    from PyQt4 import pyqtconfig
except ImportError:
    pyqtconfig = None

try:
    from PyQt4.QtCore import PYQT_CONFIGURATION
except ImportError:
    PYQT_CONFIGURATION = None

pjoin = os.path.join

NAME                = 'orangeqt-qt'
DESCRIPTION         = 'orangeqt ploting library'
URL                 = "http://orange.biolab.si"
LICENSE             = 'GNU General Public License (GPL)'
AUTHOR              = "Bioinformatics Laboratory, FRI UL"
AUTHOR_EMAIL        = "orange@fri.uni-lj.si"
VERSION             = '0.0.1a'


pyqt_conf = namedtuple(
    "pyqt_conf",
    ["sip_flags",
     "sip_dir"]
)

qt_conf = namedtuple(
    "qt_conf",
    ["prefix",
     "include_dir",
     "library_dir",
     "framework",
     "framework_dir"]
)

config = namedtuple(
    "config",
    ["sip",
     "pyqt_conf",
     "qt_conf"]
)

pyqt_sip_dir = None
pyqt_sip_flags = None

qt_dir = None
qt_include_dir = None
qt_lib_dir = None
qt_framework = False

# use PyQt4 build time config if provided
if pyqtconfig is not None:
    cfg = pyqtconfig.Configuration()
    pyqt_sip_dir = cfg.pyqt_sip_dir
    pyqt_sip_flags = cfg.pyqt_sip_flags

    qt_dir = cfg.qt_dir
    qt_include_dir = cfg.qt_inc_dir
    qt_lib_dir = cfg.qt_lib_dir
    qt_framework = bool(cfg.qt_framework)
    qt_framework_dir = qt_lib_dir

elif PYQT_CONFIGURATION is not None:
    pyqt_sip_flags = PYQT_CONFIGURATION["sip_flags"]


# if QTDIR env is defined use it
if "QTDIR" in os.environ:
    qt_dir = os.environ["QTDIR"]
    if sys.platform == "darwin":
        if glob.glob(pjoin(qt_dir, "lib", "Qt*.framework")):
            # This is the standard Qt4 framework layout
            qt_framework = True
            qt_framework_dir = pjoin(qt_dir, "lib")
        elif glob(pjoin(qt_dir, "Frameworks", "Qt*.framework")):
            # Also worth checking (standard for bundled apps)
            qt_framework = True
            qt_framework_dir = pjoin(qt_dir, "Frameworks")

    if not qt_framework:
        # Assume standard layout
        qt_framework = False
        qt_include_dir = pjoin(qt_dir, "include")
        qt_lib_dir = pjoin(qt_dir, "lib")


extra_compile_args = []
extra_link_args = []
include_dirs = []
library_dirs = []


def site_config():
    parser = SafeConfigParser(dict(os.environ))
    parser.read(["site.cfg",
                 os.path.expanduser("~/.orangeqt-site.cfg")])

    def get(section, option, default=None, type=None):
        if parser.has_option(section, option):
            if type is None:
                return parser.get(section, option)
            elif type is bool:
                return parser.getboolean(section, option)
            elif type is int:
                return parser.getint(section, option)
            else:
                raise TypeError
        else:
            return default

    sip_bin = get("sip", "sip_bin")

    sip_flags = get("pyqt", "sip_flags", default=pyqt_sip_flags)
    sip_dir = get("pyqt", "sip_dir", default=pyqt_sip_dir)

    if sip_flags is not None:
        sip_flags = shlex.split(sip_flags)
    else:
        sip_flags = []

    prefix = get("qt", "qt_dir", default=qt_dir)
    include_dir = get("qt", "include_dir", default=qt_include_dir)
    library_dir = get("qt", "library_dir", default=qt_lib_dir)
    framework = get("qt", "framework", default=qt_framework, type=bool)
    framework_dir = get("qt", "framework_dir", default=qt_framework_dir)

    def path_list(path):
        if path and path.strip():
            return path.split(os.pathsep)
        else:
            return []

    include_dir = path_list(include_dir)
    library_dir = path_list(library_dir)

    conf = config(
        sip_bin,
        pyqt_conf(sip_flags, sip_dir),
        qt_conf(prefix, include_dir, library_dir,
                framework, framework_dir)
    )
    return conf

site_cfg = site_config()

if sys.platform == "darwin":
    sip_plaftorm_tag = "WS_MACX"
elif sys.platform == "win32":
    sip_plaftorm_tag = "WS_WIN"
elif sys.platform.startswith("linux"):
    sip_plaftorm_tag = "WS_X11"
else:
    sip_plaftorm_tag = ""

def which(name):
    """
    Return the path of program named 'name' on the $PATH.
    """
    if os.name == "nt" and not name.endswith(".exe"):
        name = name + ".exe"

    for path in os.environ["PATH"].split(os.pathsep):
        path = os.path.join(path, name)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    return None


class PyQt4Extension(Extension):
    pass


class build_pyqt_ext(sipdistutils.build_ext):
    """
    A build_ext command for building PyQt4 sip based extensions
    """
    description = "Build a orangeqt PyQt4 extension."

    def finalize_options(self):
        sipdistutils.build_ext.finalize_options(self)
        self.sip_opts = self.sip_opts + \
                        site_cfg.pyqt_conf.sip_flags + \
                        ["-j", "1"]  # without -j1 it does not build (??)

    def build_extension(self, ext):
        if not isinstance(ext, PyQt4Extension):
            return

        cppsources = [source for source in ext.sources
                      if source.endswith(".cpp")]

        dir_util.mkpath(self.build_temp, dry_run=self.dry_run)

        # Run moc on all header files.
        for source in cppsources:
            header = source.replace(".cpp", ".h")
            if os.path.exists(header):
                moc_file = os.path.basename(header).replace(".h", ".moc")
                out_file = os.path.join(self.build_temp, moc_file)
                call_arg = ["moc", "-o", out_file, header]
                spawn.spawn(call_arg, dry_run=self.dry_run)

        # Add the temp build directory to include path, for compiler to find
        # the created .moc files
        ext.include_dirs = ext.include_dirs + [self.build_temp]

        sipdistutils.build_ext.build_extension(self, ext)

    def _find_sip(self):
        if site_cfg.sip:
            log.info("Using sip at %r (from .cfg file)" % site_cfg.sip)
            return site_cfg.sip

        # Try the base implementation
        sip = sipdistutils.build_ext._find_sip(self)
        if os.path.isfile(sip):
            return sip

        log.warn("Could not find sip executable at %r." % sip)

        # Find sip on $PATH
        sip = which("sip")

        if sip:
            log.info("Found sip on $PATH at: %s" % sip)
            return sip

        return sip

    # For sipdistutils to find PyQt4's .sip files
    def _sip_sipfiles_dir(self):
        if site_cfg.pyqt_conf.sip_dir:
            return site_cfg.pyqt_conf.sip_dir

        if os.path.isdir(pyqt_sip_dir):
            return pyqt_sip_dir

        log.warn("The default sip include directory %r does not exist" %
                 pyqt_sip_dir)

        path = os.path.join(sys.prefix, "share/sip/PyQt4")
        if os.path.isdir(path):
            log.info("Found sip include directory at %r" % path)
            return path

        return "."


def get_source_files(path, ext="cpp", exclude=[]):
    files = glob.glob(os.path.join(path, "*." + ext))
    files = [file for file in files if os.path.basename(file) not in exclude]
    return files


# Used Qt4 libraries
qt_libs = ["QtCore", "QtGui"]

if site_cfg.qt_conf.framework:
    framework_dir = site_cfg.qt_conf.framework_dir
    extra_compile_args = ["-F%s" % framework_dir]
    extra_link_args = ["-F%s" % framework_dir]
    for lib in qt_libs:
        include_dirs += [os.path.join(framework_dir,
                                      lib + ".framework", "Headers")]
        extra_link_args += ["-framework", lib]
    qt_libs = []
else:
    if type(site_cfg.qt_conf.include_dir) == list:
        include_dirs = site_cfg.qt_conf.include_dir + \
                [pjoin(d, lib)
                for lib in qt_libs for d in site_cfg.qt_conf.include_dir]
    else:
        include_dirs = [site_cfg.qt_conf.include_dir] + \
                   [pjoin(site_cfg.qt_conf.include_dir, lib)
                    for lib in qt_libs]
    library_dirs += site_cfg.qt_conf.library_dir

if sys.platform == "win32":
    # Qt libs on windows have a 4 added
    qt_libs = [lib + "4" for lib in qt_libs]

include_dirs += [numpy.get_include(), "./"]

orangeqt_ext = PyQt4Extension(
    "orangeqt",
    ["orangeqt.sip"] + get_source_files(
        "", "cpp",
        exclude=["canvas3d.cpp", "plot3d.cpp", "glextensions.cpp"]
     ),
    include_dirs=include_dirs,
    extra_compile_args=extra_compile_args + \
                         ["-DORANGEQT_EXPORTS"],
    extra_link_args=extra_link_args,
    libraries=qt_libs,
    library_dirs=library_dirs
)

ENTRY_POINTS = {
    'orange.addons': (
        'orangeqt = orangeqt',
    ),
}


def setup_package():
    setup(name=NAME,
          description=DESCRIPTION,
          version=VERSION,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          url=URL,
          license=LICENSE,
          ext_modules=[orangeqt_ext],
          cmdclass={"build_ext": build_pyqt_ext},
          entry_points=ENTRY_POINTS,
          )

if __name__ == '__main__':
    setup_package()
