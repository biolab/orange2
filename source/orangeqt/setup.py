#!usr/bin/env python

import os, sys

try:
    from setuptools import setup
    have_setuptools = True
except ImportError:
    from distutils.core import setup
    have_setuptools = False

from distutils.core import Extension

import subprocess


NAME                = 'orangeqt-qt'
DESCRIPTION         = 'orangeqt ploting library'
URL                 = "http://orange.biolab.si"
LICENSE             = 'GNU General Public License (GPL)'
AUTHOR              = "Bioinformatics Laboratory, FRI UL"
AUTHOR_EMAIL        = "orange@fri.uni-lj.si"
VERSION             = '0.0.1a'

import numpy

import glob

import PyQt4.QtCore
from PyQt4 import pyqtconfig

cfg = pyqtconfig.Configuration()

pyqt_sip_dir = cfg.pyqt_sip_dir

import sipdistutils

extra_compile_args = []
extra_link_args = []

if sys.platform == "darwin":
    sip_plaftorm_tag = "WS_MACX"
elif sys.platform == "win32":
    sip_plaftorm_tag = "WS_WIN"
elif sys.platform.startswith("linux"):
    sip_plaftorm_tag = "WS_X11"
else:
    sip_plaftorm_tag = ""

class my_build_ext(sipdistutils.build_ext):
    def finalize_options(self):
        sipdistutils.build_ext.finalize_options(self)
        self.sip_opts = self.sip_opts + ["-k", "-j", "1", "-t", 
                        sip_plaftorm_tag, "-t",
                        "Qt_" + PyQt4.QtCore.QT_VERSION_STR.replace('.', '_')]

    def build_extension(self, ext):
        cppsources = [source for source in ext.sources if source.endswith(".cpp")]
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        for source in cppsources:
            header = source.replace(".cpp", ".h")
            if os.path.exists(header):
                moc_file = header.split('/')[-1].replace(".h", ".moc")
                try:
                    subprocess.call(["moc", "-o" + os.path.join(self.build_temp, moc_file), header])
                except OSError:
                    raise OSError("Could not locate 'moc' executable.")
        ext.extra_compile_args = ext.extra_compile_args + ["-I" + self.build_temp]
        sipdistutils.build_ext.build_extension(self, ext)

    def _sip_sipfiles_dir(self):
        return pyqt_sip_dir

def get_source_files(path, ext="cpp", exclude=[]):
    files = glob.glob(os.path.join(path, "*." + ext))
    files = [file for file in files if os.path.basename(file) not in exclude]
    return files

qt_libs = ["QtCore", "QtGui", "QtOpenGL"]

if cfg.qt_framework:
    extra_includes = []
    extra_link_args = ["-F%s" % cfg.qt_lib_dir]
    for lib in qt_libs:
        extra_includes += ["-I" + os.path.join(cfg.qt_lib_dir,
                                        lib + ".framework", "Headers")]
        extra_link_args += ["-framework", lib]
    qt_libs = []
else:
    extra_includes = ["-I" + cfg.qt_inc_dir] \
            + ["-I" + os.path.join(cfg.qt_inc_dir, lib) for lib in qt_libs]
    extra_link_args = ["-L%s" % cfg.qt_lib_dir]

extra_includes += ["-I" + numpy.get_include()] + ["-I./"]

orangeqt_ext = Extension("orangeqt",
                        ["orangeqt.sip"] + get_source_files("", "cpp"),
                        extra_compile_args = extra_compile_args
                                            + ["-DORANGEQT_EXPORTS"]
                                            + extra_includes,
                        extra_link_args = extra_link_args,
                        libraries = qt_libs
                        )

def setup_package():
    setup(name = NAME,
          description = DESCRIPTION,
          version = VERSION,
          author = AUTHOR,
          author_email = AUTHOR_EMAIL,
          url = URL,
          license = LICENSE,
          ext_modules = [orangeqt_ext],
          cmdclass={"build_ext": my_build_ext},
          )

if __name__ == '__main__':
    setup_package()
