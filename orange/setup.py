from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext

import os, sys, re
import glob

from subprocess import check_call

from types import *

from distutils.dep_util import newer_group
from distutils import log

from distutils.sysconfig import get_python_inc
import numpy
numpy_include_dir = numpy.get_include();
python_include_dir = get_python_inc(plat_specific=1)

include_dirs = [python_include_dir, numpy_include_dir, "source/include"]

if sys.platform == "darwin":
    extra_compile_args = "-fPIC -fpermissive -fno-common -w -DDARWIN ".split()
    extra_link_args = "-headerpad_max_install_names -undefined dynamic_lookup -lstdc++ -lorange_include".split()
elif sys.platform == "win32":
    extra_compile_args = []
    extra_link_args = []
elif sys.platform == "linux2":
    extra_compile_args = "-fPIC -fpermissive -w -DLINUX".split()
    extra_link_args = []    
else:
    extra_compile_args = []
    extra_link_args = []
    
define_macros=[('NDEBUG', '1'),
               ('HAVE_STRFTIME', None)],


#re_defvectors = re.compile(r"^\tpython \.\./pyxtract/defvectors\.py.*$")
#re_pyprops = re.compile(r"^\tpython \.\./pyxtract/pyprops\.py.*$")
#re_pyextract = re.compile(r"\tpython \.\./pyxtract/pyxtract\.py.*$")

class LibShared(Extension):
    def __init__(self, name, *args, **kwargs):
        Extension.__init__(self, name, *args, **kwargs)
        
class LibStatic(Extension):
    pass
        
class pyextract_build_ext(build_ext):
    
    def run_pyextract(self, ext, dir):
        original_dir = os.path.realpath(os.path.curdir)
        log.info("running pyextract for %s" % ext.name)
        try:
            os.chdir(dir)
            ## we use the commands which are used for building under windows
            if os.path.exists("_pyxtract.bat"): 
                pyextract_cmds = open("_pyxtract.bat").read().strip().splitlines()
                for cmd in pyextract_cmds:
#                    print " ".join([sys.executable] + cmd.split()[1:])
                    check_call([sys.executable] + cmd.split()[1:])
                    ext.include_dirs.append(os.path.join(dir, "ppp"))
                    ext.include_dirs.append(os.path.join(dir, "px"))

        finally:
            os.chdir(original_dir)
        
    def finalize_options(self):
        build_ext.finalize_options(self)
        self.library_dirs.append(self.build_lib) # add the build lib dir (for liborange_include
        
    def build_extension(self, ext):
        dir = os.path.commonprefix([os.path.split(s)[0] for s in ext.sources])
        self.run_pyextract(ext, dir)
        
        if isinstance(ext, LibStatic):
            self.build_static(ext)
        elif isinstance(ext, LibShared):
            build_ext.build_extension(self, ext)
            # Make lib{name}.so link to {name}.so
            from distutils.file_util import copy_file
            build_dir = self.build_lib
            ext_path = self.get_ext_fullpath(ext.name)
            ext_path, ext_filename = os.path.split(ext_path)
            try:
#                copy_file(os.path.join(ext_path, ext_filename), os.path.join(ext_path, "lib"+ext_filename), link="sym")
                copy_file(ext_filename, os.path.join(ext_path, "lib"+ext_filename), link="sym")
            except OSError:
                pass
        else:
            build_ext.build_extension(self, ext)
            
    def build_static(self, ext):
        ## mostly copied from build_extension, changed
        sources = ext.sources
        if sources is None or type(sources) not in (ListType, TupleType):
            raise DistutilsSetupError, \
                  ("in 'ext_modules' option (extension '%s'), " +
                   "'sources' must be present and must be " +
                   "a list of source filenames") % ext.name
        sources = list(sources)
        
        ext_path = self.get_ext_fullpath(ext.name)
        output_dir, _ = os.path.split(ext_path)
        lib_filename = self.compiler.library_filename(ext.name, lib_type='static', output_dir=output_dir)
        
        depends = sources + ext.depends
        if not (self.force or newer_group(depends, lib_filename, 'newer')):
            log.debug("skipping '%s' extension (up-to-date)", ext.name)
            return
        else:
            log.info("building '%s' extension", ext.name)

        # First, scan the sources for SWIG definition files (.i), run
        # SWIG on 'em to create .c files, and modify the sources list
        # accordingly.
        sources = self.swig_sources(sources, ext)

        # Next, compile the source code to object files.

        # XXX not honouring 'define_macros' or 'undef_macros' -- the
        # CCompiler API needs to change to accommodate this, and I
        # want to do one thing at a time!

        # Two possible sources for extra compiler arguments:
        #   - 'extra_compile_args' in Extension object
        #   - CFLAGS environment variable (not particularly
        #     elegant, but people seem to expect it and I
        #     guess it's useful)
        # The environment variable should take precedence, and
        # any sensible compiler will give precedence to later
        # command line args.  Hence we combine them in order:
        extra_args = ext.extra_compile_args or []

        macros = ext.define_macros[:]
        for undef in ext.undef_macros:
            macros.append((undef,))

        objects = self.compiler.compile(sources,
                                         output_dir=self.build_temp,
                                         macros=macros,
                                         include_dirs=ext.include_dirs,
                                         debug=self.debug,
                                         extra_postargs=extra_args,
                                         depends=ext.depends)

        # XXX -- this is a Vile HACK!
        #
        # The setup.py script for Python on Unix needs to be able to
        # get this list so it can perform all the clean up needed to
        # avoid keeping object files around when cleaning out a failed
        # build of an extension module.  Since Distutils does not
        # track dependencies, we have to get rid of intermediates to
        # ensure all the intermediates will be properly re-built.
        #
        self._built_objects = objects[:]

        # Now link the object files together into a "shared object" --
        # of course, first we have to figure out all the other things
        # that go into the mix.
        if ext.extra_objects:
            objects.extend(ext.extra_objects)
        extra_args = ext.extra_link_args or []

        # Detect target language, if not provided
        language = ext.language or self.compiler.detect_language(sources)

        self.compiler.create_static_lib(
            objects, ext.name, output_dir,
            debug=self.debug,
            target_lang=language)
        

def get_source_files(path, ext="cpp"):
    return glob.glob(os.path.join(path, "*." + ext))

include_ext = LibStatic("orange_include", get_source_files("source/include/"), include_dirs=include_dirs)

libraries = ["stdc++", "orange_include"]

orange_ext = LibShared("orange", get_source_files("source/orange/") + get_source_files("source/orange/blas/", "c"),
                       include_dirs=include_dirs,
                       extra_compile_args = extra_compile_args + ["-DORANGE_EXPORTS"],
                       extra_link_args = extra_link_args,
                       libraries=libraries)

orangeom_ext = Extension("orangeom", get_source_files("source/orangeom/") + get_source_files("source/orangeom/qhull/", "c"),
                         include_dirs=include_dirs + ["source/orange/"],
                         extra_compile_args = extra_compile_args + ["-DORANGEOM_EXPORTS"],
                         extra_link_args = extra_link_args,
                         libraries=libraries# + ["orange"]
                         )

orangene_ext = Extension("orangene", get_source_files("source/orangene/"), 
                         include_dirs=include_dirs + ["source/orange/"], 
                         extra_compile_args = extra_compile_args + ["-DORANGENE_EXPORTS"],
                         extra_link_args = extra_link_args,
                         libraries=libraries #+ ["orange"]
                         )

corn_ext = Extension("corn", get_source_files("source/corn/"), 
                     include_dirs=include_dirs + ["source/orange/"], 
                     extra_compile_args = extra_compile_args + ["-DCORN_EXPORTS"],
                     extra_link_args = extra_link_args,
                     libraries=libraries #+ ["orange"])
                     )

statc_ext = Extension("statc", get_source_files("source/statc/"), 
                      include_dirs=include_dirs + ["source/orange/"], 
                      extra_compile_args = extra_compile_args + ["-DSTATC_EXPORTS"],
                      extra_link_args = extra_link_args,
                      libraries=libraries #+ ["orange"]
                      )
 

pkg_re = re.compile("Orange/(.+?)/__init__.py")
packages = ["Orange"] + ["Orange." + pkg_re.findall(p)[0] for p in glob.glob("Orange/*/__init__.py")]
setup(cmdclass={"build_ext": pyextract_build_ext},
      name ="orange",
      version = "2.0b",
      description = "Orange data mining library for python.",
      author = "Bioinformatics Laboratory, FRI UL",
      author_email = "orange@fri.uni-lj.si",
      url = "www.ailab.si/orange",
      packages = packages + [".",
                             "OrangeCanvas", 
                             "OrangeWidgets", 
                             "OrangeWidgets.Associate",
                             "OrangeWidgets.Classify",
                             "OrangeWidgets.Data",
                             "OrangeWidgets.Evaluate",
                             "OrangeWidgets.Prototypes",
                             "OrangeWidgets.Regression",
                             "OrangeWidgets.Unsupervised",
                             "OrangeWidgets.Visualize",
                             ],
      package_data = {"OrangeCanvas": ["icons/*.png", "orngCanvas.pyw"],
                      "OrangeWidgets": ["icons/*.png", "icons/backgrounds/*.png", "report/index.html"],
                      "OrangeWidgets.Associate": ["icons/*.png"],
                      "OrangeWidgets.Classify": ["icons/*.png"],
                      "OrangeWidgets.Data": ["icons/*.png"],
                      "OrangeWidgets.Evaluate": ["icons/*.png"],
                      "OrangeWidgets.Prototypes": ["icons/*.png"],
                      "OrangeWidgets.Regression": ["icons/*.png"],
                      "OrangeWidgets.Unsupervised": ["icons/*.png"],
                      "OrangeWidgets.Visualize": ["icons/*.png"]
                      },
      ext_modules = [include_ext, orange_ext, orangeom_ext, orangene_ext, corn_ext, statc_ext],
      extra_path=("orange", "orange"),
      scripts = ["orange-canvas"],
      long_description="""Orange data mining library for python.""",
      classifiers = []
      )
      

