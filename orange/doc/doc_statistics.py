import orngEnviron
import glob
import os.path
import sys

test_type = "orange" if len(sys.argv) < 2 else sys.argv[1]
if test_type in ["orange", "orng"]:
    orange_dir = orngEnviron.orangeDir
    module_head = "orng"
elif test_type == "obi":
    orange_dir = orngEnviron.addOnsDir + "/Bioinformatics"
    module_head = "obi"
elif test_type == "text":
    orange_dir = orngEnviron.addOnsDir + "/Text"
    module_head = "orng"
else:
    print "Error: wrong arguments"
    print "%s [orng|obi|text]"
    sys.exit(1)

# utility functions

def pp_names(names, lead="   ", cols=80, sep=", "):
    """Pretty-print list of names."""
    buffer = []
    for name in names:
        if (len(lead) + len(sep.join(buffer))) > cols:
            print "%s%s" % (lead, sep.join(buffer))
            buffer = []
        buffer.append(name)
    print "%s%s" % (lead, sep.join(buffer))

# compile a list of Orange modules, check for their documentation

modules = [os.path.basename(m)[:-3] for m in glob.glob(orange_dir + "/%s*.py" % module_head)]
module_docs = [os.path.basename(m)[:-4] for m in glob.glob(orange_dir + "/doc/modules/*.htm")]
not_documented = [m for m in modules if m not in module_docs]
documented = [m for m in modules if m in module_docs]

print "DOCUMENTATION OF MODULES"
print "There are %d modules, %d with documentation" % (len(modules), len(documented)) 
print "Of %d modules, %d are not documented:" % (len(modules), len(not_documented))
pp_names(sorted(not_documented))

# compile a list of scripts for regression testing

def collect_documentation_scripts(basedir, dir):
    """Return a list of documentation scripts in a specific directory"""
    excl_name = "%s/%s/exclude-from-regression.txt" % (basedir, dir)
    exclude_files = [line.rstrip() for line in file(excl_name).readlines()] if os.path.exists(excl_name) else []
    names = [os.path.basename(m) for m in glob.glob("%s/%s/*.py" % (basedir, dir))]
    return [n for n in names if n not in exclude_files]

def collect_imports(dir, scriptname):
    """For script name return the list of names of imported documentation modules"""
    names = []
    for scriptname in file("%s/%s" % (dir, scriptname)).readlines():
        if scriptname.startswith("import "):
            names.extend(scriptname[7:].rstrip().replace(" ", "").split(","))
        if scriptname.startswith("from ") and "import" in scriptname:
            names.extend(scriptname[5:scriptname.index("import")].replace(" ", "").split(","))
    return set([n for n in names if n in modules])

class DocScript:
    """Stores info on documentation script."""
    def __init__(self, **karg):
        self.__dict__ = karg
        self.modules = set()
    def __str__(self):
        return "%s/%s" % (dir, name)
    
print
print "REGRESSION TESTING SCRIPTS"

print "Regression scripts for documentation section:"
scripts = {}
exclude_dirs = set([".svn", "datasets", "widgets"])
script_dirs = [f for f in os.listdir(orange_dir + "/doc") if os.path.isdir(orange_dir + "/doc/" + f) and f not in exclude_dirs]
for dir in script_dirs:
    ms = dict([((dir, name), DocScript(name=name, dir=dir)) for name in collect_documentation_scripts(orange_dir + "/doc", dir)])
    print "   %-10s %d" % (dir, len(ms))
    scripts.update(ms)

for (_, script), info in scripts.items():
    info.modules.update(collect_imports(orange_dir + "/doc/" + info.dir, script))

module_scripts = {}
for info in scripts.values():
    for m in info.modules:
        module_scripts.setdefault(m, []).append(info.name)
        
module2nscripts = dict([(m, len(ss)) for m, ss in module_scripts.items()])

print "Regression scripts that use a specific documented module:"
for (n, m) in sorted([(module2nscripts.get(m, 0), m) for m in documented]):
    print "   %-15s %d" % (m, n)
