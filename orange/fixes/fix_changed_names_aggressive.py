from lib2to3 import pytree
from lib2to3.fixer_util import Name, Dot, Node, attr_chain, touch_import, find_root
from lib2to3 import fixer_base
from lib2to3 import patcomp
from collections import defaultdict

from .fix_changed_names import FixChangedNames, MAPPING

def find_matches(pattern, tree):
    for node in tree.pre_order():
        res = {}
        match = pattern.match(node, res)
        if match:
            yield res
    
def build_pattern(mapping=MAPPING):
    PATTERN = """
    power< local=(%s)
         tail=any*
    >
    """ 
    return PATTERN % "|".join("'%s'" % key.split(".")[-1] for key in mapping.keys())


## Pattern for finding from module import * 
from_import_pattern = """import_from<'from' module=(%s) 'import' star='*'>"""
module_names = set(["'%s'" % key.split(".", 1)[0] for key in MAPPING.keys()])
from_import_pattern = from_import_pattern % "|".join(module_names)

from_import_expr = patcomp.compile_pattern(from_import_pattern)
 
class FixChangedNamesAggressive(FixChangedNames):
    mapping = MAPPING 
    run_order = 2
    
    def compile_pattern(self):
        # We override this, so MAPPING can be pragmatically altered and the
        # changes will be reflected in PATTERN.
        self.PATTERN = build_pattern(self.mapping)
        name2mod = defaultdict(set)
        for key in self.mapping.keys():
            mod, name = key.split(".", 1)
            name2mod[name].add(mod)
        self._names_to_modules = name2mod
        
        fixer_base.BaseFix.compile_pattern(self)
        
    def start_tree(self, tree, filename):
        super(FixChangedNamesAggressive, self).start_tree(tree, filename)
        ## Find unqualified imports
        self._import_matches = list(find_matches(from_import_expr, tree))
        
    def finish_tree(self, tree, filename):
        del self._import_matches
                
    def transform(self, node, results):
        local = results.get("local")
        tail = results.get("tail")
        if local:
            local = local[0]
            local_name = local.value
            modules = self._names_to_modules[local_name]
            if len(modules) > 1:
                self.warnings(node, "Conflicting name '%s' is present in %s! Ignoring transformation!" % local_name, modules)
                return
            
            module = list(modules)[0]
            
            if all("module" not in res for res in self._import_matches): 
                self.warning(node, "Aggressive name matched '%s' but no corresponding import! Fix manualy." % local)
                return
                
            new_name = unicode(self.mapping[module + "." + local_name])
            
            syms = self.syms
            
            if tail:
                tail = [t.clone() for t in tail]
#            tail = tail.clone()
            new = self.package_tree(new_name)
            new = pytree.Node(syms.power, new + tail, prefix=local.prefix)
            
            # Make sure the proper package is imported
            package = new_name.rsplit(".", 1)[0]
            touch_import(None, package, node)
            return new
        