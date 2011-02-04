from lib2to3 import pytree
from lib2to3.fixer_util import Name, Dot, Node, attr_chain, touch_import
from lib2to3 import fixer_base
from collections import defaultdict

from .fix_changed_names import FixChangedNames, MAPPING

def build_pattern(mapping=MAPPING):
    PATTERN = """
    power< local=(%s)
         tail=any*
    >
    """ 
    return PATTERN % "|".join("'%s'" % key.split(".")[-1] for key in mapping.keys())

class FixChangedNamesAggressive(FixChangedNames):
    mapping = MAPPING 
    run_order = 2
    
    def compile_pattern(self):
        # We override this, so MAPPING can be pragmatically altered and the
        # changes will be reflected in PATTERN.
        self.PATTERN = build_pattern(self.mapping)
#        self._modules_to_change = [key.split(".", 1)[0] for key in self.mapping.keys()]
        name2mod = defaultdict(set)
        for key in self.mapping.keys():
            mod, name = key.split(".", 1)
            name2mod[name].add(mod)
        self._names_to_modules = name2mod
        
        fixer_base.BaseFix.compile_pattern(self)
                
    def transform(self, node, results):
        local = results.get("local")
        tail = results.get("tail")
        if local:
            local = local[0]
            tail = tail[0]
            local_name = local.value
            modules = self._names_to_modules[local_name]
            if len(modules) > 1:
                import warnings
                warnings.warn("Conflicting name '%s' is present in %s! Ignoring transformation!" % local_name, modules)
                return
            
            ## TODOL check if tree contains a from module import * statement. if not ignore the transformation
            module = list(modules)[0]
            new_name = unicode(self.mapping[module + "." + local_name])
            
            syms = self.syms
            
            tail = tail.clone()
            new = self.package_tree(new_name)
            new = pytree.Node(syms.power, new + [tail])
            
            # Make sure the proper package is imported
            package = new_name.rsplit(".", 1)[0]
            touch_import(None, package, node)
            return new    