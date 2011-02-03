
from lib2to3 import fixer_base
from lib2to3 import fixer_util
from lib2to3.fixer_util import Name, attr_chain

from lib2to3.fixes import fix_imports

#Why does this not work?? but the copy paste of FixImports class does

#class FixOrangeImports(fix_imports.FixImports):
#    mapping = {"orange": "Orange.core"}
#    
#    def transfrom(self, node, results):
#        if "use" in results:
#            nodes = [fixer_util.Node("core"),
#                     fixer_util.Dot()
#                     ]
#        return fixer_util.FromImport("Orange", "core")
#    

"""Fix incompatible imports and module references."""
# Authors: Collin Winter, Nick Edds

# Local imports
from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, attr_chain

MAPPING = {"orange": "Orange.core",
           "orngSVM": "Orange.classification.svm",
           "orngSOM": "Orange.projection.som",
           "orngClustering": "Orange.cluster"
           }


def alternates(members):
    return "(" + "|".join(map(repr, members)) + ")"


def build_pattern(mapping=MAPPING):
    mod_list = ' | '.join(["module_name='%s'" % key for key in mapping])
    bare_names = alternates(mapping.keys())

    yield """name_import=import_name< 'import' ((%s) |
               multiple_imports=dotted_as_names< any* (%s) any* >) >
          """ % (mod_list, mod_list)
    yield """import_from< 'from' (%s) 'import' ['(']
              ( any | import_as_name< any 'as' any > |
                import_as_names< any* >)  [')'] >
          """ % mod_list
    yield """import_name< 'import' (dotted_as_name< (%s) 'as' any > |
               multiple_imports=dotted_as_names<
                 any* dotted_as_name< (%s) 'as' any > any* >) >
          """ % (mod_list, mod_list)

    # Find usages of module members in code e.g. thread.foo(bar)
    yield "power< bare_with_attr=(%s) trailer<'.' any > any* >" % bare_names


class FixOrangeImports(fixer_base.BaseFix):
    mapping = MAPPING

    # We want to run this fixer late, so fix_import doesn't try to make stdlib
    # renames into relative imports.
    run_order = 6

    def build_pattern(self):
        return "|".join(build_pattern(self.mapping))

    def compile_pattern(self):
        # We override this, so MAPPING can be pragmatically altered and the
        # changes will be reflected in PATTERN.
        self.PATTERN = self.build_pattern()
#        print self.PATTERN
        super(FixOrangeImports, self).compile_pattern()
#        print self.pattern

    # Don't match the node if it's within another match.
    def match(self, node):
        match = super(FixOrangeImports, self).match
        results = match(node)
        if results:
            # Module usage could be in the trailer of an attribute lookup, so we
            # might have nested matches when "bare_with_attr" is present.
            if "bare_with_attr" not in results and \
                    any(match(obj) for obj in attr_chain(node, "parent")):
                return False
            return results
        return False

    def start_tree(self, tree, filename):
        super(FixOrangeImports, self).start_tree(tree, filename)
        self.replace = {}

    def transform(self, node, results):
        import_mod = results.get("module_name")
        if import_mod:
            mod_name = import_mod.value
            new_name = unicode(self.mapping[mod_name])
            import_mod.replace(Name(new_name, prefix=import_mod.prefix))
            if "name_import" in results:
                # If it's not a "from x import x, y" or "import x as y" import,
                # marked its usage to be replaced.
                self.replace[mod_name] = new_name
            if "multiple_imports" in results:
                # This is a nasty hack to fix multiple imports on a line (e.g.,
                # "import StringIO, urlparse"). The problem is that I can't
                # figure out an easy way to make a pattern recognize the keys of
                # MAPPING randomly sprinkled in an import statement.
                results = self.match(node)
                if results:
                    self.transform(node, results)
        else:
            # Replace usage of the module.
            bare_name = results["bare_with_attr"][0]
            new_name = self.replace.get(bare_name.value)
            if new_name:
                bare_name.replace(Name(new_name, prefix=bare_name.prefix))
