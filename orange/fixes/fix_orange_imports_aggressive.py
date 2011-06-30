"""Same as fix_orange_imports but will fix orange module name uses even if
they are not imported. To use this fixer you must pass the -a (--aggressive)
command line option."""

from .fix_orange_imports import FixOrangeImports, Name
from lib2to3.fixer_util import touch_import

class FixOrangeImportsAggressive(FixOrangeImports):
    run_order = 7
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
            new_name = self.mapping.get(bare_name.value)
            if new_name:
                bare_name.replace(Name(new_name, prefix=bare_name.prefix))
                touch_import(None, new_name, node)
                