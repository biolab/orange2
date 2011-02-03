from lib2to3 import fixer_base
from lib2to3 import fixer_util

class FixFromOrangeImports(fixer_base.BaseFix):
    PATTERN = """
    import_from < 'from' trailer module_name='orange' trailer 'import' trailer names=any* >
    """
    
    run_order = 10
    
    def transform(self, node, results):
        names = results["names"]
        new = fixer_util.FromImport("Orange.core", names.clone())