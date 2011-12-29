from lib2to3 import fixer_base
from lib2to3 import fixer_util
from lib2to3.fixer_util import Name, attr_chain, syms, FromImport, token, Node, Leaf, BlankLine, Comma, touch_import, String

from lib2to3.fixes import fix_imports

"""
Replaces explicit import to Orange submodules with "import Orange".
"""

def Import(name_leafs):

    for leaf in name_leafs:
        # Pull the leaves out of their old tree
        leaf.remove()

    def add_commas(leafs):
        yield leafs[0]
        for a in leafs[1:]:
            yield Comma()
            yield a

    children = [Leaf(token.NAME, u'import'),
                Node(syms.dotted_as_names, list(add_commas(name_leafs)))]
    imp = Node(syms.import_name, children)
    return imp

class FixOrangeTopImport(fixer_base.BaseFix):

    PATTERN = """import_name< 'import' imp=any >"""

    run_order = 7

    def transform(self, node, results):
        imp = results['imp']

        def t(imp):
            new_contents = []
        
            def handle_one(n):
                if n.type == token.NAME:
                    if "Orange." not in n.value:
                        new_contents.append(n)
                    else:
                        touch_import(None, "Orange", node)
                else:
                    new_contents.append(n)

            if imp.type == syms.dotted_as_names:
                for n in imp.children[::2]:
                    handle_one(n)
            else:
                handle_one(imp)
    
            #copy prefix, so you do not lose comments
            opref = node.prefix
            if new_contents:
                nn = Import(new_contents)
                nn.prefix = opref
            else:
                nn = BlankLine()
                if opref and opref[-1] in ["\n"]: #remove previous newline
                    opref = opref[:-1]
                nn.prefix = opref
        
            return nn

        return t(imp)
