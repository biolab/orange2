from lib2to3 import fixer_base
from lib2to3 import fixer_util
from lib2to3 import pytree
from lib2to3.fixer_util import Name, Dot, Node, attr_chain, touch_import

# keys must be in the form of 'orange.name' not name or orange.bla.name 
MAPPING = {"orange.ExampleTable": "Orange.data.Table",
           "orange.Example": "Orange.data.Instance",
           "orange.Domain": "Orange.data.Domain",
           "orange.Value": "Orange.data.Value",
           "orange.VarTypes": "Orange.data.Type",
           "orange.Variable": "Orange.data.Feature",
           "orange.EnumVariable": "Orange.data.Discrete",
           "orange.FloatVariable": "Orange.data.Continuous",
           "orange.StringVariable": "Orange.data.String",
           "orange.PythonVariable": "Orange.data.Python",
           "orange.VarList": "Orange.data.Features",
           
           "orange.MeasureAttribute_gainRatio": "Orange.feature.scoring.GainRatio",
           "orange.MeasureAttribute": "Orange.feature.scoring.Measure",
           "orange.MeasureAttribute_relief": "Orange.feature.scoring.Relief",
           "orange.MeasureAttribute_info": "Orange.feature.scoring.InfoGain",
           "orange.MeasureAttribute_gini": "Orange.feature.scoring.Gini",
           
           "orange.ExamplesDistance_Hamming": "Orange.distances.ExamplesDistance_Hamming",
           "orange.ExamplesDistance_Normalized": "Orange.distances.ExamplesDistance_Normalized",
           "orange.ExamplesDistance_DTW": "Orange.distances.ExamplesDistance_DTW", 
           "orange.ExamplesDistance_Euclidean": "Orange.distances.ExamplesDistance_Euclidean", 
           "orange.ExamplesDistance_Manhattan": "Orange.distances.ExamplesDistance_Manhattan", 
           "orange.ExamplesDistance_Maximal": "Orange.distances.ExamplesDistance_Maximal", 
           "orange.ExamplesDistance_Relief": "Orange.distances.ExamplesDistance_Relief", 
           "orange.ExamplesDistanceConstructor": "Orange.distances.ExamplesDistanceConstructor",
           "orange.ExamplesDistanceConstructor_DTW": "Orange.distances.ExamplesDistanceConstructor_DTW", 
           "orange.ExamplesDistanceConstructor_Euclidean": "Orange.distances.ExamplesDistanceConstructor_Euclidean", 
           "orange.ExamplesDistanceConstructor_Hamming": "Orange.distances.ExamplesDistanceConstructor_Hamming",
           "orange.ExamplesDistanceConstructor_Manhattan": "Orange.distances.ExamplesDistanceConstructor_Manhattan",
           "orange.ExamplesDistanceConstructor_Maximal": "Orange.distances.ExamplesDistanceConstructor_Maximal",
           "orange.ExamplesDistanceConstructor_Relief": "Orange.distances.ExamplesDistanceConstructor_Relief",
           
           "orngSVM.RBFKernelWrapper": "Orange.classification.svm.kernels.RBFKernelWrapper",
           "orngSVM.CompositeKernelWrapper": "Orange.classification.svm.kernels.CompositeKernelWrapper",
           "orngSVM.KernelWrapper": "Orange.classification.svm.kernels.KernelWrapper",
           "orngSVM.DualKernelWrapper": "Orange.classification.svm.kernels.DualKernelWrapper",
           "orngSVM.PolyKernelWrapper": "Orange.classification.svm.kernels.PolyKernelWrapper",
           "orngSVM.AdditionKernelWrapper": "Orange.classification.svm.kernels.AdditionKernelWrapper",
           "orngSVM.MultiplicationKernelWrapper": "Orange.classification.svm.kernels.MultiplicationKernelWrapper",
           "orngSVM.SparseLinKernel": "Orange.classification.svm.kernels.SparseLinKernel",
           "orngSVM.BagOfWords": "Orange.classification.svm.kernels.BagOfWords",
           
           }

def build_pattern(mapping=MAPPING):
    def split_dots(name):
        return " '.' ".join(["'%s'" % n for n in name.split(".")])
        
    names = "(" + "|".join("%s" % split_dots(key) for key in mapping.keys()) + ")"
    
    yield "power< bare_with_attr = (%s) trailer< any*> any*>"% names
    
def build_pattern(mapping=MAPPING):
    PATTERN = """
    power< head=any+
         trailer< '.' member=(%s) >
         tail=any*
    >
    """ 
    return PATTERN % "|".join("'%s'" % key.split(".")[-1] for key in mapping.keys())
    
class FixChangedNames(fixer_base.BaseFix):
    mapping = MAPPING 
    
    run_order = 1
    
    def compile_pattern(self):
        # We override this, so MAPPING can be pragmatically altered and the
        # changes will be reflected in PATTERN.
        self.PATTERN = build_pattern(self.mapping)
        self._modules_to_change = [key.split(".", 1)[0] for key in self.mapping.keys()]
        super(FixChangedNames, self).compile_pattern()
        
    def package_tree(self, package):
        """ Return pytree tree for asscesing the package
        
        Example:
            >>> package_tree("Orange.feature.scoring")
            [Name('Orange'), trailer('.' 'feature'), trailer('.', 'scoring')]
        """
        path = package.split('.')
        nodes = []
        nodes.append(Name(path[0]))
        for name in path[1:]:
            new = pytree.Node(self.syms.trailer, [Dot(), Name(name)])
            nodes.append(new)
        return nodes
        
        
    def transform(self, node, results):
        member = results.get("member")
        head = results.get("head")
        tail = results.get("tail")
        module = head[0].value
        
        if member and module in self._modules_to_change:
            node = member[0]
            head = head[0]
            tail = tail[0]
            
            new_name = unicode(self.mapping[module + "." + node.value])
            
            syms = self.syms
            
            tail = tail.clone()
            new = self.package_tree(new_name)
            new = pytree.Node(syms.power, new + [tail])
            
            # Make sure the proper package is imported
            package = new_name.rsplit(".", 1)[0]
            touch_import(None, package, node)
            return new    
    