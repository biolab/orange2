""" This fixer changes all occurrences of the form 'module.member' from the
global dictionary MAPPING's keys and replaces them with the corresponding
value. It adds the proper imports to make it available in the script

For example this code::
    import orange
    import orngSVM
    data = orange.ExampleTable("iris")
    learner = orngSVM.SVMLearner(name='svm')
    
will be replaced with::
    import Orange.data
    import Orange.classification.svm
    data =Orange.data.Table('iris')
    learner =Orange.classification.svm.SVMLearner(name='svm')
    
Try to add as much name mappings as possible (This fixer is prefered 
(and will run before) the fix_orange_imports  
    
"""
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
           "orange.Variable": "Orange.data.feature.Feature",
           "orange.EnumVariable": "Orange.data.feature.Discrete",
           "orange.FloatVariable": "Orange.data.feature.Continuous",
           "orange.StringVariable": "Orange.data.feature.String",
           "orange.PythonVariable": "Orange.data.feature.Python",
           "orange.VarList": "Orange.data.feature.Features",
          
           "orange.MeasureAttribute": "Orange.feature.scoring.Measure", 
           "orange.MeasureAttribute_gainRatio": "Orange.feature.scoring.GainRatio",
           "orange.MeasureAttribute": "Orange.feature.scoring.Measure",
           "orange.MeasureAttribute_relief": "Orange.feature.scoring.Relief",
           "orange.MeasureAttribute_info": "Orange.feature.scoring.InfoGain",
           "orange.MeasureAttribute_gini": "Orange.feature.scoring.Gini",

           "orange.MeasureAttribute_relevance": "Orange.feature.scoring.Relevance",
           "orange.MeasureAttribute_cost": "Orange.feature.scoring.Cost",
           "orange.MeasureAttribute_MSE": "Orange.feature.scoring.MSE",

           "orngFSS.attMeasure": "Orange.feature.scoring.attMeasure",
           "orngFSS.bestNAtts": "Orange.feature.scoring.bestNAtts",
           "orngFSS.attsAbovethreshold": "Orange.feature.selection.attsAbovethreshold",
           "orngFSS.selectBestNAtts": "Orange.feature.selection.selectBestNAtts",
           "orngFSS.selectAttsAboveThresh": "Orange.feature.selection.selectAttsAboveThresh",
           "orngFSS.filterRelieff": "Orange.feature.selection.filterRelieff",
           "orngFSS.FilterAttsAboveThresh": "Orange.feature.selection.FilterAttsAboveThresh",
           "orngFSS.FilterAttsAboveThresh_Class": "Orange.feature.selection.FilterAttsAboveThresh_Class",
           "orngFSS.FilterBestNAtts": "Orange.feature.selection.FilterBestNAtts",
           "orngFSS.FilterBestNAtts_Class": "Orange.feature.selection.FilterBestNAtts_Class",
           "orngFSS.FilterRelief": "Orange.feature.selection.FilterRelief",
           "orngFSS.FilterRelief_Class": "Orange.feature.selection.FilterRelief_Class",
           "orngFSS.FilteredLearner": "Orange.feature.selection.FilteredLearner",
           "orngFSS.FilteredLearner_Class": "Orange.feature.selection.FilteredLearner_Class",
           "orngFSS.FilteredClassifier": "Orange.feature.selection.FilteredClassifier",
           "orngFSS.StepwiseLearner_Class": "Orange.classification.wrappers.StepwiseLearner_Class",
           "orngFSS.StepwiseLearner": "Orange.classification.wrappers.StepwiseLearner",
 
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
           
           "orange.kNNLearner":"Orange.classification.knn.kNNLearner",
           "orange.kNNClassifier":"Orange.classification.knn.kNNClassifier",
           "orange.FindNearest_BruteForce":"Orange.classification.knn.FindNearest",
           "orange.FindNearestConstructor_BruteForce":"Orange.classification.knn.FindNearestConstructor",
           "orange.P2NN":"Orange.classification.knn.P2NN",
           
            "orange.TreeLearner": "Orange.classification.tree.TreeLearnerBase",
           "orange.TreeClassifier": "Orange.classification.tree.TreeClassifier",
           "orange.C45Learner": "Orange.classification.tree.C45Learner",
           "orange.C45Classifier": "Orange.classification.tree.C45Classifier",
           "orange.C45TreeNode": "Orange.classification.tree.C45Node",
           "orange.C45TreeNodeList": "C45NodeList",
           "orange.TreeDescender": "Orange.classification.tree.Descender",
           "orange.TreeDescender_UnknownMergeAsBranchSizes": "Orange.classification.tree.Descender_UnknownMergeAsBranchSizes",
           "orange.TreeDescender_UnknownMergeAsSelector": "Orange.classification.tree.Descender_UnknownMergeAsSelector",
           "orange.TreeDescender_UnknownToBranch": "Orange.classification.tree.Descender_UnknownToBranch",
           "orange.TreeDescender_UnknownToCommonBranch": "Orange.classification.tree.Descender_UnknownToCommonBranch",
           "orange.TreeDescender_UnknownToCommonSelector":"Orange.classification.tree.Descender_UnknownToCommonSelector",
           "orange.TreeExampleSplitter":"Orange.classification.tree.Splitter",
           "orange.TreeExampleSplitter_IgnoreUnknowns":"Orange.classification.tree.Splitter_IgnoreUnknowns",
           "orange.TreeExampleSplitter_UnknownsAsBranchSizes":"Orange.classification.tree.Splitter_UnknownsAsBranchSizes",
           "orange.TreeExampleSplitter_UnknownsAsSelecto":"Orange.classification.tree.Splitter_UnknownsAsSelector",
           "orange.TreeExampleSplitter_UnknownsToAll":"Orange.classification.tree.Splitter_UnknownsToAll",
           "orange.TreeExampleSplitter_UnknownsToBranch":"Orange.classification.tree.Splitter_UnknownsToBranch",
           "orange.TreeExampleSplitter_UnknownsToCommon":"Orange.classification.tree.Splitter_UnknownsToCommon",
           "orange.TreeExampleSplitter_UnknownsToRandom":"Orange.classification.tree.Splitter_UnknownsToRandom",
           "orange.TreeNode":"Orange.classification.tree.Node",
           "orange.TreeNodeList":"Orange.classification.tree.NodeList",
           "orange.TreePruner":"Orange.classification.tree.Pruner",
           "orange.TreePruner_SameMajority":"Orange.classification.tree.Pruner_SameMajority",
           "orange.TreePruner_m":"Orange.classification.tree.Pruner_m",
           "orange.TreeSplitConstructor":"Orange.classification.tree.SplitConstructor",
           "orange.TreeSplitConstructor_Combined":"Orange.classification.tree.SplitConstructor_Combined",
           "orange.TreeSplitConstructor_Measure":"Orange.classification.tree.SplitConstructor_Score",
           "orange.TreeSplitConstructor_Attribute":"Orange.classification.tree.SplitConstructor_Feature",
           "orange.TreeSplitConstructor_ExhaustiveBinary":"Orange.classification.tree.SplitConstructor_ExhaustiveBinary",
           "orange.TreeSplitConstructor_OneAgainstOthers":"Orange.classification.tree.SplitConstructor_OneAgainstOthers",
           "orange.TreeSplitConstructor_Threshold":"Orange.classification.tree.SplitConstructor_Threshold",
           "orange.TreeStopCriteria":"Orange.classification.tree.StopCriteria",
           "orange.TreeStopCriteria_Python":"Orange.classification.tree.StopCriteria_Python",
           "orange.TreeStopCriteria_common":"Orange.classification.tree.StopCriteria_common",

           "orngCN2.ruleToString": "Orange.classification.rules.ruleToString",
           "orngCN2.LaplaceEvaluator": "Orange.classification.rules.LaplaceEvaluator",
           "orngCN2.WRACCEvaluator": "Orange.classification.rules.WRACCEvaluator",
           "orngCN2.mEstimate": "Orange.classification.rules.MEstimateEvaluator",
           "orngCN2.RuleStopping_apriori": "Orange.classification.rules.RuleStopping_Apriori",
           "orngCN2.LengthValidator": "Orange.classification.rules.LengthValidator",
           "orngCN2.supervisedClassCheck": "Orange.classification.rules.supervisedClassCheck",
           "orngCN2.CN2Learner": "Orange.classification.rules.CN2Learner",
           "orngCN2.CN2Classifier": "Orange.classification.rules.CN2Classifier",
           "orngCN2.CN2UnorderedLearner": "Orange.classification.rules.CN2UnorderedLearner",
           "orngCN2.CN2UnorderedClassifier": "Orange.classification.rules.CN2UnorderedClassifier",
           "orngCN2.RuleClassifier_bestRule": "Orange.classification.rules.RuleClassifier_BestRule",
           "orngCN2.CovererAndRemover_multWeights": "Orange.classification.rules.CovererAndRemover_MultWeights",
           "orngCN2.CovererAndRemover_addWeights": "Orange.classification.rules.CovererAndRemover_AddWeights",
           "orngCN2.rule_in_set": "Orange.classification.rules.rule_in_set",
           "orngCN2.rules_equal": "Orange.classification.rules.rules_equal",
           "orngCN2.noDuplicates_validator": "Orange.classification.rules.NoDuplicatesValidator",
           "orngCN2.ruleSt_setRules": "Orange.classification.rules.RuleStopping_SetRules",
           "orngCN2.CN2SDUnorderedLearner": "Orange.classification.rules.CN2SDUnorderedLearner",
           "orngCN2.avg": "Orange.classification.rules.avg",
           "orngCN2.var": "Orange.classification.rules.var",
           "orngCN2.median": "Orange.classification.rules.median",
           "orngCN2.perc": "Orange.classification.rules.perc",
           "orngCN2.createRandomDataSet": "Orange.classification.rules.createRandomDataSet",
           "orngCN2.compParameters": "Orange.classification.rules.compParameters",
           "orngCN2.computeDists": "Orange.classification.rules.computeDists",
           "orngCN2.createEVDistList": "Orange.classification.rules.createEVDistList",
           "orngCN2.CovererAndRemover_Prob": "Orange.classification.rules.CovererAndRemover_Prob",
           "orngCN2.add_sub_rules": "Orange.classification.rules.add_sub_rules",
           "orngCN2.CN2EVCUnorderedLearner": "Orange.classification.rules.CN2EVCUnorderedLearner",
           
           "orngMDS.KruskalStress": "Orange.projection.mds.KruskalStress",
           "orngMDS.SammonStress": "Orange.projection.mds.SammonStress",
           "orngMDS.SgnSammonStress": "Orange.projection.mds.SgnSammonStress",
           "orngMDS.SgnRelStress": "Orange.projection.mds.SgnRelStress",
           "orngMDS.PointList": "Orange.projection.mds.PointList",
           "orngMDS.FloatListList": "Orange.projection.mds.FloatListList",
           "orngMDS.PivotMDS": "Orange.projection.mds.PivotMDS",
           "orngMDS.MDS": "Orange.projection.mds.MDS"

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
            
            new_name = unicode(self.mapping[module + "." + node.value])
            
            syms = self.syms
            
            if tail:
                tail = [t.clone() for t in  tail]
            new = self.package_tree(new_name)
            new = pytree.Node(syms.power, new + tail, prefix=head.prefix)
            
            # Make sure the proper package is imported
            package = new_name.rsplit(".", 1)[0]
            touch_import(None, package, node)
            return new    
    
