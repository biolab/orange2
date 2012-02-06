""" This fixer changes all occurrences of the form 'module.member' from the
global dictionary MAPPING's keys and replaces them with the corresponding
value. It adds the proper imports to make it available in the script. 

For example this code::
    import orange
    import orngSVM
    data = orange.ExampleTable("iris")
    learner = orngSVM.SVMLearner(name='svm')
    
will be replaced with::
    import Orange.data
    import Orange.classification.svm
    data = Orange.data.Table('iris')
    learner = Orange.classification.svm.SVMLearner(name='svm')
    
Try to add as much name mappings as possible (This fixer is prefered 
(and will run before) the fix_orange_imports  
    
"""
from lib2to3 import fixer_base
from lib2to3 import fixer_util
from lib2to3 import pytree
from lib2to3.fixer_util import Name, Dot, Node, attr_chain, touch_import

# Keys must be in the form of 'orange.name' not name or orange.bla.name 
# If the values name a doted name inside of the package the package and name
# must be separated by ':' e.g. Orange.classification:Classifier.GetValue
# indicates Classifier.GetValue is a name inside package Orange.classification,
# do not use Orange.classification.Classifier.GetValue as this is assumed that 
# Orange.classification.Classifier is a package
# 
MAPPING = {"orange.ExampleTable": "Orange.data.Table",
           "orange.Example": "Orange.data.Instance",
           "orange.Domain": "Orange.data.Domain",
           "orange.Value": "Orange.data.Value",
           "orange.VarTypes": "Orange.data.Type",
           "orange.Variable": "Orange.data.variable.Variable",
           "orange.EnumVariable": "Orange.data.variable.Discrete",
           "orange.FloatVariable": "Orange.data.variable.Continuous",
           "orange.StringVariable": "Orange.data.variable.String",
           "orange.PythonVariable": "Orange.data.variable.Python",
           "orange.VarList": "Orange.data.variable.Variables",
           "orange.SymMatrix": "Orange.data.SymMatrix",
           "orange.GetValue": "Orange.classification:Classifier.GetValue",
           "orange.GetProbabilities": "Orange.classification:Classifier.GetProbabilities",
           "orange.GetBoth": "Orange.classification:Classifier.GetBoth",

           "orange.newmetaid": "Orange.data.new_meta_id",

           "orange.Distribution": "Orange.statistics.distribution.Distribution",
           "orange.DiscDistribution": "Orange.statistics.distribution.Discrete",
           "orange.ContDistribution": "Orange.statistics.distribution.Continuous",
           "orange.GaussianDistribution": "Orange.statistics.distribution.Gaussian",
           "orange.DomainDistributions": "Orange.statistics.distribution.Domain",

           "orange.BasicAttrStat": "Orange.statistics.basic.Variable",
           "orange.DomainBasicAttrStat": "Orange.statistics.basic.Domain",

           "orange.ContingencyAttrAttr": "Orange.statistics.contingency.VarVar",
           "orange.ContingencyClass": "Orange.statistics.contingency.Class",
           "orange.ContingencyAttrClass": "Orange.statistics.contingency.VarClass",
           "orange.ContingencyClassAttr": "Orange.statistics.contingency.ClassVar",
           "orange.DomainContingency": "Orange.statistics.contingency.Domain",
           "orange.Contingency": "Orange.statistics.contingency.Table",

           "orange.MeasureAttribute": "Orange.feature.scoring.Score",
           "orange.MeasureAttributeFromProbabilities": "Orange.feature.scoring.ScoreFromProbabilities",
           "orange.MeasureAttribute_gainRatio": "Orange.feature.scoring.GainRatio",
           "orange.MeasureAttribute_relief": "Orange.feature.scoring.Relief",
           "orange.MeasureAttribute_info": "Orange.feature.scoring.InfoGain",
           "orange.MeasureAttribute_gini": "Orange.feature.scoring.Gini",

           "orange.MeasureAttribute_relevance": "Orange.feature.scoring.Relevance",
           "orange.MeasureAttribute_cost": "Orange.feature.scoring.Cost",
           "orange.MeasureAttribute_MSE": "Orange.feature.scoring.MSE",

           "orngFSS.attMeasure": "Orange.feature.scoring.score_all",
           "orngFSS.bestNAtts": "Orange.feature.selection.best_n",
           "orngFSS.attsAbovethreshold": "Orange.feature.selection.above_threshold",
           "orngFSS.selectBestNAtts": "Orange.feature.selection.select_best_n",
           "orngFSS.selectAttsAboveThresh": "Orange.feature.selection.select_above_threshold",
           "orngFSS.filterRelieff": "Orange.feature.selection.select_relief",
           "orngFSS.FilterAttsAboveThresh": "Orange.feature.selection.FilterAboveThreshold",
           "orngFSS.FilterAttsAboveThresh_Class": "Orange.feature.selection.FilterAboveThreshold",
           "orngFSS.FilterBestNAtts": "Orange.feature.selection.FilterBestN",
           "orngFSS.FilterBestNAtts_Class": "Orange.feature.selection.FilterBestN",
           "orngFSS.FilterRelief": "Orange.feature.selection.FilterRelief",
           "orngFSS.FilterRelief_Class": "Orange.feature.selection.FilterRelief",
           "orngFSS.FilteredLearner": "Orange.feature.selection.FilteredLearner",
           "orngFSS.FilteredLearner_Class": "Orange.feature.selection.FilteredLearner",
           "orngFSS.FilteredClassifier": "Orange.feature.selection.FilteredClassifier",
           "orngFSS.StepwiseLearner_Class": "Orange.classification.wrappers.StepwiseLearner",
           "orngFSS.StepwiseLearner": "Orange.classification.wrappers.StepwiseLearner",

           "orange.ImputerConstructor_model": "Orange.feature.imputation.ImputerConstructor_model",
           "orange.Imputer_model": "Orange.feature.imputation.Imputer_model",
           "orange.Imputer_defaults": "Orange.feature.imputation.Imputer_defaults",
           "orange.ImputerConstructor_asValue": "Orange.feature.imputation.ImputerConstructor_asValue",
           "orange.ImputerConstructor_minimal": "Orange.feature.imputation.ImputerConstructor_minimal",
           "orange.ImputerConstructor_maximal": "Orange.feature.imputation.ImputerConstructor_maximal",
           "orange.ImputerConstructor_average": "Orange.feature.imputation.ImputerConstructor_average",

           "orange.ExamplesDistance": "Orange.distance.Distance",
           "orange.ExamplesDistance_Normalized": "Orange.distance.DistanceNormalized",
           "orange.ExamplesDistanceConstructor": "Orange.distance.DistanceConstructor",
           "orange.ExamplesDistance_Hamming": "Orange.distance.HammingDistance",
           "orange.ExamplesDistance_DTW": "Orange.distance.DTWDistance",
           "orange.ExamplesDistance_Euclidean": "Orange.distance.EuclideanDistance",
           "orange.ExamplesDistance_Manhattan": "Orange.distance.ManhattanDistance",
           "orange.ExamplesDistance_Maximal": "Orange.distance.MaximalDistance",
           "orange.ExamplesDistance_Relief": "Orange.distance.ReliefDistance",

           "orange.ExamplesDistanceConstructor_DTW": "Orange.distance.DTW",
           "orange.ExamplesDistanceConstructor_Euclidean": "Orange.distance.Euclidean",
           "orange.ExamplesDistanceConstructor_Hamming": "Orange.distance.Hamming",
           "orange.ExamplesDistanceConstructor_Manhattan": "Orange.distance.Manhattan",
           "orange.ExamplesDistanceConstructor_Maximal": "Orange.distance.Maximal",
           "orange.ExamplesDistanceConstructor_Relief": "Orange.distance.Relief",

           "orngClustering.ExamplesDistanceConstructor_PearsonR": "Orange.distance.PearsonR",
           "orngClustering.ExamplesDistance_PearsonR": "Orange.distance.PearsonRDistance",
           "orngClustering.ExamplesDistanceConstructor_SpearmanR": "Orange.distance.SpearmanR",
           "orngClustering.ExamplesDistance_SpearmanR": "Orange.distance.SpearmanRDistance",

           "orngClustering.KMeans": "Orange.clustering.kmeans.Clustering",
           "orngClustering.kmeans_init_random": "Orange.clustering.kmeans.init_random",
           "orngClustering.kmeans_init_diversity": "Orange.clustering.kmeans.init_diversity",
           "orngClustering.KMeans_init_hierarchicalClustering": "Orange.clustering.kmeans.init_hclustering",
           "orngClustering.data_center": "Orange.clustering.kmeans.data_center",
           "orngClustering.plot_silhouette": "Orange.clustering.kmeans.plot_silhouette",
           "orngClustering.score_distance_to_centroids": "Orange.clustering.kmeans.score_distance_to_centroids",
           "orngClustering.score_silhouette": "Orange.clustering.kmeans.score_silhouette",

           "orange.HierarchicalClustering": "Orange.clustering.hierarchical.HierarchicalClustering",
           "orange.HierarchicalCluster": "Orange.clustering.hierarchical.HierarchicalCluster",
           "orngClustering.hierarchicalClustering": "Orange.clustering.hierarchical.clustering",
           "orngClustering.hierarchicalClustering_attributes": "Orange.clustering.hierarchical.clustering_features",
           "orngClustering.hierarchicalClustering_clusterList": "Orange.clustering.hierarchical.cluster_to_list",
           "orngClustering.hierarchicalClustering_topClusters": "Orange.clustering.hierarchical.top_clusters",
           "orngClustering.hierarhicalClustering_topClustersMembership": "Orange.clustering.hierarchical.top_cluster_membership",
           "orngClustering.orderLeaves": "Orange.clustering.hierarchical.order_leaves",
           "orngClustering.dendrogram_draw": "Orange.clustering.hierarchical.dendrogram_draw",
           "orngClustering.DendrogramPlot": "Orange.clustering.hierarchical.DendrogramPlot",
           "orngClustering.DendrogramPlotPylab": "Orange.clustering.hierarchical.DendrogramPlotPylab",

           "orngSVM.RBFKernelWrapper": "Orange.classification.svm.kernels.RBFKernelWrapper",
           "orngSVM.CompositeKernelWrapper": "Orange.classification.svm.kernels.CompositeKernelWrapper",
           "orngSVM.KernelWrapper": "Orange.classification.svm.kernels.KernelWrapper",
           "orngSVM.DualKernelWrapper": "Orange.classification.svm.kernels.DualKernelWrapper",
           "orngSVM.PolyKernelWrapper": "Orange.classification.svm.kernels.PolyKernelWrapper",
           "orngSVM.AdditionKernelWrapper": "Orange.classification.svm.kernels.AdditionKernelWrapper",
           "orngSVM.MultiplicationKernelWrapper": "Orange.classification.svm.kernels.MultiplicationKernelWrapper",
           "orngSVM.SparseLinKernel": "Orange.classification.svm.kernels.SparseLinKernel",
           "orngSVM.BagOfWords": "Orange.classification.svm.kernels.BagOfWords",
           "orngSVM.SVMLearner": "Orange.classification.svm.SVMLearner",
           "orngSVM.SVMLearnerEasy": "Orange.classification.svm.SVMLearnerEasy",
           "orngSVM.SVMLearnerSparse": "Orange.classification.svm.SVMLearnerSparse",

           "orange.kNNLearner":"Orange.classification.knn.kNNLearner",
           "orange.kNNClassifier":"Orange.classification.knn.kNNClassifier",
           "orange.FindNearest_BruteForce":"Orange.classification.knn.FindNearest",
           "orange.FindNearestConstructor_BruteForce":"Orange.classification.knn.FindNearestConstructor",
           "orange.P2NN":"Orange.classification.knn.P2NN",

           "orange.BayesLearner":"Orange.classification.bayes.NaiveLearner",
           "orange.BayesClassifier":"Orange.classification.bayes.NaiveClassifier",
           "orngBayes.BayesLearner":"Orange.classification.bayes.NaiveLearner",
           "orngBayes.BayesClassifier":"Orange.classification.bayes.NaiveClassifier",
           "orngBayes.printModel": "Orange.classification.bayes.printModel",

           "orngNetwork.MdsTypeClass":"Orange.network.MdsTypeClass",
           "orngNetwork.Network":"Orange.network.Network",
           "orngNetwork.NetworkOptimization":"Orange.network.NetworkOptimization",
           "orngNetwork.NetworkClustering":"Orange.network.NetworkClustering",
           "orange.Graph":"Orange.network.Graph",
           "orange.GraphAsList":"Orange.network.GraphAsList",
           "orange.GraphAsMatrix":"Orange.network.GraphAsMatrix",
           "orange.GraphAsTree":"Orange.network.GraphAsTree",

           "orngEnsemble.MeasureAttribute_randomForests":"Orange.ensemble.forest.ScoreFeature",

           "orange.TreeLearner": "Orange.classification.tree.TreeLearner",
           "orange.TreeClassifier": "Orange.classification.tree.TreeClassifier",
           "orange.C45Learner": "Orange.classification.tree.C45Learner",
           "orange.C45Classifier": "Orange.classification.tree.C45Classifier",
           "orange.C45TreeNode": "Orange.classification.tree.C45Node",
           "orange.C45TreeNodeList": "Orange.classification.tree.C45NodeList",
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

           "orngTree.printTxt": "Orange.classification.tree:TreeClassifier.dump",
           "orngTree.printTree": "Orange.classification.tree:TreeClassifier.dump",
           "orngTree.dumpTree": "Orange.classification.tree:TreeClassifier.dump",
           "orngTree.printDot": "Orange.classification.tree:TreeClassifier.dot",
           "orngTree.dotTree": "Orange.classification.tree:TreeClassifier.dot",
           "orngTree.dump": "Orange.classification.tree:TreeClassifier.dump",
           "orngTree.dot": "Orange.classification.tree:TreeClassifier.dot",
           "orngTree.countLeaves": "Orange.classification.tree:TreeClassifier.count_leaves",
           "orngTree.countNodes": "Orange.classification.tree:TreeClassifier.count_nodes",
           "orngTree.byWhom": "Orange.classification.tree.by_whom",
           "orngTree.insertStr": "Orange.classification.tree.insert_str",
           "orngTree.insertDot": "Orange.classification.tree.insert_dot",
           "orngTree.insertNum": "Orange.classification.tree.insert_num",

           "orange.MajorityLearner":"Orange.classification.majority.MajorityLearner",
           "orange.DefaultClassifier":"Orange.classification.ConstantClassifier",

           "orange.LookupLearner":"Orange.classification.lookup.LookupLearner",
           "orange.ClassifierByLookupTable":"Orange.classification.lookup.ClassifierByLookupTable",
           "orange.ClassifierByLookupTable1":"Orange.classification.lookup.ClassifierByLookupTable1",
           "orange.ClassifierByLookupTable2":"Orange.classification.lookup.ClassifierByLookupTable2",
           "orange.ClassifierByLookupTable3":"Orange.classification.lookup.ClassifierByLookupTable3",
           "orange.ClassifierByExampleTable":"Orange.classification.lookup.ClassifierByDataTable",

           "orngLookup.lookupFromBound":"Orange.classification.lookup.lookup_from_bound",
           "orngLookup.lookupFromExamples":"Orange.classification.lookup.lookup_from_data",
           "orngLookup.lookupFromFunction":"Orange.classification.lookup.lookup_from_function",
           "orngLookup.printLookupFunction":"Orange.classification.lookup.dump_lookup_function",

           "orange.AssociationRule" : "Orange.associate.AssociationRule",
           "orange.AssociationRules" : "Orange.associate.AssociationRules",
           "orange.AssociationRulesInducer" : "Orange.associate.AssociationRulesInducer",
           "orange.AssociationRulesSparseInducer" : "Orange.associate.AssociationRulesSparseInducer",
           "orange.ItemsetNodeProxy" : "Orange.associate.ItemsetNodeProxy",
           "orange.ItemsetsSparseInducer" : "Orange.associate.ItemsetsSparseInducer",

           "orngCN2.ruleToString": "Orange.classification.rules.rule_to_string",
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
#           "orngCN2.avg": "Orange.classification.rules.avg",
#           "orngCN2.var": "Orange.classification.rules.var",
#           "orngCN2.median": "Orange.classification.rules.median",
#           "orngCN2.perc": "Orange.classification.rules.perc",
#           "orngCN2.createRandomDataSet": "Orange.classification.rules.createRandomDataSet",
#           "orngCN2.compParameters": "Orange.classification.rules.compParameters",
#           "orngCN2.computeDists": "Orange.classification.rules.computeDists",
#           "orngCN2.createEVDistList": "Orange.classification.rules.createEVDistList",
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
           "orngMDS.MDS": "Orange.projection.mds.MDS",

           "orngStat.AP" : "Orange.evaluation.scoring.AP",
           "orngStat.AROC" : "Orange.evaluation.scoring.AROC",
           "orngStat.AROCFromCDT" : "Orange.evaluation.scoring.AROC_from_CDT",
           "orngStat.AUC" : "Orange.evaluation.scoring.AUC",
           "orngStat.AUCWilcoxon" : "Orange.evaluation.scoring.AUCWilcoxon",
           "orngStat.AUC_binary" : "Orange.evaluation.scoring.AUC_binary",
           "orngStat.AUC_i" : "Orange.evaluation.scoring.AUC_i",
           "orngStat.AUC_ij" : "Orange.evaluation.scoring.AUC_ij",
           "orngStat.AUC_iterations" : "Orange.evaluation.scoring.AUC_iterations",
           "orngStat.AUC_matrix" : "Orange.evaluation.scoring.AUC_matrix",
           "orngStat.AUC_multi" : "Orange.evaluation.scoring.AUC_multi",
           "orngStat.AUC_pair" : "Orange.evaluation.scoring.AUC_pair",
           "orngStat.AUC_single" : "Orange.evaluation.scoring.AUC_single",
           "orngStat.AUC_x" : "Orange.evaluation.scoring.AUC_x",
           "orngStat.BSS" : "Orange.evaluation.scoring.BSS",
           "orngStat.BrierScore" : "Orange.evaluation.scoring.Brier_score",
           "orngStat.CA" : "Orange.evaluation.scoring.CA",
           "orngStat.CA_se" : "Orange.evaluation.scoring.CA_se",
           "orngStat.CDT" : "Orange.evaluation.scoring.CDT",
           "orngStat.ConfusionMatrix" : "Orange.evaluation.scoring.ConfusionMatrix",
           "orngStat.F1" : "Orange.evaluation.scoring.F1",
           "orngStat.Falpha" : "Orange.evaluation.scoring.Falpha",
           "orngStat.Friedman" : "Orange.evaluation.scoring.Friedman",
           "orngStat.IS" : "Orange.evaluation.scoring.IS",
           "orngStat.IS_ex" : "Orange.evaluation.scoring.IS_ex",
           "orngStat.MAE" : "Orange.evaluation.scoring.MAE",
           "orngStat.MCC" : "Orange.evaluation.scoring.MCC",
           "orngStat.ME" : "Orange.evaluation.scoring.ME",
           "orngStat.MSE" : "Orange.evaluation.scoring.MSE",
           "orngStat.MSE_old" : "Orange.evaluation.scoring.MSE_old",
           "orngStat.McNemar" : "Orange.evaluation.scoring.McNemar",
           "orngStat.McNemarOfTwo" : "Orange.evaluation.scoring.McNemar_of_two",
           "orngStat.NPV" : "Orange.evaluation.scoring.NPV",
           "orngStat.PPV" : "Orange.evaluation.scoring.PPV",
           "orngStat.R2" : "Orange.evaluation.scoring.R2",
           "orngStat.RAE" : "Orange.evaluation.scoring.RAE",
           "orngStat.RMSE" : "Orange.evaluation.scoring.RMSE",
           "orngStat.RMSE_old" : "Orange.evaluation.scoring.RMSE_old",
           "orngStat.ROCaddPoint" : "Orange.evaluation.scoring.ROC_add_point",
           "orngStat.ROCsFromCDT" : "Orange.evaluation.scoring.ROCs_from_CDT",
           "orngStat.ROCslope" : "Orange.evaluation.scoring.ROC_slope",
           "orngStat.RRSE" : "Orange.evaluation.scoring.RRSE",
           "orngStat.RSE" : "Orange.evaluation.scoring.RSE",
           "orngStat.TCbestThresholdsOnROCcurve" : "Orange.evaluation.scoring.TC_best_thresholds_on_ROC_curve",
           "orngStat.TCcomputeROC" : "Orange.evaluation.scoring.TC_compute_ROC",
           "orngStat.TCthresholdlAverageROC" : "Orange.evaluation.scoring.TC_threshold_average_ROC",
           "orngStat.TCverticalAverageROC" : "Orange.evaluation.scoring.TC_vertical_average_ROC",
           "orngStat.Wilcoxon" : "Orange.evaluation.scoring.Wilcoxon",
           "orngStat.WilcoxonPairs" : "Orange.evaluation.scoring.Wilcoxon_pairs",
#           "orngStat.add" : "Orange.evaluation.scoring.add",
#           "orngStat.checkArgkw" : "Orange.evaluation.scoring.checkArgkw",
#           "orngStat.checkNonZero" : "Orange.evaluation.scoring.checkNonZero",
           "orngStat.classProbabilitiesFromRes" : "Orange.evaluation.scoring.class_probabilities_from_res",
           "orngStat.compare2AROCs" : "Orange.evaluation.scoring.compare_2_AROCs",
           "orngStat.compare2AUCs" : "Orange.evaluation.scoring.compare_2_AUCs",
           "orngStat.computeCDT" : "Orange.evaluation.scoring.compute_CDT",
           "orngStat.computeCalibrationCurve" : "Orange.evaluation.scoring.compute_calibration_curve",
           "orngStat.computeConfusionMatrices" : "Orange.evaluation.scoring.compute_confusion_matrices",
           "orngStat.computeLiftCurve" : "Orange.evaluation.scoring.compute_lift_curve",
           "orngStat.computeROC" : "Orange.evaluation.scoring.compute_ROC",
           "orngStat.compute_CD" : "Orange.evaluation.scoring.compute_CD",
           "orngStat.compute_friedman" : "Orange.evaluation.scoring.compute_friedman",
           "orngStat.confusionChiSquare" : "Orange.evaluation.scoring.confusion_chi_square",
           "orngStat.confusionMatrices" : "Orange.evaluation.scoring.confusion_matrices",
           "orngStat.defaultLineTypes" : "Orange.evaluation.scoring.default_line_types",
           "orngStat.defaultPointTypes" : "Orange.evaluation.scoring.default_point_types",
#           "orngStat.frange" : "Orange.evaluation.scoring.frange",
#           "orngStat.gettotsize" : "Orange.evaluation.scoring.gettotsize",
#           "orngStat.gettotweight" : "Orange.evaluation.scoring.gettotweight",
           "orngStat.graph_ranks" : "Orange.evaluation.scoring.graph_ranks",
           "orngStat.isCDTEmpty" : "Orange.evaluation.scoring.is_CDT_empty",
           "orngStat.learningCurve2PiCTeX" : "Orange.evaluation.scoring.learning_curve_to_PiCTeX",
           "orngStat.learningCurveLearners2PiCTeX" : "Orange.evaluation.scoring.learning_curve_learners_to_PiCTeX",
           "orngStat.legend2PiCTeX" : "Orange.evaluation.scoring.legend_to_PiCTeX",
           "orngStat.legendLearners2PiCTeX" : "Orange.evaluation.scoring.legend_learners_to_PiCTeX",
#           "orngStat.log2" : "Orange.evaluation.scoring.log2",
#           "orngStat.math" : "Orange.evaluation.scoring.math",
#           "orngStat.numpy" : "Orange.evaluation.scoring.numpy",
#           "orngStat.operator" : "Orange.evaluation.scoring.operator",
           "orngStat.plotLearningCurve" : "Orange.evaluation.scoring.plot_learning_curve",
           "orngStat.plotLearningCurveLearners" : "Orange.evaluation.scoring.plot_learning_curve_learners",
           "orngStat.plotMcNemarCurve" : "Orange.evaluation.scoring.plot_McNemar_curve",
           "orngStat.plotMcNemarCurveLearners" : "Orange.evaluation.scoring.plot_McNemar_curve_learners",
           "orngStat.plotROC" : "Orange.evaluation.scoring.plot_ROC",
           "orngStat.plotROCLearners" : "Orange.evaluation.scoring.plot_ROC_learners",
           "orngStat.precision" : "Orange.evaluation.scoring.precision",
           "orngStat.printSingleROCCurveCoordinates" : "Orange.evaluation.scoring.print_single_ROC_curve_coordinates",
           "orngStat.rankDifference" : "Orange.evaluation.scoring.rank_difference",
           "orngStat.recall" : "Orange.evaluation.scoring.recall",
           "orngStat.regressionError" : "Orange.evaluation.scoring.regression_error",
           "orngStat.scottsPi" : "Orange.evaluation.scoring.scotts_pi",
           "orngStat.sens" : "Orange.evaluation.scoring.sens",
           "orngStat.spec" : "Orange.evaluation.scoring.spec",
           "orngStat.splitByIterations" : "Orange.evaluation.scoring.split_by_iterations",
#           "orngStat.statc" : "Orange.evaluation.scoring.statc",
           "orngStat.statisticsByFolds" : "Orange.evaluation.scoring.statistics_by_folds",
#           "orngStat.x" : "Orange.evaluation.scoring.x",

           # Orange.selection
           "orngMisc.BestOnTheFly":"Orange.misc.selection.BestOnTheFly",
           "orngMisc.selectBest":"Orange.misc.selection.select_best",
           "orngMisc.selectBestIndex":"Orange.misc.selection.select_best_index",
           "orngMisc.compare2_firstBigger":"Orange.misc.selection.compare_first_bigger",
           "orngMisc.compare2_firstBigger":"Orange.misc.selection.compare_first_bigger",
           "orngMisc.compare2_firstSmaller":"Orange.misc.selection.compare_first_smaller",
           "orngMisc.compare2_lastBigger":"Orange.misc.selection.compare_last_bigger",
           "orngMisc.compare2_lastSmaller":"Orange.misc.selection.compare_last_smaller",
           "orngMisc.compare2_bigger":"Orange.misc.selection.compare_bigger",
           "orngMisc.compare2_smaller":"Orange.misc.selection.compare_smaller",

           "orngMisc.Renderer": "Orange.misc.render.Renderer",
           "orngMisc.EPSRenderer": "Orange.misc.render.EPSRenderer",
           "orngMisc.SVGRenderer": "Orange.misc.render.SVGRenderer",
           "orngMisc.PILRenderer": "Orange.misc.render.PILRenderer",
           # The rest of orngMisc is handled by fix_orange_imports (maps to Orange.misc) 

           "orngEnsemble.BaggedLearner":"Orange.ensemble.bagging.BaggedLearner",
           "orngEnsemble.BaggedClassifier":"Orange.ensemble.bagging.BaggedClassifier",
           "orngEnsemble.BoostedLearner":"Orange.ensemble.boosting.BoostedLearner",
           "orngEnsemble.BoostedClassifier":"Orange.ensemble.boosting.BoostedClassifier",
           "orngEnsemble.RandomForestClassifier":"Orange.ensemble.forest.RandomForestClassifier",
           "orngEnsemble.RandomForestLearner":"Orange.ensemble.forest.RandomForestLearner",
           "orngEnsemble.MeasureAttribute_randomForests":"Orange.ensemble.forest.ScoreFeature",
           "orngEnsemble.SplitConstructor_AttributeSubset":"Orange.ensemble.forest.SplitConstructor_AttributeSubset",

           "orngTest.proportionTest":"Orange.evaluation.testing.proportion_test",
           "orngTest.leaveOneOut":"Orange.evaluation.testing.leave_one_out",
           "orngTest.crossValidation":"Orange.evaluation.testing.cross_validation",
           "orngTest.testWithIndices":"Orange.evaluation.testing.test_with_indices",
           "orngTest.learningCurve":"Orange.evaluation.testing.learning_curve",
           "orngTest.learningCurveN":"Orange.evaluation.testing.learning_curve_n",
           "orngTest.learningCurveWithTestData":"Orange.evaluation.testing.learning_curve_with_test_data",
           "orngTest.learnAndTestOnTestData":"Orange.evaluation.testing.learn_and_test_on_test_data",
           "orngTest.learnAndTestOnLearnData":"Orange.evaluation.testing.learn_and_test_on_learn_data",
           "orngTest.testOnData":"Orange.evaluation.testing.test_on_data",
           "orngTest.TestedExample":"Orange.evaluation.testing.TestedExample",
           "orngTest.ExperimentResults":"Orange.evaluation.testing.ExperimentResults",

           "orngLR.dump":"Orange.classification.logreg.dump",
           "orngLR.printOUT":"Orange.classification.logreg.dump",
           "orngLR.printOut":"Orange.classification.logreg.dump",
           "orngLR.hasDiscreteValues":"Orange.classification.logreg.has_discrete_values",
           "orngLR.LogRegLearner":"Orange.classification.logreg.LogRegLearner",
           "orngLR.LogRegLearnerClass":"Orange.classification.logreg.LogRegLearner",
           "orngLR.Univariate_LogRegLearner":"Orange.classification.logreg.UnivariateLogRegLearner",
           "orngLR.Univariate_LogRegLearner_Class":"Orange.classification.logreg.UnivariateLogRegLearner",
           "orngLR.Univariate_LogRegClassifier":"Orange.classification.logreg.UnivariateLogRegClassifier",
           "orngLR.LogRegLearner_getPriors":"Orange.classification.logreg.LogRegLearnerGetPriors",
           "orngLR.LogRegLearnerClass_getPriors":"Orange.classification.logreg.LogRegLearnerGetPriors",
           "orngLR.LogRegLearnerClass_getPriors_OneTable":"Orange.classification.logreg.LogRegLearnerGetPriorsOneTable",
           "orngLR.Pr":"Orange.classification.logreg.pr",
           "orngLR.lh":"Orange.classification.logreg.lh",
           "orngLR.diag":"Orange.classification.logreg.diag",
           "orngLR.simpleFitter":"Orange.classification.logreg.SimpleFitter",
           "orngLR.Pr_bx":"Orange.classification.logreg.pr_bx",
           "orngLR.bayesianFitter":"Orange.classification.logreg.BayesianFitter",
           "orngLR.StepWiseFSS":"Orange.classification.logreg.StepWiseFSS",
           "orngLR.getLikelihood":"Orange.classification.logreg.get_likelihood",
           "orngLR.StepWiseFSS_class":"Orange.classification.logreg.StepWiseFSS",
           "orngLR.StepWiseFSS_Filter":"Orange.classification.logreg.StepWiseFSSFilter",
           "orngLR.StepWiseFSS_Filter_class":"Orange.classification.logreg.StepWiseFSSFilter",
           "orngLR.lchisqprob":"Orange.classification.logreg.lchisqprob",
           "orngLR.zprob":"Orange.classification.logreg.zprob",

           "orange.Preprocessor": "Orange.preprocess.Preprocessor",
           "orange.Preprocessor_addCensorWeight": "Orange.preprocess.Preprocessor_addCensorWeight",
           "orange.Preprocessor_addClassNoise": "Orange.preprocess.Preprocessor_addClassNoise",
           "orange.Preprocessor_addClassWeight": "Orange.preprocess.Preprocessor_addClassWeight",
           "orange.Preprocessor_addGaussianClassNoise": "Orange.preprocess.Preprocessor_addGaussianClassNoise",
           "orange.Preprocessor_addGaussianNoise": "Orange.preprocess.Preprocessor_addGaussianNoise",
           "orange.Preprocessor_addMissing": "Orange.preprocess.Preprocessor_addMissing",
           "orange.Preprocessor_addMissingClasses": "Orange.preprocess.Preprocessor_addMissingClasses",
           "orange.Preprocessor_addNoise": "Orange.preprocess.Preprocessor_addNoise",
           "orange.Preprocessor_discretize": "Orange.preprocess.Preprocessor_discretize",
           "orange.Preprocessor_drop": "Orange.preprocess.Preprocessor_drop",
           "orange.Preprocessor_dropMissing": "Orange.preprocess.Preprocessor_dropMissing",
           "orange.Preprocessor_dropMissingClasses": "Orange.preprocess.Preprocessor_dropMissingClasses",
           "orange.Preprocessor_filter": "Orange.preprocess.Preprocessor_filter",
           "orange.Preprocessor_ignore": "Orange.preprocess.Preprocessor_ignore",
           "orange.Preprocessor_imputeByLearner": "Orange.preprocess.Preprocessor_imputeByLearner",
           "orange.Preprocessor_removeDuplicates": "Orange.preprocess.Preprocessor_removeDuplicates",
           "orange.Preprocessor_select": "Orange.preprocess.Preprocessor_select",
           "orange.Preprocessor_shuffle": "Orange.preprocess.Preprocessor_shuffle",
           "orange.Preprocessor_take": "Orange.preprocess.Preprocessor_take",
           "orange.Preprocessor_takeMissing": "Orange.preprocess.Preprocessor_takeMissing",
           "orange.Preprocessor_takeMissingClasses": "Orange.preprocess.Preprocessor_takeMissingClasses",

           "orange.Discretizer": "Orange.feature.discretization.Discretizer",
           "orange.BiModalDiscretizer": "Orange.feature.discretization.BiModalDiscretizer",
           "orange.EquiDistDiscretizer": "Orange.feature.discretization.EquiDistDiscretizer",
           "orange.IntervalDiscretizer": "Orange.feature.discretization.IntervalDiscretizer",
           "orange.ThresholdDiscretizer": "Orange.feature.discretization.ThresholdDiscretizer",
           "orange.EntropyDiscretization": "Orange.feature.discretization.EntropyDiscretization",
           "orange.Discrete2Continuous": "Orange.feature.discretization.Discrete2Continuous",

           "orange.DomainContinuizer": "Orange.feature.continuization.DomainContinuizer",

           "orange.MakeRandomIndices": "Orange.data.sample.SubsetIndices",
           "orange.MakeRandomIndicesN": "Orange.data.sample.SubsetIndicesN",
           "orange.MakeRandomIndicesCV": "Orange.data.sample.SubsetIndicesCV",
           "orange.MakeRandomIndicesMultiple": "Orange.data.sample.SubsetIndicesMultiple",
           "orange.MakeRandomIndices2": "Orange.data.sample.SubsetIndices2",

           "orngLinProj.FAST_IMPLEMENTATION": "Orange.projection.linear.FAST_IMPLEMENTATION",
           "orngLinProj.SLOW_IMPLEMENTATION": "Orange.projection.linear.SLOW_IMPLEMENTATION",
           "orngLinProj.LDA_IMPLEMENTATION": "Orange.projection.linear.LDA_IMPLEMENTATION",
           "orngLinProj.LAW_LINEAR": "Orange.projection.linear.LAW_LINEAR",
           "orngLinProj.LAW_SQUARE": "Orange.projection.linear.LAW_SQUARE",
           "orngLinProj.LAW_GAUSSIAN": "Orange.projection.linear.LAW_GAUSSIAN",
           "orngLinProj.LAW_KNN": "Orange.projection.linear.LAW_KNN",
           "orngLinProj.LAW_LINEAR_PLUS": "Orange.projection.linear.LAW_LINEAR_PLUS",
           "orngLinProj.DR_PCA": "Orange.projection.linear.DR_PCA",
           "orngLinProj.DR_SPCA": "Orange.projection.linear.DR_SPCA",
           "orngLinProj.DR_PLS": "Orange.projection.linear.DR_PLS",
           "orngLinProj.normalize": "Orange.projection.linear.normalize",
           "orngLinProj.center": "Orange.projection.linear.center",
           "orngLinProj.FreeViz": "Orange.projection.linear.FreeViz",
           "orngLinProj.createPLSProjection": "Orange.projection.linear.create_pls_projection",
           "orngLinProj.createPCAProjection": "Orange.projection.linear.create_pca_projection",
           "orngLinProj.FreeVizClassifier": "Orange.projection.linear.FreeVizClassifier",
           "orngLinProj.FreeVizLearner": "Orange.projection.linear.FreeVizLearner",
           "orngLinProj.S2NHeuristicLearner": "Orange.projection.linear.S2NHeuristicLearner",

           "orngDisc.entropyDiscretization": "Orange.feature.discretization.entropyDiscretization_wrapper",
           "orngDisc.EntropyDiscretization": "Orange.feature.discretization.EntropyDiscretization_wrapper",

           "orange.ProbabilityEstimator": "Orange.statistics.estimate.ProbabilityEstimator",
           "orange.ProbabilityEstimator_FromDistribution": "Orange.statistics.estimate.ProbabilityEstimator_FromDistribution",
           "orange.ProbabilityEstimatorConstructor": "Orange.statistics.estimate.ProbabilityEstimatorConstructor",
           "orange.ProbabilityEstimatorConstructor_Laplace": "Orange.statistics.estimate.ProbabilityEstimatorConstructor_Laplace",
           "orange.ProbabilityEstimatorConstructor_kernel": "Orange.statistics.estimate.ProbabilityEstimatorConstructor_kernel",
           "orange.ProbabilityEstimatorConstructor_loess": "Orange.statistics.estimate.ProbabilityEstimatorConstructor_loess",
           "orange.ProbabilityEstimatorConstructor_m": "Orange.statistics.estimate.ProbabilityEstimatorConstructor_m",
           "orange.ProbabilityEstimatorConstructor_relative": "Orange.statistics.estimate.ProbabilityEstimatorConstructor_relative",
           "orange.ProbabilityEstimatorList": "Orange.statistics.estimate.ProbabilityEstimatorList",

           "orange.FilterList": "Orange.preprocess.FilterList",
           "orange.Filter": "Orange.preprocess.Filter",
           "orange.Filter_conjunction": "Orange.preprocess.Filter_conjunction",
           "orange.Filter_disjunction": "Orange.preprocess.Filter_disjunction",
           "orange.Filter_hasClassValue": "Orange.preprocess.Filter_hasClassValue",
           "orange.Filter_hasMeta": "Orange.preprocess.Filter_hasMeta",
           "orange.Filter_hasSpecial": "Orange.preprocess.Filter_hasSpecial",
           "orange.Filter_isDefined": "Orange.preprocess.Filter_isDefined",
           "orange.Filter_random": "Orange.preprocess.Filter_random",
           "orange.Filter_sameValue": "Orange.preprocess.Filter_sameValue",
           "orange.Filter_values": "Orange.preprocess.Filter_values",

           # orngEnviron

           "orngEnviron.orangeDir": "Orange.misc.environ.install_dir",
           "orngEnviron.orangeDocDir": "Orange.misc.environ.doc_install_dir",
           "orngEnviron.orangeVer": "Orange.misc.environ.version",
           "orngEnviron.canvasDir": "Orange.misc.environ.canvas_install_dir",
           "orngEnviron.widgetDir": "Orange.misc.environ.widget_install_dir",
           "orngEnviron.picsDir": "Orange.misc.environ.icons_install_dir",
           "orngEnviron.addOnsDirSys": "Orange.misc.environ.add_ons_dir",
           "orngEnviron.addOnsDirUser": "Orange.misc.environ.add_ons_dir_user",
           "orngEnviron.applicationDir": "Orange.misc.environ.application_dir",
           "orngEnviron.outputDir": "Orange.misc.environ.output_dir",
           "orngEnviron.defaultReportsDir": "Orange.misc.environ.default_reports_dir",
           "orngEnviron.orangeSettingsDir": "Orange.misc.environ.orange_settings_dir",
           "orngEnviron.widgetSettingsDir": "Orange.misc.environ.widget_settings_dir",
           "orngEnviron.canvasSettingsDir": "Orange.misc.environ.canvas_settings_dir",
           "orngEnviron.bufferDir": "Orange.misc.environ.buffer_dir",
           "orngEnviron.directoryNames": "Orange.misc.environ.directories",
           "orngEnviron.samepath": "Orange.misc.environ.samepath",
           "orngEnviron.addOrangeDirectoriesToPath": "Orange.misc.environ.add_orange_directories_to_path",

           "orngScaleData.getVariableValuesSorted": "Orange.preprocess.scaling.get_variable_values_sorted",
           "orngScaleData.getVariableValueIndices": "Orange.preprocess.scaling.get_variable_value_indices",
           "orngScaleData.discretizeDomain": "Orange.preprocess.scaling.discretize_domain",
           "orngScaleData.orngScaleData": "Orange.preprocess.scaling.ScaleData",
           "orngScaleLinProjData.orngScaleLinProjData": "Orange.preprocess.scaling.ScaleLinProjData",
           "orngScalePolyvizData.orngScalePolyvizData": "Orange.preprocess.scaling.ScalePolyvizData",
           "orngScaleScatterPlotData.orngScaleScatterPlotData": "Orange.preprocess.scaling.ScaleScatterPlotData",

           "orngEvalAttr.mergeAttrValues": "Orange.feature.scoring.merge_values",
           "orngEvalAttr.MeasureAttribute_MDL": "Orange.feature.scoring.MDL",
           "orngEvalAttr.MeasureAttribute_MDLClass": "Orange.feature.scoring.MDL",
           "orngEvalAttr.MeasureAttribute_Distance": "Orange.feature.scoring.Distance",
           "orngEvalAttr.MeasureAttribute_DistanceClass": "Orange.feature.scoring.Distance",
           "orngEvalAttr.OrderAttributesByMeasure": "Orange.feature.scoring.OrderAttributes",

           "orange.ProbabilityEstimator": "Orange.statistics.estimate.Estimator",
           "orange.ProbabilityEstimator_FromDistribution": "Orange.statistics.estimate.EstimatorFromDistribution",
           "orange.ProbabilityEstimatorConstructor": "Orange.statistics.estimate.EstimatorConstructor",
           "orange.ProbabilityEstimatorConstructor_Laplace": "Orange.statistics.estimate.Laplace",
           "orange.ProbabilityEstimatorConstructor_kernel": "Orange.statistics.estimate.Kernel",
           "orange.ProbabilityEstimatorConstructor_loess": "Orange.statistics.estimate.Loess",
           "orange.ProbabilityEstimatorConstructor_m": "Orange.statistics.estimate.M",
           "orange.ProbabilityEstimatorConstructor_relative": "Orange.statistics.estimate.RelativeFrequency",
           "orange.ConditionalProbabilityEstimator": "Orange.statistics.estimate.ConditionalEstimator",
           "orange.ConditionalProbabilityEstimator_FromDistribution": "Orange.statistics.estimate.ConditionalEstimatorFromDistribution",
           "orange.ConditionalProbabilityEstimator_ByRows": "Orange.statistics.estimate.ConditionalEstimatorByRows",
           "orange.ConditionalProbabilityEstimatorConstructor_ByRows": "Orange.statistics.estimate.ConditionalByRows",
           "orange.ConditionalProbabilityEstimatorConstructor_loess": "Orange.statistics.estimate.ConditionalLoess",

           "orange.RandomGenerator": "Orange.misc.Random",

           }

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
        self.PATTERN = build_pattern(self.mapping)
        self._modules_to_change = [key.split(".", 1)[0] for key in self.mapping.keys()]
        super(FixChangedNames, self).compile_pattern()

    def package_tree(self, package):
        """ Return pytree tree for accessing the package
        
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

            old_name = module + "." + node.value
            if old_name not in self.mapping:
                return

            new_name = unicode(self.mapping[old_name])

            if ":" in new_name:
                # ':' is the delimiter used to separate module namespace
                package = new_name.split(":", 1)[0]
                new_name = new_name.replace(":", ".")
            else:
                package = new_name.rsplit(".", 1)[0]

            syms = self.syms

            if tail:
                tail = [t.clone() for t in  tail]
            new = self.package_tree(new_name)
            new = pytree.Node(syms.power, new + tail, prefix=head.prefix)

            # Make sure the proper package is imported
#            if ":" in new_name:
#                package = new_name.split(":",1)[0]
#            else:
#                package = new_name.rsplit(".", 1)[0]

            def orange_to_root(package):
                return "Orange" if package.startswith("Orange.") else package

            touch_import(None, orange_to_root(package), node)
            return new

