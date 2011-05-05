from Orange.evaluation.scoring import *

from Orange.evaluation.scoring import \
    check_non_zero as checkNonZero, \
    split_by_iterations as splitByIterations, \
    class_probabilities_from_res as classProbabilitiesFromRes, \
    statistics_by_folds as statisticsByFolds, \
    check_argkw as checkArgkw, \
    regression_error as regressionError, \
    Brier_score as BrierScore, \
    rank_difference as rankDifference, \
    confusion_matrices as confusionMatrices, \
    confusion_chi_square as confusionChiSquare, \
    compare_2_AUCs as compare2AUCs, \
    compare_2_AROCs as compare2AROCs, \
    compute_ROC as computeROC, \
    ROC_slope as ROCslope, \
    ROC_add_point as ROCaddPoint, \
    TC_compute_ROC as TCcomputeROC, \
    TC_best_thresholds_on_ROC_curve as TCbestThresholdsOnROCcurve, \
    TC_vertical_average_ROC as TCverticalAverageROC, \
    TC_threshold_average_ROC as TCthresholdlAverageROC, \
    compute_calibration_curve as computeCalibrationCurve, \
    compute_lift_curve as computeLiftCurve, \
    is_CDT_empty as isCDTempty, \
    compute_CDT as computeCDT, \
    ROCs_from_CDT as ROCsFromCDT, \
    AROC_from_CDT as AROCFromCDT, \
    McNemar_of_two as McNemarOfTwo, \
    Wilcoxon_pairs as WilcoxonPairs, \
    plot_learning_curve_learners as plotLearningCurveLearners, \
    plot_learning_curve as plotLearningCurve, \
    print_single_ROC_curve_coordinates as printSingleROCCurveCoordinates, \
    plot_ROC_learners as plotROCLearners, \
    plot_ROC as plotROC, \
    plot_McNemar_curve_learners as plotMcNemarCurveLearners, \
    plot_McNemar_curve as plotMcNemarCurve, \
    learning_curve_learners_to_PiCTeX as learningCurveLearners2PiCTeX, \
    learning_curve_to_PiCTeX as learningCurve2PiCTeX, \
    legend_learners_to_PiCTeX as legendLearners2PiCTeX, \
    legend_to_PiCTeX as legend2PiCTeX


# obsolete (renamed)
computeConfusionMatrices = confusionMatrices