from Orange.evaluation.testing import \
    encode_PP as encodePP,\
    TestedExample,\
    ExperimentResults,\
    leave_one_out as leaveOneOut,\
    proportion_test as proportionTest,\
    cross_validation as crossValidation,\
    learning_curve_n as learningCurveN,\
    learning_curve as learningCurve,\
    learning_curve_with_test_data as learningCurveWithTestData,\
    test_with_indices as testWithIndices,\
    learn_and_test_on_test_data as learnAndTestOnTestData,\
    learn_and_test_on_learn_data as learnAndTestOnLearnData,\
    test_on_data as testOnData
    

'''
RENAMED METHODS:

TestedExample.addResult -> TestedExample.add_result
TestedExample.setResult -> TestedExample.set_result
ExperimentResults.loadFromFiles -> ExperimentResults.load_from_files
ExperimentResults.saveToFiles -> ExperimentResults.save_to_files
'''
