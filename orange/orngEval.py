from orngTest import *
from orngStat import *

#### Compatibility stuff
LeaveOneOut = leaveOneOut
CrossValidation = crossValidation
PercentTest = proportionTest
LearningCurveCrossValidation = learningCurveN
LearningCurve = learningCurve
LearningCurveWithTestData = learningCurveWithTestData
TestWithIndices = testWithIndices
TestWithTestData = testOnData
LearnAndTestWithTestData = learnAndTestOnTestData


CA_dev = CA_se
CA2 = CA
computeAROC = AROC
aROC = AROCFromCDT

def TFPosNeg(res, cutoff=0.5, classIndex=1):
    return computeConfusionMatrices(res, classIndex, cutoff=0.5)

def print_aROC(res):
    print "Concordant  = %5.1f       Somers' D = %1.3f" % (res[0], res[4])
    print "Discordant  = %5.1f       Gamma     = %1.3f" % (res[1], res[5]>0 and res[5] or "N/A")
    print "Tied        = %5.1f       Tau-a     = %1.3f" % (res[2], res[6])
    print " %6d pairs             c         = %1.3f"    % (res[3], res[7])
