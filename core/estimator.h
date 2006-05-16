#if !defined(ESTIMATOR_H)
#define ESTIMATOR_H

#include "ftree.h"
#include "contain.h"
#include "utils.h"

const int constNAdiscretizationIntervals = 5 ;
const int constAverageExamplesPerInterval = 5 ;
const int maxVal4ExhDisc = 32 ; // maximum number of values of attribute for considering exhaustive search for discretization


class estimation
{
friend void singleEstimation(featureTree* const Tree) ;
friend void allSplitsEstimation(featureTree* const Tree) ;
friend void domainCharacteristics(featureTree* const Tree) ;
friend class featureTree ;
friend class construct ;
friend class expr ;
friend class rf ;

    featureTree *fTree ;
    mmatrix<int> DiscValues ;
    mmatrix<double> ContValues ;
    marray<double> ContEstimation, DiscEstimation ;
//    marray<double> contDiffA, discDiffA ;
//    double diffC ;
    marray<double> weight;
    mmatrix<marray<double> > NAdiscValue, NAcontValue ;
    marray<double> minValue, maxValue, valueInterval, step ;
    mmatrix<double> ContDistance, DiscDistance ;
    marray<int> discNoValues ;
    marray<marray<sortRec> > distanceArray, diffSorted ; // manipulation of the nearest examples
    marray<sortRec> distanceRarray, diffRsorted ; // R variant: manipulation of the nearest examples
    marray<sortRec> distanceEHarray, diffEHsorted,
                    distanceEMarray, diffEMsorted ; // E variant: manipulation of the nearest examples

    int NoDiscrete, NoContinuous, TrainSize ;
    int currentContSize, currentDiscSize, discUpper, contUpper ;
    int NoIterations ;
    int kNearestEqual, kDensity ;
    double varianceDistanceDensity ;
    int NoClasses ;
    int noNAdiscretizationIntervals ;
 
    void initialize(marray<int> &inDTrain, marray<double> &inpDTrain, int inTrainSize) ;
    double CAdiff(int AttrNo, int I1, int I2) ;
    double DAdiff(int AttrNo, int I1, int I2) ;
    void prepareDistanceFactors(int current, int distanceType) ;
    void RprepareDistanceFactors(int current, int distanceType) ; // for E variant of Relief
    void EprepareDistanceFactors(int current, int distanceType) ;
    void computeDistances(int Example) ;
    double CaseDistance(int I1) ;
    void findHitMiss(int current, int &hit, int &miss) ;
    inline double NAcontDiff(int AttrIdx, int ClassValue, double Value) ;
    void stratifiedExpCostSample(marray<int> &sampleIdx, int sampleSize, int domainSize, marray<double> &probClass, marray<int> &noExInClass) ;
	void computeDistancesOrd(int Example)  ;
	inline double DAdiffOrd(int AttrIdx, int I1, int I2)  ;
	inline double DAdiffSign(int AttrIdx, int I1, int I2)  ;
    void prepare3clDistanceFactors(int current, int distanceType) ;


#ifdef RAMP_FUNCTION
    marray<double> DifferentDistance, EqualDistance, CAslope ;
    inline double CARamp(int AttrIdx, double distance) ;
#endif


public:
    // marray<int> OriginalDTrain ;
    estimation(featureTree *fTreeParent, marray<int> &DTrain,
                marray<double> &pDTrain, int TrainSize) ;
    ~estimation() { }
    void destroy(void) ;
    int estimate(int selectedEstimator, int contAttrFrom, int contAttrTo, 
                         int discAttrFrom, int discAttrTo, attributeCount &bestType) ;
    int estimateConstruct(int selectedEstimator, int contAttrFrom, int contAttrTo, int discAttrFrom, int discAttrTo, attributeCount &bestType, marray<construct> &DiscConstruct, marray<construct> &ContConstruct) ;
	int estimateSelected(int selectedEstimator, marray<boolean> &mask, attributeCount &bestType) ;
    void ReliefF(int contAttrFrom, int contAttrTo, int discAttrFrom, int discAttrTo, int distanceType) ;
    void ReliefFbestK(int contAttrFrom, int contAttrTo, int discAttrFrom, int discAttrTo, int distanceType) ;
    void Relief(int contAttrFrom, int contAttrTo, int discAttrFrom, int discAttrTo) ;
    void ReliefFmerit(int contAttrFrom, int contAttrTo, int discAttrFrom, int discAttrTo, int distanceType) ;
    void ReliefFavgC(int contAttrFrom, int contAttrTo, int discAttrFrom, int discAttrTo, int distanceType) ;
    void ReliefFexpC(int contAttrFrom, int contAttrTo, int discAttrFrom, int discAttrTo, int distanceType) ;
    void ReliefFpa(int contAttrFrom, int contAttrTo, int discAttrFrom, int discAttrTo, int distanceType) ;
    void ReliefFpe(int contAttrFrom, int contAttrTo, int discAttrFrom, int discAttrTo, int distanceType) ;
    void ReliefFsmp(int contAttrFrom, int contAttrTo, int discAttrFrom, int discAttrTo, int distanceType) ;
    void ReliefRcost(int contAttrFrom, int contAttrTo, int discAttrFrom, int discAttrTo, int distanceType) ;
    void ReliefEcost(int contAttrFrom, int contAttrTo, int discAttrFrom, int discAttrTo, int distanceType) ;
    void ReliefFcostKukar(int contAttrFrom, int contAttrTo, int discAttrFrom, int discAttrTo) ;
    void infGain(int discAttrFrom, int discAttrTo) ;
    void gainRatio(int discAttrFrom, int discAttrTo) ;
    void mdl(int discAttrFrom, int discAttrTo) ;
    void MDLsmp(int discAttrFrom, int discAttrTo) ;
    void ReliefMyopic(int discAttrFrom, int discAttrTo) ;
    void Accuracy(int discAttrFrom, int discAttrTo) ;
    double binAccEst(mmatrix<int> &noClassAttrVal, int noValues) ;
    void BinAccuracy(int discAttrFrom, int discAttrTo) ;
    void Gini(int discAttrFrom, int discAttrTo) ;
    void DKM(int discAttrFrom, int discAttrTo) ;
    void DKMc(int discAttrFrom, int discAttrTo) ;
    void gainRatioC(int discAttrFrom, int discAttrTo) ;
	void aVReliefF(int discAttrFrom, int discAttrTo, marray<marray<double> > &result, int distanceType) ;
    void adjustTables(int newContSize, int newDiscSize) ;
    void prepareContAttr(int attrIdx) ;
    void prepareDiscAttr(int attrIdx, int noValues) ;
 	void binarizeGeneral(construct &nodeConstruct, double &bestEstimation, int firstFreeDiscSlot) ;
    double bestSplitGeneral(construct &nodeConstruct, double &bestEstimation, int firstFreeDiscSlot) ;
    double discretizeGreedy(int ContAttrIdx, marray<double> &Bounds, int firstFreeDiscSlot) ;
	void estBinarized(int selectedEstimator, int contAttrFrom, int contAttrTo, int discAttrFrom, int discAttrTo, int firstFreeDiscSlot) ;
    double CVVilalta(int contAttrFrom, int contAttrTo, int discAttrFrom, int discAttrTo) ;
    double CVmodified(int contAttrFrom, int contAttrTo, int discAttrFrom, int discAttrTo) ;
    void discretizeEqualFrequency(int ContAttrIdx, int noIntervals, marray<double> &Bounds) ;
    void ordAvReliefF(int discAttrFrom, int discAttrTo, 
	        marray<marray<double> > &resultCpAp, marray<marray<double> > &resultCpAn,
			marray<marray<double> > &resultCpAe, marray<marray<double> > &resultCnAp,
			marray<marray<double> > &resultCnAn, marray<marray<double> > &resultCnAe, 
			marray<marray<double> > &resultCeAp, marray<marray<double> > &resultCeAn, 
			marray<marray<double> > &resultCeAe, int distanceType) ;
	void ordAV3clReliefF(int discAttrFrom, int discAttrTo, 
	        marray<marray<double> > &resultCpAp, marray<marray<double> > &resultCpAn,
			marray<marray<double> > &resultCpAe, marray<marray<double> > &resultCnAp, 
			marray<marray<double> > &resultCnAn, marray<marray<double> > &resultCnAe,
			marray<marray<double> > &resultCeAp, marray<marray<double> > &resultCeAn, 
			marray<marray<double> > &resultCeAe, int distanceType) ;
}  ;

#endif
