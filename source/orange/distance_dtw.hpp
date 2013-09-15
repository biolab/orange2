#include "orvector.hpp"
#include <vector>

#include "distance.hpp"

using namespace std; 
 
class ORANGE_API TAlignment
{
public:
	int i;
	int j;

    TAlignment();
    TAlignment(int i, int j);
    TAlignment(const TAlignment &);

    bool operator==(const TAlignment &) const;
    bool operator<(const TAlignment &) const;
};

#define TWarpPath TOrangeVector<TAlignment, false>
VWRAPPER(WarpPath)

class TdtwElement;
typedef vector<TdtwElement> TdtwVector;
typedef vector<TdtwElement*> PdtwVector;
typedef vector<TdtwVector> TdtwMatrix;

#define TAlignmentList TWarpPath
#define PAlignmentList PWarpPath


class ORANGE_API TExamplesDistance_DTW : public TExamplesDistance_Normalized
{
public:
    __REGISTER_CLASS

  CLASSCONSTANTS(DistanceType: Euclidean; Derivative)
	enum { DTW_EUCLIDEAN, DTW_DERIVATIVE };
	
	int dtwDistance; //P(&ExamplesDistance_DTW_DistanceType) distance measure between individual attributes (default: square of difference)
		
	TExamplesDistance_DTW();
    TExamplesDistance_DTW(const int &distance, const bool &normalize, const bool &ignoreClass, PExampleGenerator egen, PDomainDistributions ddist, PDomainBasicAttrStat dstat);
  
    virtual float operator()(const TExample &, const TExample &) const;
    virtual float operator()(const TExample &, const TExample &, PWarpPath &) const;

//private:
	void getDerivatives(vector<float> &seq1, vector<float> &der1) const;
	void initMatrix(const vector<float> &seq1, const vector<float> &seq2, TdtwMatrix &mtrx) const;
	float calcDistance(TdtwMatrix &mtrx) const;
	PWarpPath setWarpPath(const TdtwMatrix &mtrx) const;
	void printMatrix(const TdtwMatrix &mtrx) const;
};


class ORANGE_API TExamplesDistanceConstructor_DTW : public TExamplesDistanceConstructor_Normalized {
public:
  __REGISTER_CLASS

  int dtwDistance; //P distance measure between individual attributes (default: square of difference)

  TExamplesDistanceConstructor_DTW();
  TExamplesDistanceConstructor_DTW(PExampleGenerator);
  virtual PExamplesDistance operator()(PExampleGenerator, const int & = 0, PDomainDistributions = PDomainDistributions(), PDomainBasicAttrStat = PDomainBasicAttrStat()) const;
};
