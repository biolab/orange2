#include "orvector.hpp"
#include <vector>

#include "distance.hpp"

using namespace std; 
 
class TAlignment
{
public:
	int i;
	int j;

    TAlignment();
	TAlignment(int i, int j);
    TAlignment(const TAlignment &);

    bool operator==(const TAlignment &);
    bool operator<(const TAlignment &);
};

#define TWarpPath _TOrangeVector<TAlignment>
VWRAPPER(WarpPath)

class TdtwElement;
typedef vector<TdtwElement> TdtwVector;
typedef vector<TdtwElement*> TPdtwVector;
typedef vector<TdtwVector> TdtwMatrix;

class TExamplesDistance_DTW : public TExamplesDistance_Normalized
{
public:
    __REGISTER_CLASS

	TExamplesDistance_DTW();
    TExamplesDistance_DTW(const bool &ignoreClass, PExampleGenerator egen, PDomainDistributions ddist, PDomainBasicAttrStat dstat);
  
    virtual float operator()(const TExample &, const TExample &) const;
    virtual float operator()(const TExample &, const TExample &, PWarpPath &) const;

private:
	void initMatrix(const vector<float> &seq1, const vector<float> &seq2, vector<TdtwVector> &mtrx) const;
	float calcDistance(vector<TdtwVector> &mtrx) const;
	PWarpPath setWarpPath(const vector<TdtwVector> &mtrx) const;
	void printMatrix(const TdtwMatrix &mtrx) const;
};


class TExamplesDistanceConstructor_DTW : public TExamplesDistanceConstructor {
public:
  __REGISTER_CLASS

  TExamplesDistanceConstructor_DTW();
  TExamplesDistanceConstructor_DTW(PExampleGenerator);
  virtual PExamplesDistance operator()(PExampleGenerator, const int & = 0, PDomainDistributions = PDomainDistributions(), PDomainBasicAttrStat = PDomainBasicAttrStat()) const;
};
