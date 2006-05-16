#if !defined(FRONTEND_H)
#define FRONTEND_H

enum demandType { none=0,
                  leafDistribution=1,
                  constructs=2, distributionAndConstructs=3,
                  priorDiscretization=4, priorDiscretizationOut = 5,
                  residuals=6, avReliefF=7, avRF=8, ordEval=9,
				  ordEval3cl=10
} ;
void mainMenu(void) ;
void singleEstimation(featureTree* const Tree) ;
void allSplitsEstimation(featureTree* const Tree) ;
void singleTree(featureTree* const Tree) ;
void allSplitsTree(featureTree* const Tree) ;
void singleRF(featureTree* const Tree) ;
void allSplitsRF(featureTree* const Tree) ;
void domainCharacteristics(featureTree* const Tree);
void outVersion(FILE *fout) ; 
FILE* prepareDistrFile(int fileIdx) ;
void evalAttrVal(featureTree*  Tree, demandType demand) ;
void evalOrdAttrVal(featureTree*  Tree, demandType demand)  ;

#endif